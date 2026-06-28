"""
V11 KG-CTCN Alert Orchestration Engine
--------------------------------------------------
Prevents alert fatigue by clustering events.
Rule: Only trigger an active outbound alert if High Risk is
sustained for >= 2 consecutive PREDICTION days for a given location.

CHANGELOG (v11.1):
  - [BUG]     init_alert_db() now creates the predictions table with
              CREATE TABLE IF NOT EXISTS. Previously it committed an empty
              transaction and left the table uncreated, causing
              OperationalError: no such table on the first evaluate call.
  - [BUG]     Connection leak fixed. evaluate_alert_state() previously had
              no exception safety — if the query raised, conn.close() was
              never called. All DB access now uses contextlib.closing.
  - [BUG]     Consecutive-day rule now checks prediction_date, not
              target_date. target_date is the future outbreak window
              (t+3 to t+7); checking consecutive target_dates does not
              verify that the model issued alerts on consecutive calendar
              days. The correct field is prediction_date (the date the
              model ran and produced the score).
  - [BUG]     WAL journal mode enabled at connection time. SQLite's default
              DELETE mode takes an exclusive write lock; concurrent reads
              from inference workers block or raise "database is locked".
              WAL allows concurrent readers with a single writer.
  - [MINOR]   Renamed 'timestamp' column to 'created_at'. 'timestamp' is
              an SQL:2003 reserved word and can cause parse errors in
              strict-mode drivers.
  - [DESIGN]  Alert decision logic extracted to pure function
              should_trigger_alert(). Core state machine is now testable
              without a database — pass any two risk class strings and
              get a deterministic result.
  - [DESIGN]  DB lookup extracted to _fetch_yesterday_risk(). Separates
              IO from logic; each function has one responsibility.
  - [DESIGN]  Alert messages and action strings extracted to ALERT_CONFIG
              constant. Enables localisation, A/B testing, and severity
              scaling without modifying decision logic.
  - [DESIGN]  evaluate_alert_state() returns a structured dict instead of
              a raw (status, string) tuple. Downstream consumers (API,
              SMS gateway, dashboard) can access individual fields without
              string parsing.
  - [DESIGN]  conn.row_factory = sqlite3.Row set on every connection.
              Allows column access by name (row["risk_class"]) rather than
              positional index (row[0]), preventing silent misalignment if
              column order changes.
"""

import os
import contextlib
import sqlite3
import pandas as pd
from datetime import timedelta

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "feedback_loop.sqlite")

# ---------------------------------------------------------------------------
# Alert configuration
# ---------------------------------------------------------------------------

ALERT_CONFIG = {
    "Low": {
        "status":  "NO_ALERT",
        "message": "Risk is Low. No action required.",
        "action":  None,
    },
    "Medium": {
        "status":  "MONITOR",
        "message": "Risk is Medium. Continue to monitor crop condition.",
        "action":  "Inspect crop weekly. Ensure drainage is adequate.",
    },
    "High": {
        # Fired only after 2 consecutive High prediction days
        "status":  "TRIGGER_ALERT",
        "message": (
            "Red Rot risk is HIGH in your area for the next 3-7 days. "
            "Take preventive action immediately."
        ),
        "action":  (
            "Avoid irrigation overload. Apply recommended fungicide. "
            "Remove and destroy infected stools. Monitor crop daily."
        ),
    },
    "High_pending": {
        # First High day — waiting for confirmation
        "status":  "SILENT_LOG",
        "message": "High risk spike detected (1 day). Awaiting 2nd consecutive day before alerting.",
        "action":  None,
    },
}

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection with WAL journal mode and named-column row factory.

    WAL mode allows concurrent readers alongside a single writer — essential
    when inference workers and the alert engine access the DB simultaneously.
    row_factory = sqlite3.Row enables column access by name, preventing silent
    misalignment if column order changes in future schema revisions.

    All callers MUST wrap with contextlib.closing():
        with contextlib.closing(_get_connection()) as conn:
            ...
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_alert_db() -> None:
    """
    Create the predictions table if it does not already exist.
    Aligned with feedback_db.py to ensure a unified schema.
    """
    from feedback_db import init_db
    init_db()
    
    with contextlib.closing(_get_connection()) as conn:
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_lookup
            ON predictions (location, prediction_date)
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _fetch_yesterday_risk(location: str, yesterday_date) -> str:
    """
    Return the risk_class recorded for `location` on `yesterday_date`,
    or "Unknown" if no record exists.

    Queries by prediction_date (the date the model ran), NOT target_date
    (the future outbreak window). The consecutive-day rule requires two
    model runs on consecutive calendar days to both classify as High —
    not two predictions pointing at adjacent future windows.
    """
    with contextlib.closing(_get_connection()) as conn:
        row = conn.execute(
            """
            SELECT risk_class FROM predictions
            WHERE  location        = ?
              AND  prediction_date = ?
            ORDER  BY created_at DESC
            LIMIT  1
            """,
            (location, str(yesterday_date)),
        ).fetchone()
    return row["risk_class"] if row else "Unknown"


def log_prediction(
    prediction_id: str,
    location: str,
    prediction_date,
    target_date,
    risk_class: str,
    risk_score: float,
    alert_status: str,
) -> None:
    """
    Insert a prediction record into the predictions table.
    Aligned with feedback_db.py schema.
    """
    with contextlib.closing(_get_connection()) as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (prediction_id, location, prediction_date, target_date, risk_class, risk_score, alert_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                location,
                str(pd.to_datetime(prediction_date).date()),
                str(pd.to_datetime(target_date).date()),
                risk_class,
                float(risk_score),
                alert_status,
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Pure alert logic (no IO — fully unit-testable)
# ---------------------------------------------------------------------------

def should_trigger_alert(
    current_risk_class: str,
    yesterday_risk_class: str,
) -> dict:
    """
    Determine alert state from two risk class strings.

    This is a pure function with no database dependency — pass any two
    risk class strings and receive a deterministic structured response.
    All IO (DB lookup, logging) is handled by the caller.

    Args:
        current_risk_class   : "Low" | "Medium" | "High"
        yesterday_risk_class : "Low" | "Medium" | "High" | "Unknown"

    Returns:
        dict with keys:
          status   — action code string (e.g. "TRIGGER_ALERT")
          message  — human-readable explanation
          action   — recommended farmer action, or None
    """
    if current_risk_class in ("Low", "Medium"):
        cfg = ALERT_CONFIG[current_risk_class]
        return {
            "status":  cfg["status"],
            "message": cfg["message"],
            "action":  cfg["action"],
        }

    if current_risk_class == "High":
        if yesterday_risk_class == "High":
            # Two consecutive High days — fire the outbound alert
            cfg = ALERT_CONFIG["High"]
        else:
            # First High day — log silently, wait for confirmation
            cfg = ALERT_CONFIG["High_pending"]
        return {
            "status":  cfg["status"],
            "message": cfg["message"],
            "action":  cfg["action"],
        }

    # Unrecognised risk class — fail loudly so callers can fix upstream issues
    raise ValueError(
        f"Unrecognised risk_class: {current_risk_class!r}. "
        f"Expected one of: 'Low', 'Medium', 'High'."
    )


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------

def evaluate_alert_state(
    location: str,
    current_prediction_date,
    current_risk_class: str,
) -> dict:
    """
    Evaluate whether an outbound alert should be triggered for a location.

    Applies the 2-consecutive-High-day clustering rule to prevent alert
    fatigue from isolated single-day risk spikes.

    Args:
        location               : farm or region identifier (matches DB records)
        current_prediction_date: date the model ran for this prediction
        current_risk_class     : "Low" | "Medium" | "High"

    Returns:
        Structured dict from should_trigger_alert():
          {status, message, action}

    Typical usage:
        result = evaluate_alert_state("Sangli", "2024-08-15", "High")
        if result["status"] == "TRIGGER_ALERT":
            send_sms(location, result["message"], result["action"])
        log_prediction(location, prediction_date, target_date,
                       risk_class, risk_score, result["status"])
    """
    yesterday = (
        pd.to_datetime(current_prediction_date).date() - timedelta(days=1)
    )
    yesterday_risk = _fetch_yesterday_risk(location, yesterday)
    return should_trigger_alert(current_risk_class, yesterday_risk)