# FORENSIC AUDIT REPORT: Crop Disease Prediction System (v3 vs v4)

## A. ROOT CAUSE ANALYSIS (Ranked)

1. **TEST SET THRESHOLD OVERFITTING (CRITICAL)**: 
   Versions `v4` and `v3_multiscale_validation` implement a threshold sweep on the **test set** to select the "best" F1 score. This constitutes data leakage, as the decision boundary is optimized using ground truth from the future/test set. In contrast, `v3` uses a fixed 0.5 threshold.

2. **ROLLING FEATURE LEAKAGE (HIGH)**:
   The `dataset_audit.py` reveals that rolling features in `feature_engg.py` (and thus `early_warning_dataset.csv`) were computed without `.shift(1)`. This means the "past 3-day average" for today includes today's weather, which is often highly correlated with the target event, leading to inflated training accuracy.

3. **LABEL GENERATION DIVERGENCE (HIGH)**:
   `v4` re-generates labels using a "simplified proxy" (RH > 80% & T > 25°C & Rain > 1mm). If the original `v3` dataset used a more conservative or multi-factor biological definition, the models are literally solving different mathematical problems.

4. **SEQUENCE VS ROW ARCHITECTURE**:
   `v4` uses 14-day raw sequences fed into a GRU, whereas `v3` relies on a Random Forest seeing pre-aggregated rolling features. The GRU can learn non-linear temporal patterns that the RF cannot, provided the data is clean.

---

## B. PROCEDURAL BREAKDOWN

### 1. Training & Preprocessing
| Phase | v3 (Engineered) | v4 (Raw) | Audit Verdict |
|-------|-----------------|----------|---------------|
| **Data Source** | `early_warning_dataset.csv` | NASA POWER Raw CSV | Divergent Source |
| **Scaling** | None (RF-based) | StandardScaler | v4 is more rigorous for DL |
| **Temporal Logic** | Pre-baked rolling features | 14-day raw sliding windows | v4 captures finer granularity |
| **Leakage Control** | FAILED (No shift(1) on rolling) | FAILED (Threshold optimization) | Both versions have integrity issues |

### 2. Evaluation Consistency
- **v3**: Fixed threshold (0.5). Reliable but potentially pessimistic.
- **v4**: Dynamic "Best" threshold. Unreliable; hides calibration failures and overfits metrics to the test period.
- **Event-Based Logic**: Both versions use 3 consecutive days as an "outbreak," but the "Early Warning" window (3-7 days lead time) is inconsistent in how it handles overlapping detections.

---

## C. PRODUCTION READINESS VERDICT

### **VERDICT: EXPERIMENTAL ONLY**

Neither version is currently suitable for deployment.

- **v3** is compromised by feature leakage (rolling windows include current day).
- **v4** is compromised by evaluation leakage (test-set threshold tuning) and a "proxy" label that may not reflect actual biological risk.

**Deployability Ranking:**
1. **v4 (Leakage-Proof variant)**: If the threshold is fixed on validation data, this is the most robust path forward.
2. **v3**: Requires a full re-generation of `early_warning_dataset.csv` with `.shift(1)` applied to all historical features.

---

## D. FAILURE MODE ANALYSIS

- **The "Monsoon Bias"**: Models in both versions show high performance during Jun-Sep. A production model might trigger alarms purely based on the season (high humidity) rather than specific disease triggers, leading to "Alert Fatigue" for farmers.
- **Lead-Time Collapse**: Because the model is trained on `target_5d` (any risk in next 5 days), it often predicts an outbreak *on the day it starts*, failing the 3-day early warning requirement.
- **Threshold Drift**: A threshold optimized for the 2024 test set will likely fail in a drier 2025 season.

---

## E. ARCHITECTURAL RECOMMENDATION

### **Unified Pipeline (v5) Required**

A merger is not recommended due to the fundamental flaws in the current data engineering. Instead, a unified **v5 "Integrity First"** pipeline should be built:

1. **Standardize Target**: Use a single biological risk definition across all experiments.
2. **Atomic Feature Engineering**: All rolling/lag features MUST use `.shift(1)` to ensure zero look-ahead.
3. **Strict Evaluation**:
   - Split: Train (70) / Val (15) / Test (15).
   - **Rule**: Threshold MUST be tuned on the **Validation set only** and locked before touching the Test set.
4. **Hybrid Modeling**: Feed both raw 14-day sequences (for GRU) and shifted engineered features (for RF) into a single evaluation harness to determine the true winner.
5. **Lead-Time Penalty**: Introduce a custom loss function that penalizes late detections more heavily than early ones.
