import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from weather_service import fetch_live_weather, DISTRICTS
from advisory_engine import get_comprehensive_advisory
import performance_service as ps
import os
from datetime import datetime

# ---------------------------
# 1. PAGE CONFIG & STYLING
# ---------------------------
st.set_page_config(page_title="Red Rot Live Decision System", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-box { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1976d2; }
    .status-card { padding: 25px; border-radius: 12px; color: white; margin-bottom: 25px; }
    .critical { background-color: #d32f2f; border-bottom: 8px solid #b71c1c; }
    .warning { background-color: #f57c00; border-bottom: 8px solid #e65100; }
    .watch { background-color: #fbc02d; color: black; border-bottom: 8px solid #f9a825; }
    .stable { background-color: #388e3c; border-bottom: 8px solid #2e7d32; }
    .footer { text-align: center; color: #666; font-size: 0.85em; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 20px; }
    .suggestion-box { background-color: #fff3e0; padding: 20px; border-radius: 10px; border: 1px solid #ffb74d; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# 2. DATA ORCHESTRATION
# ---------------------------
@st.cache_data(ttl=3600)
def get_live_data(district):
    return fetch_live_weather(district)

# ---------------------------
# 3. SIDEBAR & LOCATION
# ---------------------------
st.sidebar.title("🛂 Control Panel")
selected_district = st.sidebar.selectbox("📍 Select District", list(DISTRICTS.keys()))
lang = st.sidebar.radio("🌐 Language / भाषा", ["English", "Marathi"])
lang_code = "en" if lang == "English" else "mr"

st.sidebar.divider()
st.sidebar.info("🌦️ Data Source: Live (updated hourly)")

# Fetch and Process
df_live = get_live_data(selected_district)
current_idx = len(df_live) - 1
advisory_full = get_comprehensive_advisory(df_live, current_idx)

# ---------------------------
# 4. HEADER & CRITICAL STATUS
# ---------------------------
now_str = datetime.now().strftime("%I:%M %p")
date_str = advisory_full['metadata']['date']

st.title("🚜 Sugarcane Red Rot Monitor")
st.markdown(f"**📍 District:** {selected_district} | **📅 Date:** {date_str} | **⏰ Time:** {now_str}")

adv = advisory_full[lang_code]
status = adv['status']
priority = adv['priority']
bg_class = advisory_full['en']['status'].lower()

st.markdown(f"""
    <div class="status-card {bg_class}">
        <h1 style='color: inherit; margin: 0;'>{status}</h1>
        <h3 style='color: inherit; margin: 0; opacity: 0.9;'>{priority}</h3>
    </div>
""", unsafe_allow_html=True)

# ---------------------------
# 5. TRUST & RELIABILITY (TRANS-PANEL)
# ---------------------------
st.divider()
metrics = ps.calculate_reliability_metrics()

if metrics and metrics['count'] >= 10:
    st.subheader("📊 System Reliability (Field Performance)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Alert Quality", f"{metrics['precision']*100:.0f}%" if metrics['precision'] else "N/A", help="Correct Critical Alerts / Total Critical Alerts")
    with m2:
        st.metric("Avg Lead Time", f"{metrics['avg_lead']:.1f} days", help="Time between alert and symptom")
    with m3:
        st.metric("Miss Rate", f"{metrics['miss_rate']*100:.0f}%", help="Outbreaks system failed to detect")
    with m4:
        st.metric("Samples", metrics['count'], delta="Learning Active")
else:
    st.info("📊 **System Reliability:** Collecting field performance data (Min. 10 feedback points required). Keep logging feedback to activate metrics.")

# Adaptive Tuning (Human Approval)
suggestion = ps.get_calibration_suggestion()
if suggestion:
    st.markdown("### 🧠 Learning Status: Calibration Required")
    st.markdown(f"""
        <div class="suggestion-box">
            <b>System Suggestion:</b> Update {suggestion['type'].replace('_', ' ').title()}<br>
            Current: {suggestion['current']} → <b>Proposed: {suggestion['proposed']}</b><br><br>
            ✅ <b>Benefit:</b> {suggestion['benefit']}<br>
            ⚠ <b>Trade-off:</b> {suggestion['tradeoff']}
        </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Approve & Recalibrate"):
        ps.apply_calibration(suggestion['proposed'], suggestion['type'])
        st.success("System recalibrated. New thresholds active.")
        st.rerun()

# ---------------------------
# 6. BIOLOGICAL INTERPRETATION (The "Why")
# ---------------------------
st.subheader("📊 Field Conditions & Interpretation")
c1, c2, c3 = st.columns(3)

interp = adv['interpretation']
with c1:
    st.markdown(f"<div class='metric-box'><b>Rain (3d):</b><br>{interp['rain']}</div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-box'><b>Humidity:</b><br>{interp['rh']}</div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-box'><b>Moisture Gate:</b><br>{interp['moisture']}</div>", unsafe_allow_html=True)

# ---------------------------
# 7. RISK TRAJECTORY & PERSISTENCE
# ---------------------------
st.divider()
st.subheader("📈 Risk Trajectory & Persistence")
risk_history = []
dates = []
for i in range(max(13, current_idx - 14), current_idx + 1):
    temp_adv = get_comprehensive_advisory(df_live, i)
    if temp_adv:
        risk_history.append(temp_adv['probs']['5d'])
        dates.append(temp_adv['metadata']['date'])

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=risk_history, fill='tozeroy', line=dict(color='#1976d2', width=3), name="5d Risk"))
for i in range(2, len(risk_history)):
    if risk_history[i] >= 0.15 and risk_history[i-1] >= 0.15 and risk_history[i-2] >= 0.15:
        fig.add_vrect(x0=dates[i-2], x1=dates[i], fillcolor="red", opacity=0.1, line_width=0)
fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), yaxis_range=[0, 0.6])
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 8. ADVISORY & STRUCTURED FEEDBACK
# ---------------------------
ca_col, fb_col = st.columns([2, 1])

with ca_col:
    st.success(f"### 🎯 Actionable Advisory\n\n{adv['advisory']}")
    st.markdown(f"**Action Priority:** {priority}")
    
    # Missed Outbreak Reporting (Hidden in main flow, but prominent)
    st.divider()
    st.warning("⚠ **Found something we missed?**")
    exp = st.expander("🚨 Report Observed Outbreak (Missed by System)")
    with exp:
        s_date = st.date_input("When did you first see symptoms?")
        s_note = st.text_area("Any specific field observation?")
        if st.button("📤 Log Missed Outbreak"):
            # Logging logic
            new_fb = {
                "alert_date": None,
                "symptom_date": s_date,
                "district": selected_district,
                "alert_status": "NONE",
                "feedback": "YES",
                "event_type": "MISSED_OUTBREAK"
            }
            pd.DataFrame([new_fb]).to_csv("feedback_db.csv", mode='a', header=not os.path.exists("feedback_db.csv"), index=False)
            st.success("Outbreak reported. Reliability engine is analyzing the missing signal.")

with fb_col:
    st.markdown("### 🧪 Field Feedback")
    st.write("Did you observe Red Rot symptoms recently following our alerts?")
    
    col_y, col_n, col_u = st.columns(3)
    feedback_val = None
    if col_y.button("✅ Yes"): feedback_val = "YES"
    if col_n.button("✖ No"): feedback_val = "NO"
    if col_u.button("⏳ Unsure"): feedback_val = "UNSURE"

    if feedback_val:
        sym_date = None
        if feedback_val == "YES":
            sym_date = st.date_input("When did symptoms appear?", value=datetime.now())
        
        # Save to DB
        new_fb = {
            "alert_date": date_str,
            "symptom_date": sym_date if sym_date else "",
            "district": selected_district,
            "alert_status": status,
            "prob": advisory_full['probs']['5d'],
            "feedback": feedback_val,
            "event_type": "ALERT_RESPONSE"
        }
        pd.DataFrame([new_fb]).to_csv("feedback_db.csv", mode='a', header=not os.path.exists("feedback_db.csv"), index=False)
        st.toast(f"Feedback logged: {feedback_val}")

# ---------------------------
# 9. FOOTER
# ---------------------------
st.markdown(f"""
    <div class="footer">
        ℹ️ Advisory based on weather patterns and biological persistence rules. Field inspection recommended before action.<br>
        <b>Methodology:</b> Closed-Loop Learning System | V{ps.load_system_state()['version']}
    </div>
""", unsafe_allow_html=True)
