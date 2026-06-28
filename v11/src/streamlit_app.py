"""
V11 KG-CTCN Farmer Dashboard (Streamlit)
--------------------------------------------------
Production UI for real-time Red Rot forecasting.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deployment_layer import DeploymentAPI

st.set_page_config(page_title="Red Rot Early Warning System", layout="wide")

# Initialize API
if 'api' not in st.session_state:
    st.session_state.api = DeploymentAPI()

st.title("🌾 Red Rot Early Warning System (V11)")
st.markdown("""
This system provides **3-7 day early warnings** for Red Rot disease in Sugarcane.
It uses causal temporal convolution networks (TCN) guided by biological knowledge and real-time NASA POWER weather data.
""")

# Sidebar Inputs
st.sidebar.header("📍 Farm Location")
location = st.sidebar.selectbox("Select Region", ["Sangli", "Kolhapur", "Pune"])
target_date = st.sidebar.date_input("Target Prediction Date", datetime.now())

st.sidebar.header("🚜 Agronomic Inputs (Optional)")
is_ratoon = st.sidebar.checkbox("Is Ratoon Crop?", value=False)
crop_age = st.sidebar.slider("Crop Age (Days)", 0, 360, 150)
variety_susc = st.sidebar.select_slider(
    "Variety Susceptibility",
    options=[0, 1, 2],
    value=1,
    format_func=lambda x: {0: "Resistant", 1: "Moderate", 2: "Susceptible"}[x]
)

# Action Button
if st.sidebar.button("Run Real-Time Inference"):
    with st.spinner("Fetching NASA POWER weather data and running causal inference..."):
        farmer_inputs = {
            "is_ratoon": int(is_ratoon),
            "crop_age_days": crop_age,
            "variety_susceptibility": variety_susc
        }
        
        try:
            result = st.session_state.api.predict(location, target_date, farmer_inputs)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Dashboard Layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Risk Score", f"{result['risk_score']:.2f}")
                    risk_color = "green" if result['risk_class'] == "Low" else ("orange" if result['risk_class'] == "Medium" else "red")
                    st.markdown(f"### Risk Class: <span style='color:{risk_color}'>{result['risk_class']}</span>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence", f"{result.get('confidence_score', 0):.2f}")
                    st.info(f"📅 Lead Time: {result['lead_time_window']}")
                
                with col3:
                    st.success(f"ID: {result['prediction_id'][:8]}")
                    if result.get("is_signal_saturated"):
                        st.error("⚠️ SIGNAL SATURATED")
                    else:
                        st.warning(f"Status: {result['alert_state']}")

                if result.get("is_signal_saturated"):
                    st.warning(result["alert_message"])
                
                st.divider()
                
                st.subheader("🧐 Explainability Summary")
                st.write(result['explanation'])
                
                st.subheader("📢 Advisory Action")
                st.markdown(f"**{result['advisory_action']}**")
                
                st.divider()
                
                # Feedback Section
                st.subheader("📝 Farmer Feedback")
                st.write("Your feedback helps improve the system's accuracy for future seasons.")
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    obs = st.radio("Did you observe an outbreak 3-7 days after this date?", ["Unknown", "Yes", "No"])
                with f_col2:
                    if st.button("Submit Feedback"):
                        # In a real app, this calls the /feedback endpoint
                        from feedback_db import submit_feedback
                        submit_feedback(result['prediction_id'], obs)
                        st.balloons()
                        st.success("Feedback submitted! Thank you.")
                        
        except Exception as e:
            st.error(f"System Error: {e}")

else:
    st.info("👈 Select your location and click 'Run Real-Time Inference' to get a prediction.")

st.sidebar.markdown("---")
st.sidebar.caption("V11 KG-CTCN Production Stack | NASA POWER API Data")
