# PRODUCTION PIPELINE AUDIT REPORT: Crop Disease Early Warning System

## A. PIPELINE PARITY REPORT
Comparison between Production (`production/pipelines/`) and Research (`research_comp/`):

| Component | Research (v3/v4) | Production | Parity Status |
|-----------|------------------|------------|---------------|
| **Feature Logic** | v3 Engineered / v4 Raw | Bio-Feature + Temporal Hybrid | **Drift**: Higher complexity than baseline research. |
| **Rolling Windows** | Leaky (v3) / Fixed in some audits | **LEAKY** (No `.shift(1)`) | **Matched Failure**: Inherited v3's critical leakage. |
| **Model Architecture**| RF (v3) / GRU (v4) | Hybrid Ensemble (RF + GRU) | **Drift**: Unvalidated fusion logic. |
| **Decision Logic** | 0.5 or Tuned | Multi-threshold Fusion (0.4/0.6) | **Mismatch**: Decision boundaries are ad-hoc. |
| **Label Source** | `early_warning_dataset.csv` | Re-calculated Proxy (v4 logic) | **Aligned** with v4's simplified proxy. |

---

## B. ROOT CAUSE ANALYSIS (Ranked)

1. **INHERITED v3 LEAKAGE (CRITICAL)**:
   The production `feature_engine.py` (Line 61) implements rolling means as `df[col].rolling(window=w).mean()`. This includes the current day's weather in the "past history" feature. Since today's weather is highly predictive of today's disease risk, the model's accuracy is artificially inflated by "peeking" at the current day.

2. **AD-HOC FUSION LOGIC**:
   The `fusion_logic.py` uses hardcoded thresholds (0.4, 0.6) that have no statistical basis in the research logs. There is no evidence that these specific thresholds provide optimal early warning or lead time.

3. **UNVALIDATED BIO-LAYER**:
   The "Bio-Feature Engine" adds complexity (`moisture_stress`, `fungal_risk`) that was not part of the `v4_raw_pipeline` research. While biologically sound, its interaction with the GRU model has not been audited for temporal stability.

---

## C. PRODUCTION RISK ASSESSMENT

- **Leakage Risk (CRITICAL)**: High. The system will appear to have near-perfect performance in historical backtests but will fail to predict the future when current-day weather is not yet known at inference time.
- **Drift Risk (HIGH)**: The system defines "Drift" as a 10% F1 degradation. However, with leaky features, the baseline F1 is false, making the drift monitor blind to real performance decay.
- **Alert Instability (MEDIUM)**: The "Suspicious Spike" alert (GRU-only trigger) is prone to seasonal noise, specifically during monsoon onset where short-term humidity spikes are frequent but biological buildup is not yet complete.

---

## D. FAILURE MODE ANALYSIS

1. **"The Invisible Outbreak"**: In real-time inference, the model relies on weather data that is usually 12-24 hours delayed. Because it was trained on leaky current-day data, it will likely miss outbreaks entirely when "today's" data is missing or estimated.
2. **"Monsoon Triggering"**: During the transition from dry to wet season, the `dry_to_wet_trigger` will fire simultaneously with the GRU spike detection. This will likely trigger a "CRITICAL ALERT" on the first day of rain, potentially 7-10 days before actual biological risk materializes, leading to late or false alarms.

---

## E. FINAL ALIGNMENT VERDICT

### **VERDICT: ❌ MISALIGNED / UNSAFE FOR DEPLOYMENT**

**Reasoning**:
The production system is a "Frankenstein" of v3's leaky features and v4's model architecture, wrapped in an unvalidated fusion logic. The **Critical Leakage** in the rolling window implementation invalidates all performance metrics reported in the production logs.

**Immediate Action Items**:
1. Apply `.shift(1)` to all rolling/max/mean operations in `production/pipelines/feature_engine.py`.
2. Move threshold selection from `fusion_logic.py` hardcoding to a data-driven approach using the **Validation set**.
3. Re-validate the GRU's performance on raw sequences vs. the leaky Bio-Layer.
