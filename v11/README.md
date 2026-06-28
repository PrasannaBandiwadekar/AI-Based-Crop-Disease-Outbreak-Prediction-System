# 🌾 V11 KG-CTCN: Causal Red Rot Early Warning System

V11 KG-CTCN is a production-grade, knowledge-guided agricultural forecasting system designed to predict **Red Rot disease outbreaks in Sugarcane** with a 3–7 day lead time. 

Unlike traditional black-box models, V11 combines **Causal Temporal Convolutional Networks (TCN)** with biological constraints derived from a disease Knowledge Graph (KG), ensuring all predictions are physically grounded and causally consistent.

## 🧠 Core Architecture: Knowledge-Guided Causal TCN

The system employs a dual-branch architecture:
- **Weather Branch (Causal TCN)**: Processes 28-day sequences of climate variables (Temp, Humidity, Rainfall) using dilated causal convolutions. It captures multi-scale temporal dependencies (7, 14, 28-day windows) without future data leakage.
- **Agronomic Branch (MLP)**: Embeds static field-level vulnerability (variety susceptibility, crop age, ratoon status) to modulate the climate risk.
- **Fusion Layer**: An attention-based mechanism that combines climate pressure with agronomic vulnerability to produce a final risk probability.

## 🚀 Production Stack

V11 is built for real-world deployment with a focus on reliability and interpretability:

- **Real-Time Data**: Integrates with the **NASA POWER API** for live weather ingestion (t-120 to t window).
- **FastAPI Backend**: Provides high-performance REST endpoints (`/predict`, `/feedback`) for mobile and SMS integration.
- **Streamlit Dashboard**: A dedicated farmer-facing UI for location-based risk assessment and feedback collection.
- **Zero-CSV Inference**: The production path is 100% weather-driven and uses pre-fitted scalers, removing dependencies on historical training files during runtime.
- **Event Clustering**: An intelligent alert engine that prevents "alert fatigue" by only triggering outbound notifications when High Risk is sustained for ≥2 consecutive days.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Active internet connection (for NASA POWER API access)

### Setup
1. Clone the repository and navigate to the `v11/` directory.
2. Install dependencies:
   ```bash
   pip install torch pandas numpy scikit-learn requests fastapi uvicorn streamlit joblib
   ```
3. Ensure the frozen model and scalers are in place:
   - `v11/models/v11_kg_ctcn.pth`
   - `v11/models/weather_scaler.pkl`
   - `v11/models/agro_scaler.pkl`

## 📡 Usage

### Running the API
```bash
python v11/src/api_layer.py
```
Access the interactive documentation at `http://localhost:8000/docs`.

### Running the Farmer Dashboard
```bash
python -m streamlit run v11/src/streamlit_app.py

```

### Stress Testing & Validation
To verify the system's resilience to extreme environmental shifts:
```bash
python v11/src/stress_test.py
```

## 🔄 Feedback Loop & Retraining

V11 implements a **Human-in-the-loop** learning cycle:
1. **Log**: Every prediction is stored in a local SQLite database (`feedback_loop.sqlite`) with a unique UUID.
2. **Feedback**: Farmers submit observations (Outbreak Observed: Yes/No).
3. **Batch Retraining**: Periodically, the `retraining_pipeline.py` script aggregates validated feedback to refine the model weights in an offline environment, following strict versioning (V11 -> V11.1).

## 🧪 System Guarantees
- **Strict Causality**: No future leakage; t-120 causal windowing enforced.
- **Biological Realism**: Knowledge Graph constraints (lags, accumulations) baked into feature engineering.
- **Explainability**: Complex signals translated into human-readable agricultural reasoning.

---
*Developed as part of the Crop Disease Outbreak Prediction System (Red Rot Sugarcane).*
