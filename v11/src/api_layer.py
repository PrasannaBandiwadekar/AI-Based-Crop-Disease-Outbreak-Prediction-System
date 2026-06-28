"""
V11 KG-CTCN Production API (FastAPI)
--------------------------------------------------
Serves the Deployment Layer logic via HTTP endpoints:
/predict and /feedback
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Literal
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deployment_layer import DeploymentAPI
from feedback_db import submit_feedback, init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the deployment API (loads frozen model and scalers) on startup.
    # This prevents the app from starting if model files are missing, but
    # avoids global import-time crashes and ensures thread-safe access via state.
    print("Initializing Feedback Database...")
    init_db()
    
    print("Loading V11 Model and Scalers...")
    try:
        app.state.deployment_api = DeploymentAPI()
    except Exception as e:
        print(f"CRITICAL: Failed to load V11 Deployment Engine: {e}")
        raise RuntimeError(f"Startup failed: {e}") from e
    yield

app = FastAPI(
    title="V11 KG-CTCN Agricultural Early Warning System",
    description="Causal, explainable, event-driven Red Rot forecasting API.",
    version="11.2",
    lifespan=lifespan
)

class PredictRequest(BaseModel):
    location: str
    date: Optional[str] = None
    is_ratoon: Optional[bool] = None
    variety_susceptibility: Optional[int] = Field(default=None, ge=0, le=2)
    crop_age_days: Optional[int] = Field(default=None, ge=0, le=365)

class FeedbackRequest(BaseModel):
    prediction_id: str
    outbreak_observed: Literal["Yes", "No", "Unknown"]
    expert_validated: Optional[bool] = False

@app.post("/predict")
def predict_risk(req: PredictRequest, request: Request):
    """
    Generate a 3-7 day early warning risk prediction.
    """
    farmer_inputs = {}
    for field in ["is_ratoon", "variety_susceptibility", "crop_age_days"]:
        val = getattr(req, field, None)
        if val is not None:
            # Convert bool to int for is_ratoon, or pass through ints
            farmer_inputs[field] = int(val)
        
    # Access stateful deployment API from app state
    result = request.app.state.deployment_api.predict(req.location, req.date, farmer_inputs)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result

@app.post("/feedback")
def log_feedback(req: FeedbackRequest):
    """
    Submit delayed ground-truth feedback for a previous prediction.
    """
    try:
        submit_feedback(
            req.prediction_id, 
            req.outbreak_observed, 
            1 if req.expert_validated else 0
        )
        return {"status": "success", "message": "Feedback successfully recorded for offline retraining."}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))

if __name__ == "__main__":
    import uvicorn
    # This acts as the local development server launcher
    print("=" * 60)
    print(" Starting V11 API Layer (v11.2) ")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
