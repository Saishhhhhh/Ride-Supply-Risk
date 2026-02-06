from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from src.inference import RideRiskPredictor
from datetime import datetime

app = FastAPI(
    title="Ride Supply Risk API",
    description="API for predicting ride cancellation probability and supply-demand stress.",
    version="1.0.0",
    docs_url="/swagger"
)


# Initialize predictor
# We initialize it globally so it loads once on startup.
try:
    predictor = RideRiskPredictor()
except Exception as e:
    print(f"Failed to load model: {e}")
    predictor = None

class RideRequest(BaseModel):
    vehicle_type: str = Field(..., example="Auto")
    pickup_location: str = Field(..., example="Rohini West")
    drop_location: str = Field(..., example="Dwarka Mor")
    booking_value: float = Field(..., example=450.0)
    ride_distance: float = Field(..., example=12.5)
    payment_method: str = Field(..., example="UPI")
    driver_rating: Optional[float] = Field(None, example=4.5)
    customer_rating: Optional[float] = Field(None, example=None)
    time_of_booking: Optional[datetime] = Field(default_factory=datetime.now)

class RiskResponse(BaseModel):
    cancellation_probability: float
    supply_stress_probability: float

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": predictor is not None}

@app.post("/predict", response_model=RiskResponse, tags=["Predictions"])
def predict_risk(request: RideRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to dict and handle optional logic if needed
        data = request.dict()
        # time_of_booking in pydantic is datetime object, preprocessing handles it.
        
        result = predictor.predict(data)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# For debugging running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
