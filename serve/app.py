# serve/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# 1) Load the registered MLflow model
model = mlflow.pyfunc.load_model("models:/UpsellClassifier/1")

app = FastAPI(title="Upsell Prediction Service")

# 2) Define the booking schema
class Booking(BaseModel):
    lead_time: float
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adr: float
    hotel: str
    meal: str
    market_segment: str
    distribution_channel: str
    deposit_type: str
    customer_type: str

@app.post("/predict")
def predict(b: Booking):
    # 3) Turn into DataFrame
    df = pd.DataFrame([b.dict()])
    # 4) Compute any derived fields your model expects
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["revenue"]      = df["adr"] * df["total_nights"]
    # 5) Call the model
    prob = model.predict(df)[0]
    return {"upsell_prob": float(prob)}

