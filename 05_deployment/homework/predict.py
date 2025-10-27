import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="lead_scoring")


with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

customer1 = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# response schema
class PredictResponse(BaseModel):
    convert_probability: float
    convert: bool

@app.post("/predict")
def predict(customer: Dict[str, Any]) -> PredictResponse:
    # pipeline does transformation
    prob = pipeline.predict_proba(customer)[0, 1]

    return PredictResponse(
        convert_probability=prob,
        convert= bool(prob >= 0.5)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)