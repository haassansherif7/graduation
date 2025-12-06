from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# تحميل الموديل
with open("medical_model.pkl", "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    try:
        X = np.array([data.features])
        y = model.predict(X)
        return {"prediction": y.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
