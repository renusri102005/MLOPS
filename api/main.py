from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
app=FastAPI()
model_path=os.path.join(os.path.dirname(__file__),"model.pkl")
model=joblib.load(model_path)
class Inputdata(BaseModel):
    marks:list[float]
@app.post("/predict")
def predict(data:Inputdata):
    prediction=model.predict([data.marks])
    return {"prediction":int(prediction[0])}