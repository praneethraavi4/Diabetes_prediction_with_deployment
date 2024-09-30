from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np

class Data(BaseModel):
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int




MODEL = load_model("model.h5")

app = FastAPI()

@app.get("/")
async def root():
    return "welcome to diabetes classification"

@app.post("/predict")
async def predict(data: Data):
    input = data.dict()
    Pregnancies = input['Pregnancies']
    Glucose = input['Glucose']
    BloodPressure = input['BloodPressure']
    SkinThickness = input['SkinThickness']
    Insulin = input['Insulin']
    BMI = input['BMI']
    DiabetesPedigreeFunction = input['DiabetesPedigreeFunction']
    Age = input['Age']
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    predict = MODEL.predict(input_data)


    if predict[0] == 0:
        return "The person is non diabetic"
    return "The person is diabetic"


if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost',port = 8000)