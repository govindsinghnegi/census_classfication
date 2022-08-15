from fastapi import FastAPI
from pydantic import BaseModel

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: male
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

@app.post("/create_prediction")
async def exercise_function(census_data: CensusData):
    print(f'--- post called-----')
    print(f'age: {census_data.age}')
    print(f'workclass: {census_data.workclass}')
    print(f'marital_status: {census_data.marital_status}')
    print(f'race: {census_data.race}')
    print(f'hours_per_week: {census_data.hours_per_week}')
    print(f'native_country: {census_data.native_country}')
    return census_data
