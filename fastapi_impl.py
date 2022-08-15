from fastapi import FastAPI
from pydantic import BaseModel, Field

class CensusData(BaseModel):
    age: int = Field(example=30)
    workclass: str = Field(example="Private")
    fnlgt: int = Field(example=12345)
    education: str = Field(example="Bachelors")
    education_num: int = Field(example=13)
    marital_status: str = Field(example="Separated")
    occupation: str = Field(example="Sales")
    relationship: str = Field(example="Unmarried")
    race: str = Field(example="Black")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=0)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(example="Mexico")

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
