import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import load_model


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
    current_dir = os.path.dirname(__file__)
    encoder_path = os.path.join(current_dir, 'model/encoder.pkl')
    lb_path = os.path.join(current_dir, 'model/lb.pkl')
    model_path = os.path.join(current_dir, 'model/model.pkl')

    saved_model = load_model(model_path)
    saved_encoder = load_model(encoder_path)
    saved_lb = load_model(lb_path)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    census_data_dict = {
        'age': [census_data.age],
        'workclass': [census_data.workclass],
        'fnlgt': [census_data.fnlgt],
        'education': [census_data.education],
        'education-num': [census_data.education_num],
        'marital-status': [census_data.marital_status],
        'occupation': [census_data.occupation],
        'relationship': [census_data.relationship],
        'race': [census_data.race],
        'sex': [census_data.sex],
        'capital-gain': [census_data.capital_gain],
        'capital-loss': [census_data.capital_loss],
        'hours-per-week': [census_data.hours_per_week],
        'native-country': [census_data.native_country]
    }

    df = pd.DataFrame(census_data_dict)

    print(df.shape)

    X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=saved_encoder, lb=saved_lb)

    print(f'X shape: {X.shape}')

    preds = inference(saved_model, X)

    response = '>50K' if preds[0] else '<=50K'

    print(f'preds: {preds[0]}')

    return {'prediction': response}
