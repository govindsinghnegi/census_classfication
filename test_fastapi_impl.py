import json
from fastapi.testclient import TestClient

from fastapi_impl import app

client = TestClient(app)

def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}

def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200

def test_post():
    census_data = {
    'age': 30,
    'workclass': 'Private',
    'fnlgt': 12345,
    'education': 'Bachelors',
    'education_num': 13,
    'marital_status': 'Separated',
    'occupation': 'Sales',
    'relationship': 'Unmarried',
    'race': 'Black',
    'sex': 'Male',
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 40,
    'native_country': 'Mexico'
}
    data = json.dumps(census_data)
    r = client.post("/create_prediction", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {'prediction': '<=50K'}

def test_post_malformed():
    census_data = {
    'age': 30,
    'workclass': 'Private',
    'fnlgt': 12345,
    'education': 'Bachelors',
    'education_num': 13,
    'marital_status': 'Separated',
    'occupation': 'Sales',
    'relationship': 'Unmarried'
}
    data = json.dumps(census_data)
    r = client.post("/create_prediction", data=data)
    print(r.json())
    assert r.status_code == 422



