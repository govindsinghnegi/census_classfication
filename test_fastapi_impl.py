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

def test_post_category_less_than_50K():
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

def test_post_category_more_than_50K():
    census_data = {
        'age': 35,
        'workclass': 'Private',
        'fnlgt': 123456,
        'education': 'Doctorate',
        'education_num': 16,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Prof-specialty',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }
    data = json.dumps(census_data)
    r = client.post("/create_prediction", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {'prediction': '>50K'}

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



