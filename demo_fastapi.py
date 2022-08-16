import requests
import json

census_data_1 = {
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

census_data_2 = {
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

print('Demo where model predicts <=50K for a person')
response = requests.post('https://census-classifcation.herokuapp.com/create_prediction', json=census_data_1)

print(f'response code: {response.status_code}')
print(f'response body: {response.json()}')

print('Demo where model predicts >50K for a person')
response = requests.post('https://census-classifcation.herokuapp.com/create_prediction', json=census_data_2)

print(f'response code: {response.status_code}')
print(f'response body: {response.json()}')