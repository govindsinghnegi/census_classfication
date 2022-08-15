import requests
import json

census_data = {
    'age': 30,
    'workclass': 'Private',
    'fnlwgt': 12345,
    'education': 'Bachelors',
    'education-num': 13,
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

response = requests.post('https://census-classifcation.herokuapp.com/create_prediction', json=census_data)

print(f'response code: {response.status_code}')
print(f'response body: {response.json()}')