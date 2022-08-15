import pandas as pd
import pytest
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference
from starter.train_model import load_model

@pytest.fixture
def data():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'data/census.csv')
    df = pd.read_csv(csv_path)
    return df

@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_important_column_exists(data):
    '''
    Test if the most important columns exist in the CSV data
    '''
    important_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "salary"
    ]

    # Check column presence
    assert set(data.columns.values).issuperset(set(important_columns))


def test_process_data(data, cat_features):
    '''
    Test if the process_data returns the expcted entities
    '''
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label='salary')

    assert len(X[0]) == 108
    assert len(y) == 32561
    assert type(encoder) is OneHotEncoder
    assert type(lb) is LabelBinarizer


def test_model_metrics(data, cat_features):
    '''
    Test if persisted model performance is more than the initial benchmarking result
    '''
    current_dir = os.path.dirname(__file__)
    encoder_path = os.path.join(current_dir, 'model/encoder.pkl')
    lb_path = os.path.join(current_dir, 'model/lb.pkl')
    model_path = os.path.join(current_dir, 'model/model.pkl')

    saved_model = load_model(model_path)
    saved_encoder = load_model(encoder_path)
    saved_lb = load_model(lb_path)

    train, test = train_test_split(data, test_size=0.20)

    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label='salary',
                                        training=False, encoder=saved_encoder, lb=saved_lb)

    preds = inference(saved_model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert precision > 0.75
    assert recall > 0.55
    assert fbeta > 0.65

