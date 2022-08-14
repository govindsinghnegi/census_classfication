# Script to train machine learning model.

import logging
import pandas as pd
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

from starter.ml.model import train_model, compute_model_metrics, inference

# Add the necessary imports for the starter code.

def read_df(path):
    return pd.read_csv(path)

def train_and_save_model(X_train, y_train, path):
    logger.info(f'training model')
    model = train_model(X_train, y_train)
    save_model(model, path)

def save_model(model, path):
    logger.info(f'saving model as a pkl file')
    pickle.dump(model, open(path, "wb"))

def load_model(path):
    logger.info(f'loading model from pkl file')
    saved_model = pickle.load(open(path, "rb"))
    return saved_model

