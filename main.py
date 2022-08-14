# Put the code for your API here.
import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from starter.train_model import read_df, save_model, load_model, train_and_save_model
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

label = 'salary'


def compute_and_save_model_performance_per_slice(df, cat_features, encoder, lb, model, path):
    metric_per_slice = []
    for category in cat_features:
        for category_cls in df[category].unique():
            df_slice = df[df[category] == category_cls]

            X_slice_test, y_slice_test, _, _ = process_data(df_slice, categorical_features=cat_features,
                                                            label=label, training=False, encoder=encoder, lb=lb)
            logger.info(f'processing: {category} for class: {category_cls}')

            preds = inference(model, X_slice_test)
            precision, recall, fbeta = compute_model_metrics(y_slice_test, preds)

            category_cls_name = category_cls if category_cls != '?' else 'unknown'

            metric_per_slice.append([f'{category}-{category_cls_name}', f'{precision:.3f}',
                                     f'{recall:.3f}', f'{fbeta:.3f}'])

    with open(path, 'w') as fp:
        for single_slice in metric_per_slice:
            fp.write("%s\n" % single_slice)


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'data/census.csv')

    logger.info(f'reading CSV data')
    df = read_df(csv_path)
    logger.info(f'DF shape: {df.shape}')

    train, test = train_test_split(df, test_size=0.20)

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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )

    logger.info(f'X_train shape: {X_train.shape}')
    logger.info(f'y_train shape: {y_train.shape}')

    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label=label,
                                        training=False, encoder=encoder, lb=lb)

    logger.info(f'X_test shape: {X_test.shape}')
    logger.info(f'y_test shape: {y_test.shape}')

    encoder_path = os.path.join(current_dir, 'model/encoder.pkl')
    save_model(encoder, encoder_path)
    lb_path = os.path.join(current_dir, 'model/lb.pkl')
    save_model(lb, lb_path)
    model_path = os.path.join(current_dir, 'model/model.pkl')
    train_and_save_model(X_train, y_train, model_path)

    saved_model = load_model(model_path)
    saved_encoder = load_model(encoder_path)
    saved_lb = load_model(lb_path)

    preds = inference(saved_model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    logger.info(f'Model performance on test data : \n precision: {precision:.3f}, recall: {recall:.3f}, '
                f'fbeta: {fbeta:.3f}')

    output_path = os.path.join(current_dir, 'model/slice_output.txt')
    compute_and_save_model_performance_per_slice(df, cat_features, saved_encoder, saved_lb, saved_model, output_path)

