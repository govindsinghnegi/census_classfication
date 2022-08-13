# Script to train machine learning model.

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

# Add the necessary imports for the starter code.

df = pd.read_csv('data/census.csv')
print(df.head(5))
print(df.shape)

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
    train, categorical_features=cat_features, label="salary", training=True
)

print(X_train.shape)
print(y_train.shape)

X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary",
                                             training=False, encoder=encoder, lb=lb)

print(X_test.shape)
print(y_test.shape)

# Proces the test data with the process_data function.

model = train_model(X_train, y_train)

print(f"saving model as a pkl file")
pickle.dump(model, open('model/model.pkl', "wb"))

# Train and save a model.

saved_model = pickle.load(open('model/model.pkl', "rb"))

preds = inference(saved_model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"precision: {precision:.3f}")
print(f"recall: {recall:.3f}")
print(f"fbeta: {fbeta:.3f}")

metric_per_slice = []

for category in cat_features:
    print(f'processing ------------ {category}')
    for category_cls in df[category].unique():
        df_slice = df[df[category] == category_cls]

        X_slice_test, y_slice_test, _, _ = process_data(df_slice, categorical_features=cat_features, label="salary",
                                                        training=False, encoder=encoder, lb=lb)
        print(f'for class: {category_cls}')

        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)

        category_cls_name = category_cls if category_cls != '?' else 'unknown'

        metric_per_slice.append([f'{category}-{category_cls_name}', f'{precision:.3f}',
                                 f'{recall:.3f}', f'{fbeta:.3f}'])

with open('model/slice_output.txt', 'w') as fp:
    for single_slice in metric_per_slice:
        # write each item on a new line
        fp.write("%s\n" % single_slice)