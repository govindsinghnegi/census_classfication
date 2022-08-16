## Model Details

Govind Singh created the model. It is GradientBoostingClassifier using the default hyperparameters 
in scikit-learn 1.1.1.

## Intended Use

This model should be used to predict the salary of a person based on publicly available 
Census Bureau data. The users are anyone who is interested in classifying people based on their 
salaries.

## Training Data
The data was obtained from UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/census+income.
The original data set has 32561 rows,there are 14 feature columns and target column is 'salary'.
80% of this data has been used for training purpose.

## Evaluation Data

The data was obtained from UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/census+income.
The original data set has 32561 rows,there are 14 feature columns and target column is 'salary'.
20% of this data has been used for testing/evaluation purpose.

## Metrics

For training dataset, model performance was:
precision: 0.790 , recall: 0.602 , f1: 0.683

For test/evaluation dataset, model performance was:
precision: 0.776, recall: 0.628, fbeta: 0.694

## Ethical Considerations

The data has personal information such as gender, ethnicity, profession and salary.

## Caveats and Recommendations

The overall performance can still be improved using more sophisticated ML models but that is out of
scope for this project.
