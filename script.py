import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn

import dagshub
dagshub.init(repo_owner='heschmat', repo_name='sentiment_analysis_dvc_pipeline', mlflow=True)

df = pd.read_csv('./data/imdb-50k.csv')

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=.2, random_state=31
)

model_params = {
    'max_iter': 1000,
    'C': .1
}
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(**model_params))
])

# Start MLflow experiment

mlflow.set_experiment('Sentiment Analysis IMDB')

with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred, output_dict=True)

    # Log metrics
    mlflow.log_param('C', model_params['C'])
    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('f1-score', f1_score(y_test, y_pred))

    mlflow.sklearn.log_model(pipeline, 'sentiment_model')

    print(f'Run logged under ID: {mlflow.active_run().info.run_id}')
