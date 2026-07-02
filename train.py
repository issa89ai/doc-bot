"""
Week 5 — MLflow Experiment Tracking
Trains a text topic classifier on the 20 Newsgroups dataset.
Each run logs parameters, metrics, and the model to MLflow.

Run:   python train.py
UI:    mlflow ui  →  open http://localhost:5000
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Dataset ───────────────────────────────────────────────────────────────────
# 4 topic categories — keeps training fast
CATEGORIES = ["sci.med", "sci.space", "rec.sport.hockey", "talk.politics.guns"]

print("Loading dataset...")
data = fetch_20newsgroups(subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes"))
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"  Categories: {CATEGORIES}\n")

# ── Experiments to run ────────────────────────────────────────────────────────
experiments = [
    {
        "name": "LogisticRegression-C0.1",
        "vectorizer": TfidfVectorizer(max_features=10000, ngram_range=(1, 1)),
        "model": LogisticRegression(C=0.1, max_iter=1000),
        "params": {"model": "LogisticRegression", "C": 0.1, "ngram_range": "1,1", "max_features": 10000},
    },
    {
        "name": "LogisticRegression-C1.0",
        "vectorizer": TfidfVectorizer(max_features=10000, ngram_range=(1, 1)),
        "model": LogisticRegression(C=1.0, max_iter=1000),
        "params": {"model": "LogisticRegression", "C": 1.0, "ngram_range": "1,1", "max_features": 10000},
    },
    {
        "name": "LogisticRegression-bigrams",
        "vectorizer": TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
        "model": LogisticRegression(C=1.0, max_iter=1000),
        "params": {"model": "LogisticRegression", "C": 1.0, "ngram_range": "1,2", "max_features": 20000},
    },
    {
        "name": "NaiveBayes",
        "vectorizer": TfidfVectorizer(max_features=10000, ngram_range=(1, 1)),
        "model": MultinomialNB(alpha=0.1),
        "params": {"model": "MultinomialNB", "alpha": 0.1, "ngram_range": "1,1", "max_features": 10000},
    },
    {
        "name": "LinearSVC",
        "vectorizer": TfidfVectorizer(max_features=10000, ngram_range=(1, 1)),
        "model": LinearSVC(C=1.0, max_iter=2000),
        "params": {"model": "LinearSVC", "C": 1.0, "ngram_range": "1,1", "max_features": 10000},
    },
    {
        "name": "LinearSVC-bigrams",
        "vectorizer": TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
        "model": LinearSVC(C=0.5, max_iter=2000),
        "params": {"model": "LinearSVC", "C": 0.5, "ngram_range": "1,2", "max_features": 20000},
    },
]

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("doc-topic-classifier")

results = []

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):

        # Build pipeline: TF-IDF → classifier
        pipeline = Pipeline([
            ("tfidf", exp["vectorizer"]),
            ("clf",   exp["model"]),
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log to MLflow
        mlflow.log_params(exp["params"])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        results.append({
            "run": exp["name"],
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
        })

        print(f"  {exp['name']:<30}  accuracy={accuracy:.4f}  f1={f1:.4f}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n== Results Summary ==")
df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
print(df.to_string(index=False))
best = df.iloc[0]
print(f"\n  Best model: {best['run']}  (accuracy={best['accuracy']}, f1={best['f1']})")
print("\nRun  'mlflow ui'  then open  http://localhost:5000  to explore all runs.")
