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
X_dev, X_test, y_dev, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
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
        "model": MultinomialNB(alpha=0.5),
        "params": {"model": "MultinomialNB", "alpha": 0.5, "ngram_range": "1,1", "max_features": 10000},
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
best_pipeline = None
best_validation = -1
best_name = None

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
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        mlflow.set_tag("evaluation_split", "validation")
        if accuracy > best_validation:
            best_pipeline, best_validation, best_name = pipeline, accuracy, exp["name"]

        # Log to MLflow
        mlflow.log_params(exp["params"])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.sklearn.log_model(pipeline, name="model")

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

# ── Alpha tuning runs (NaiveBayes hyperparameter search) ─────────────────────
print("\nRunning alpha tuning experiment for NaiveBayes...")
alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

for alpha in alphas:
    with mlflow.start_run(run_name=f"NaiveBayes-alpha-{alpha}"):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 1))),
            ("clf",   MultinomialNB(alpha=alpha)),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        mlflow.set_tag("evaluation_split", "validation")
        if accuracy > best_validation:
            best_pipeline, best_validation, best_name = pipeline, accuracy, f"NaiveBayes-alpha-{alpha}"

        mlflow.log_params({"model": "MultinomialNB", "alpha": alpha, "experiment_type": "alpha_tuning"})
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.sklearn.log_model(pipeline, name="model")

        print(f"  alpha={alpha:<6} accuracy={accuracy:.4f}  f1={f1:.4f}")

# Select using validation only, refit on train+validation, then evaluate once.
best_pipeline.fit(X_dev, y_dev)
with mlflow.start_run(run_name="final-held-out-test"):
    predictions = best_pipeline.predict(X_test)
    score = accuracy_score(y_test, predictions)
    mlflow.log_param("selected_model", best_name)
    mlflow.set_tag("evaluation_split", "test")
    mlflow.log_metric("test_accuracy", score)
    mlflow.log_metric("test_f1_weighted", f1_score(y_test, predictions, average="weighted"))
    mlflow.sklearn.log_model(best_pipeline, name="model")
    print(f"Final held-out test: {best_name}, accuracy={score:.4f}")
print("Run python -m mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("14 validation runs + 1 final held-out test run. Do not tune using the final test score.")
