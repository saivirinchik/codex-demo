# =============================================================================
# CDOCS POC — Notebook 04: Train and Evaluate ML Classifier
# =============================================================================
#
# TARGET LABEL: gfa_name (Governing Functional Area)
#
# Trains TF-IDF + Logistic Regression to predict which GFA a document
# belongs to, based on LLM-enriched features.
#
# Three experiments:
#   Baseline : title + object_name (no LLM)
#   LLM-only : summary + key_terms only
#   Combined : doc_subtype + classification + summary + key_terms
#
# INPUT: _dev.edl_app_dev_ops_gsc_ai.datamapping_poc_training (Notebook 03)
# =============================================================================


# -----------------------------------------------------------------------------
# CELL 1 — Install dependencies (run once)
# -----------------------------------------------------------------------------
# %pip install scikit-learn matplotlib seaborn
# dbutils.library.restartPython()


# -----------------------------------------------------------------------------
# CELL 2 — Imports
# -----------------------------------------------------------------------------
import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from pyspark.sql import SparkSession

matplotlib.use("Agg")
spark = SparkSession.builder.getOrCreate()

print("Imports OK")


# -----------------------------------------------------------------------------
# CELL 3 — Configuration
# -----------------------------------------------------------------------------
INPUT_TABLE = "_dev.edl_app_dev_ops_gsc_ai.datamapping_poc_training"

# Where to save the trained model for Notebook 05 demo
MODEL_DIR   = "/dbfs/FileStore/cdocs_poc/model/"
MODEL_PATH  = MODEL_DIR + "cdocs_gfa_classifier.joblib"
LABELS_PATH = MODEL_DIR + "gfa_labels.json"
PLOT_DIR    = "/dbfs/FileStore/cdocs_poc/plots/"

TEST_SIZE           = 0.2
RANDOM_STATE        = 42
MIN_CLASS_SAMPLES   = 5
TFIDF_MAX_FEATURES  = 10_000
TFIDF_NGRAM_RANGE   = (1, 2)
TFIDF_MIN_DF        = 2
LR_MAX_ITER         = 1000
LR_CLASS_WEIGHT     = "balanced"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("Config OK")


# -----------------------------------------------------------------------------
# CELL 4 — Load training data into pandas
# Target: gfa_name
# -----------------------------------------------------------------------------
raw_df = spark.sql(f"""
    SELECT
        source_id,
        gfa_name,
        feature_text,
        baseline_text,
        summary,
        classification,
        doc_subtype,
        llm_confidence
    FROM {INPUT_TABLE}
    WHERE gfa_name IS NOT NULL
      AND length(trim(feature_text)) > 0
""").toPandas()

print(f"Rows loaded       : {len(raw_df)}")
print(f"Unique GFAs       : {raw_df['gfa_name'].nunique()}")
print(f"\nGFA distribution:\n{raw_df['gfa_name'].value_counts()}")


# -----------------------------------------------------------------------------
# CELL 5 — Filter rare GFA classes
# Classes with too few samples cause StratifiedKFold to fail.
# -----------------------------------------------------------------------------
class_counts  = raw_df["gfa_name"].value_counts()
valid_classes = class_counts[class_counts >= MIN_CLASS_SAMPLES].index

dropped = raw_df[~raw_df["gfa_name"].isin(valid_classes)]["gfa_name"].unique()
if len(dropped) > 0:
    print(f"\n⚠ Dropping {len(dropped)} GFA(s) with < {MIN_CLASS_SAMPLES} samples: {list(dropped)}")

df = raw_df[raw_df["gfa_name"].isin(valid_classes)].copy().reset_index(drop=True)
print(f"\nFinal: {len(df)} rows | {df['gfa_name'].nunique()} GFAs")


# -----------------------------------------------------------------------------
# CELL 6 — Train/test split (stratified by GFA)
# -----------------------------------------------------------------------------
X_feature  = df["feature_text"]
X_baseline = df["baseline_text"]
y          = df["gfa_name"]

(X_feat_train, X_feat_test,
 X_base_train, X_base_test,
 y_train,      y_test) = train_test_split(
    X_feature, X_baseline, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Train: {len(X_feat_train)} | Test: {len(X_feat_test)}")
print(f"\nTrain GFA distribution:\n{y_train.value_counts()}")
print(f"\nTest GFA distribution:\n{y_test.value_counts()}")


# -----------------------------------------------------------------------------
# CELL 7 — Pipeline factory and experiment runner
# -----------------------------------------------------------------------------
def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=LR_MAX_ITER,
            class_weight=LR_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            solver="lbfgs",
            multi_class="auto",
        )),
    ])


def run_experiment(name, X_train, X_test, y_train, y_test):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc         = accuracy_score(y_test, y_pred)
    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_str  = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nAccuracy   : {acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\n{report_str}")

    return {
        "name": name, "accuracy": acc, "macro_f1": macro_f1,
        "weighted_f1": weighted_f1, "report": report_str,
        "y_pred": y_pred, "pipeline": pipe,
    }


print("Experiment runner OK")


# -----------------------------------------------------------------------------
# CELL 8 — Build LLM-only text column
# -----------------------------------------------------------------------------
df["llm_only_text"] = (
    df["summary"].fillna("") + " " + df["doc_subtype"].fillna("")
).str.strip()

X_llm_train = df.loc[X_feat_train.index, "llm_only_text"]
X_llm_test  = df.loc[X_feat_test.index,  "llm_only_text"]

print(f"LLM-only text: {X_llm_train.str.len().gt(0).sum()} valid train rows")


# -----------------------------------------------------------------------------
# CELL 9 — Run all three experiments
# -----------------------------------------------------------------------------
results = {}

# Experiment 1: Baseline — title + filename, no LLM
results["baseline"] = run_experiment(
    "Baseline: Title + Object Name",
    X_base_train, X_base_test, y_train, y_test,
)

# Experiment 2: LLM summary only
if X_llm_train.str.len().gt(0).sum() > 10:
    results["llm_only"] = run_experiment(
        "LLM Summary Only",
        X_llm_train.fillna(""), X_llm_test.fillna(""), y_train, y_test,
    )

# Experiment 3: Combined features (the real model)
results["combined"] = run_experiment(
    "Combined: Subtype + Classification + Summary + Key Terms",
    X_feat_train, X_feat_test, y_train, y_test,
)


# -----------------------------------------------------------------------------
# CELL 10 — Comparison table
# -----------------------------------------------------------------------------
print(f"\n\n{'='*70}")
print("EXPERIMENT COMPARISON — Predicting Governing Functional Area")
print(f"{'='*70}")
print(f"{'Experiment':<55} {'Accuracy':>10} {'Macro F1':>10} {'Wt F1':>10}")
print("-"*88)

for r in results.values():
    print(f"{r['name']:<55} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>10.4f}")

print("-"*88)

delta_acc = results["combined"]["accuracy"]    - results["baseline"]["accuracy"]
delta_f1  = results["combined"]["macro_f1"]    - results["baseline"]["macro_f1"]
delta_wf1 = results["combined"]["weighted_f1"] - results["baseline"]["weighted_f1"]

print(f"\n{'LLM Uplift (Combined - Baseline)':<55} {delta_acc:>+10.4f} {delta_f1:>+10.4f} {delta_wf1:>+10.4f}")


# -----------------------------------------------------------------------------
# CELL 11 — Top-K accuracy
# If the correct GFA is in the model's top 2 or 3 predictions,
# that's still useful — the user picks from a short list.
# -----------------------------------------------------------------------------
print("\n=== Top-K Accuracy ===\n")

best_pipe = results["combined"]["pipeline"]
probas    = best_pipe.predict_proba(X_feat_test)
classes   = best_pipe.classes_

for k in [1, 2, 3]:
    topk_correct = sum(
        true_label in classes[np.argsort(p)[-k:]]
        for true_label, p in zip(y_test, probas)
    ) / len(y_test)
    print(f"Top-{k} accuracy: {topk_correct:.2%}")


# -----------------------------------------------------------------------------
# CELL 12 — Confusion matrix
# -----------------------------------------------------------------------------
def plot_cm(y_true, y_pred, title, path=None):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.2), max(6, len(labels)*1.0)))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, linewidths=0.3)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted GFA")
    ax.set_ylabel("True GFA")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()

plot_cm(y_test, results["combined"]["y_pred"],
        "Combined Model — GFA Prediction",
        os.path.join(PLOT_DIR, "cm_gfa_combined.png"))

plot_cm(y_test, results["baseline"]["y_pred"],
        "Baseline Model — GFA Prediction",
        os.path.join(PLOT_DIR, "cm_gfa_baseline.png"))


# -----------------------------------------------------------------------------
# CELL 13 — Error analysis
# Which GFA pairs does the model confuse?
# -----------------------------------------------------------------------------
print("\n=== Misclassification Analysis ===\n")

test_df          = df.iloc[X_feat_test.index].copy()
test_df["true"]  = y_test.values
test_df["pred"]  = results["combined"]["y_pred"]
test_df["wrong"] = test_df["true"] != test_df["pred"]

misclassified = test_df[test_df["wrong"]]
print(f"Test rows      : {len(test_df)}")
print(f"Misclassified  : {len(misclassified)}")
print(f"Accuracy       : {(~test_df['wrong']).mean():.2%}")

if len(misclassified) > 0:
    print("\nTop error patterns (true GFA → predicted GFA):")
    patterns = (
        misclassified.groupby(["true", "pred"]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )
    print(patterns.to_string(index=False))

    print("\nSample misclassified documents:")
    print(
        misclassified[["title", "true", "pred", "classification", "doc_subtype"]]
        .head(10)
        .to_string(index=False)
    )


# -----------------------------------------------------------------------------
# CELL 14 — Cross-validation
# More reliable than a single split with only ~130 documents.
# -----------------------------------------------------------------------------
print("\n=== 5-Fold Cross-Validation (Combined Features → GFA) ===\n")

cv_scores = cross_val_score(
    build_pipeline(),
    df["feature_text"], df["gfa_name"],
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1_weighted",
)

print(f"Folds   : {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"95% CI  : ({cv_scores.mean() - 2*cv_scores.std():.4f}, "
      f"{cv_scores.mean() + 2*cv_scores.std():.4f})")


# -----------------------------------------------------------------------------
# CELL 15 — Save trained model for Notebook 05 demo
# -----------------------------------------------------------------------------
joblib.dump(results["combined"]["pipeline"], MODEL_PATH)

with open(LABELS_PATH, "w") as f:
    json.dump(sorted(df["gfa_name"].unique().tolist()), f)

print(f"\nModel saved  : {MODEL_PATH}")
print(f"Labels saved : {LABELS_PATH}")
print(f"Download from: /files/cdocs_poc/model/cdocs_gfa_classifier.joblib")


# -----------------------------------------------------------------------------
# CELL 16 — POC Summary
# -----------------------------------------------------------------------------
print(f"""
{'='*70}
POC RESULTS — Governing Functional Area Prediction
{'='*70}

Architecture:
  PDF → multi-extractor (pdfplumber/pdftotext/pypdf)
     → text quality gate
     → AI Gateway LLM (summarization only, no GFA knowledge)
     → TF-IDF + Logistic Regression → GFA prediction

Training data: {len(df)} rows | {df['gfa_name'].nunique()} GFA categories

Baseline (title + filename):
  Accuracy    = {results['baseline']['accuracy']:.2%}
  Macro F1    = {results['baseline']['macro_f1']:.2%}
  Weighted F1 = {results['baseline']['weighted_f1']:.2%}

Combined (subtype + classification + LLM summary + key terms):
  Accuracy    = {results['combined']['accuracy']:.2%}
  Macro F1    = {results['combined']['macro_f1']:.2%}
  Weighted F1 = {results['combined']['weighted_f1']:.2%}

LLM uplift: {delta_f1:+.2%} Macro F1

Cross-validation: {cv_scores.mean():.2%} ± {cv_scores.std():.2%} Weighted F1
""")

if delta_f1 > 0.02:
    print("✓ LLM summaries improve GFA prediction. Recommend proceeding.")
elif delta_f1 > 0:
    print("~ Marginal improvement. Evaluate cost vs. benefit.")
else:
    print("~ No improvement. Investigate text quality and error patterns.")
