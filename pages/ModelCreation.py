import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

DATA_DIR = "Evals/"

st.set_page_config(layout="wide")

def load_json(path):
    with open(DATA_DIR + path, "r") as f:
        return json.load(f)

# --------------------------------------------------
# Model registry (maps directly to JSON files)
# --------------------------------------------------
MODELS = {
    "BERT": {
        "baseline": "bert_results.json",
        "best": "bert_tuning_best.json",
        "tuning": "bert_tuning_results.json",
    },
    "Linear SVM": {
        "baseline": "linear_svm_results.json",
        "best": "linear_svm_tuning_best.json",
        "tuning": "linear_svm_tuning_results.json",
    },
    "Logistic Regression":{
        "baseline": "logistic_regression_results.json",
        "best": "logistic_regression_tuning_best.json",
        "tuning": "logistic_regression_tuning_results.json"
    }
}

st.set_page_config(layout="wide")
st.title("🧠 Model Evaluation")

# --------------------------------------------------
# Model selector
# --------------------------------------------------
model_name = st.selectbox("Select Model", list(MODELS.keys()))
st.title(f"{model_name} Toxicity Classifier")

files = MODELS[model_name]

baseline = load_json(files["baseline"])
best = load_json(files["best"])
tuning_results = load_json(files["tuning"])

# ==================================================
# 1️⃣ Macro Metrics (EXACT JSON VALUES)
# ==================================================

st.header("📊 Macro Metrics")

macro_df = pd.DataFrame(
    {
        "Metric": ["macro_precision", "macro_recall", "macro_f1", "macro_auc_pr"],
        "Baseline": [
            baseline["macro_precision"],
            baseline["macro_recall"],
            baseline["macro_f1"],
            baseline["macro_auc_pr"],
        ],
        "Best Tuned": [
            best["macro_precision"],
            best["macro_recall"],
            best["macro_f1"],
            best["macro_auc_pr"],
        ],
    }
)

st.dataframe(macro_df, width='stretch')

fig, ax = plt.subplots()
ax.bar(macro_df["Metric"], macro_df["Baseline"], label="Baseline")
ax.bar(
    macro_df["Metric"],
    macro_df["Best Tuned"],
    label="Best Tuned",
    alpha=0.7,
)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.legend()
st.pyplot(fig)

# ==================================================
# 2️⃣ Per-Label AUC-PR (EXACT JSON VALUES)
# ==================================================

st.header("🏷️ Per-Label AUC-PR")

labels = baseline["per_label_auc_pr"].keys()

auc_df = pd.DataFrame(
    {
        "Label": labels,
        "Baseline": [baseline["per_label_auc_pr"][l] for l in labels],
        "Best Tuned": [best["per_label_auc_pr"][l] for l in labels],
    }
)

st.dataframe(auc_df, width='stretch')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(auc_df["Label"], auc_df["Baseline"], marker="o", label="Baseline")
ax.plot(auc_df["Label"], auc_df["Best Tuned"], marker="o", label="Best Tuned")
ax.set_ylim(0, 1)
ax.set_ylabel("AUC-PR")
ax.legend()
plt.xticks(rotation=30)
st.pyplot(fig)

# ==================================================
# 3️⃣ Hyperparameter Tuning Results (FULL GRID)
# ==================================================

st.header("⚙️ Hyperparameter Tuning Results")

tuning_df = pd.DataFrame(tuning_results)
st.dataframe(tuning_df, width="stretch")

# Choose x-axis depending on model
if model_name == "BERT":
    x = tuning_df["lr"]
    xlabel = "Learning Rate (log scale)"
    xscale = "log"
else:
    x = tuning_df["C"]
    xlabel = "C (Regularization Strength)"
    xscale = "log"

fig, ax = plt.subplots()
ax.scatter(
    x,
    tuning_df["macro_f1"],
    s=tuning_df.get("max_features", pd.Series([100]*len(tuning_df))),
)
ax.set_xscale(xscale)
ax.set_xlabel(xlabel)
ax.set_ylabel("Macro F1")
ax.set_title("Macro F1 Across Hyperparameters")
st.pyplot(fig)

# ==================================================
# 4️⃣ Best Configuration Summary
# ==================================================

st.header("🏆 Best Configuration")

best_config = {k: v for k, v in best.items() if k not in ["per_label_auc_pr"]}

st.json(best_config)

# ==================================================
# 5️⃣ Notes
# ==================================================

st.header("📝 Interpretation Notes")
st.text_area(
    "Observations",
    placeholder=(
        f"- {model_name} shows different precision/recall tradeoffs\n"
        "- Macro metrics hide large per-label variance\n"
        "- Rare classes (threat, identity_hate) remain difficult\n"
        "- Tuning improves AUC-PR more than macro F1"
    ),
    height=140,
)