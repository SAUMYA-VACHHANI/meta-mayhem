# 🚀 Meta Mayhem: Auto‑ML Pipeline Recommender Using Meta‑Learning

**Meta Mayhem** is a Python toolkit that automatically predicts the best data preprocessing and ML pipeline for any tabular dataset — *before you train anything*.

It uses **meta‑learning**: a model trained on hundreds of real datasets to learn which pipelines work best on which data profiles.

---

## ✨ **Features**
✅ Extracts meta‑features from your dataset  
✅ Tests pipeline combinations, including:
- Imputation: `mean`, `median`, `knn`, `most_frequent`
- Encoding: `onehot`, `ordinal`, `target`, `frequency`
- Feature selection: `none`, `pca_5`, `pca_10`, `variance_threshold`
- Models: `lightgbm`, `xgboost`, `extratrees`, `randomforest`, `logistic`

✅ Predicts expected validation scores  
✅ Instantly recommends the best pipeline  
✅ Works offline after training

---

## ⚙ **Usage**
1️⃣ Clone the repo & install requirements:
```bash
 pip install -r requirements.txt
```

2️⃣ Run the CLI tool on your dataset:
```bash
 python mayhem_tool.py --dataset datasets/your_dataset.csv
```

