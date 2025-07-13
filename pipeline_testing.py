import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from itertools import product
import os
import time

# Load dataset catalog
dataset_catalog = pd.read_csv('datasets/dataset_catalog.csv')

# Define transformations and models
imputations = ['mean', 'median', 'knn', 'most_frequent']
encodings = ['onehot', 'target', 'ordinal', 'freq']
selections = ['none', 'pca_5', 'pca_10', 'variance_threshold']
downstream_models = ['lightgbm', 'randomforest', 'logistic', 'xgboost', 'extratrees']

pipelines = [
    {'trans_imputation': i, 'trans_encoding': e, 'trans_selection': s}
    for i, e, s in product(imputations, encodings, selections)
]

meta_rows = []
base_dir = 'datasets'

batch_size = 10
total_batches = (len(dataset_catalog) + batch_size - 1) // batch_size

def evaluate_pipeline(pipeline, model_name, dataset_id, X, y, num_cols, cat_cols):
    try:
        X_transformed = X.copy()

        # Imputation
        if pipeline['trans_imputation'] == 'mean':
            for col in num_cols:
                X_transformed[col] = X_transformed[col].fillna(X_transformed[col].mean())
        elif pipeline['trans_imputation'] == 'median':
            for col in num_cols:
                X_transformed[col] = X_transformed[col].fillna(X_transformed[col].median())
        elif pipeline['trans_imputation'] == 'most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
            X_transformed[num_cols] = imputer.fit_transform(X_transformed[num_cols])
        elif pipeline['trans_imputation'] == 'knn':
            if X_transformed.shape[0] > 10000:
                for col in num_cols:
                    X_transformed[col] = X_transformed[col].fillna(X_transformed[col].mean())
            else:
                imputer = KNNImputer()
                X_transformed[num_cols] = imputer.fit_transform(X_transformed[num_cols])

        # Fill missing categoricals
        for col in cat_cols:
            mode = X_transformed[col].mode()
            if not mode.empty:
                X_transformed[col] = X_transformed[col].fillna(mode[0])

        # Encoding
        if pipeline['trans_encoding'] == 'onehot':
            low_card_cols = [col for col in cat_cols if X_transformed[col].nunique() <= 20]
            X_transformed = pd.get_dummies(X_transformed, columns=low_card_cols, drop_first=True)
        elif pipeline['trans_encoding'] == 'target':
            for col in cat_cols:
                target_mean = y.groupby(X_transformed[col]).mean()
                X_transformed[col] = X_transformed[col].map(target_mean)
        elif pipeline['trans_encoding'] == 'ordinal':
            encoder = OrdinalEncoder()
            X_transformed[cat_cols] = encoder.fit_transform(X_transformed[cat_cols])
        elif pipeline['trans_encoding'] == 'freq':
            for col in cat_cols:
                freq = X_transformed[col].value_counts() / len(X_transformed)
                X_transformed[col] = X_transformed[col].map(freq)

        # Feature selection
        if pipeline['trans_selection'] == 'pca_5':
            if X_transformed.shape[1] >= 5:
                pca = PCA(n_components=5)
                X_transformed = pd.DataFrame(pca.fit_transform(X_transformed), index=X_transformed.index)
            else:
                return None
        elif pipeline['trans_selection'] == 'pca_10':
            if X_transformed.shape[1] >= 10:
                pca = PCA(n_components=10)
                X_transformed = pd.DataFrame(pca.fit_transform(X_transformed), index=X_transformed.index)
            else:
                return None
        elif pipeline['trans_selection'] == 'variance_threshold':
            selector = VarianceThreshold(threshold=0.01)
            X_transformed = pd.DataFrame(selector.fit_transform(X_transformed), index=X_transformed.index)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.3, random_state=42
        )

        # Choose and train model
        if model_name == 'lightgbm':
            model = LGBMClassifier(n_estimators=100, verbose=-1)
        elif model_name == 'randomforest':
            model = RandomForestClassifier(n_estimators=100, verbose=0)
        elif model_name == 'logistic':
            model = LogisticRegression(max_iter=5000, verbose=0)
        elif model_name == 'xgboost':
            model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', verbosity=0)
        elif model_name == 'extratrees':
            model = ExtraTreesClassifier(n_estimators=100, verbose=0)

        model.fit(X_train, y_train)
        val_score = model.score(X_test, y_test)

        return {
            'dataset_id': dataset_id,
            'trans_imputation': pipeline['trans_imputation'],
            'trans_encoding': pipeline['trans_encoding'],
            'trans_selection': pipeline['trans_selection'],
            'downstream_model': model_name,
            'val_score': val_score
        }

    except Exception:
        return None

# Loop in batches
for batch_idx in range(total_batches):
    start_time = time.time()
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, len(dataset_catalog))
    batch = dataset_catalog.iloc[batch_start:batch_end]

    for _, row in batch.iterrows():
        dataset_id = row['dataset_id']
        file_path = row['file_path']
        target_column = row['target_column']

        try:
            df = pd.read_csv(os.path.join(base_dir, file_path))
            df = df.dropna(subset=[target_column])
            y = df[target_column]
            X = df.drop(columns=[target_column])

            num_cols = X.select_dtypes(include=['number']).columns
            cat_cols = X.select_dtypes(include=['object', 'category']).columns

            for pipeline in pipelines:
                for model_name in downstream_models:
                    res = evaluate_pipeline(pipeline, model_name, dataset_id, X, y, num_cols, cat_cols)
                    if res is not None:
                        meta_rows.append(res)

        except Exception:
            continue

    elapsed = time.time() - start_time
    print(f"Batch {batch_idx + 1} / {total_batches} complete in {elapsed:.2f} seconds")

# Save final meta-dataset
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv('datasets/meta_dataset.csv', index=False)
print("Saved datasets/meta_dataset.csv")