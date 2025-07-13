import argparse
import pandas as pd
from itertools import product
import joblib

# Load trained model and training feature columns
model = joblib.load('saved_model/meta_mayhem.pkl')
training_columns = joblib.load('saved_model/feature_columns.pkl')

def meta_feature_extractor(file_path):
    print("Extracting meta-features...")
    df = pd.read_csv(file_path)

    n_rows, n_columns = df.shape
    num_cols = df.select_dtypes(include='number')
    cat_cols = df.select_dtypes(exclude='number')

    n_num_cols = len(num_cols.columns)
    n_cat_cols = len(cat_cols.columns)
    pct_cat_cols = (n_cat_cols / n_columns) * 100 if n_columns else 0
    pct_missing = df.isna().mean().mean() * 100

    avg_cardinality = cat_cols.nunique().mean() if n_cat_cols > 0 else 0
    avg_skewness = num_cols.skew().abs().mean() if n_num_cols > 0 else 0
    avg_kurtosis = num_cols.kurtosis().mean() if n_num_cols > 0 else 0

    return [
        n_rows, n_columns, n_num_cols, n_cat_cols,
        pct_cat_cols, pct_missing, avg_cardinality,
        avg_skewness, avg_kurtosis
    ]

def input_generator(meta_features):
    print("Generating all pipeline combinations...")
    final_input = []

    imputations = ['mean', 'median', 'knn', 'most_frequent']
    encodings = ['onehot', 'target', 'ordinal', 'freq']
    selections = ['none', 'pca_5', 'pca_10', 'variance_threshold']
    downstream_models = ['lightgbm', 'randomforest', 'logistic', 'xgboost', 'extratrees']

    pipelines = [
        {'trans_imputation': i, 'trans_encoding': e, 'trans_selection': s, 'downstream_model': m}
        for i, e, s, m in product(imputations, encodings, selections, downstream_models)
    ]

    for pipeline in pipelines:
        combined = meta_features + list(pipeline.values())
        final_input.append(combined)

    return final_input, pipelines

def pipeline_predictor(final_input_list, pipelines):
    print("Predicting pipeline scores...")

    # build DataFrame: numeric features + pipeline columns
    df = pd.DataFrame(final_input_list, columns=[
        'n_rows', 'n_columns', 'n_num_cols', 'n_cat_cols',
        'pct_cat_cols', 'pct_missing', 'avg_cardinality',
        'avg_skewness', 'avg_kurtosis',
        'trans_imputation', 'trans_encoding', 'trans_selection', 'downstream_model'
    ])

    # encode pipeline columns
    df_encoded = pd.get_dummies(df)

    # align to training columns: add missing columns with 0
    df_encoded = df_encoded.reindex(columns=training_columns, fill_value=0)

    # predict all
    predictions = model.predict(df_encoded)
    df['predicted_score'] = predictions

    # return best row
    best_row = df.loc[df['predicted_score'].idxmax()]
    return best_row

def main():
    print("Meta Mayhem Tool")
    parser = argparse.ArgumentParser(description="Predict best pipeline for a new dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Path to CSV dataset")
    args = parser.parse_args()

    meta_features = meta_feature_extractor(args.dataset)
    final_input, pipelines = input_generator(meta_features)
    best_pipeline = pipeline_predictor(final_input, pipelines)

    print("\nBest pipeline predicted:")
    print(best_pipeline)

if __name__ == "__main__":
    main()
