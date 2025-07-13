import os
import pandas as pd 

df=pd.read_csv('datasets/dataset_catalog.csv')
base_dir = 'datasets'

meta_rows=[]

for i in range(len(df)):
    dataset_id = df.at[i, 'dataset_id']
    relative_path = df.at[i, 'file_path']
    full_path = os.path.join(base_dir, relative_path)
    dfx = pd.read_csv(full_path)

    dataset_id=dataset_id

    n_rows,n_colums=dfx.shape

    num_cols=dfx.select_dtypes(include='number')
    cat_cols=dfx.select_dtypes(exclude='number')

    n_num_cols=len(num_cols)
    n_cat_cols=len(cat_cols)

    pct_cat_cols= (n_cat_cols/n_colums * 100)

    pct_missing= df.isna().mean().mean() * 100

    if n_cat_cols>0:
        avg_cardinality=cat_cols.nunique().mean()
    else:
        avg_cardinality=0
        
    if n_num_cols > 0:
            avg_skewness = num_cols.skew().abs().mean()
            avg_kurtosis = num_cols.kurtosis().mean()
    else:
            avg_skewness = 0
            avg_kurtosis = 0

    meta_rows.append({
        'dataset_id': dataset_id,
        'n_rows': n_rows,
        'n_columns': n_colums,
        'n_num_cols': n_num_cols,
        'n_cat_cols': n_cat_cols,
        'pct_cat_cols': pct_cat_cols,
        'pct_missing': pct_missing,
        'avg_cardinality': avg_cardinality,
        'avg_skewness': avg_skewness,
        'avg_kurtosis': avg_kurtosis
    })



meta_features=pd.DataFrame(meta_rows)
print(len(meta_features))
meta_features.to_csv(os.path.join(base_dir,'meta_features.csv'),index=False)