import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

df1=pd.read_csv('datasets/meta_features.csv')
df2=pd.read_csv('datasets/meta_dataset.csv')
pd.merge(df1,df2,on='dataset_id').to_csv('datasets/mayhem_meta.csv',index=False)

df=pd.read_csv('datasets/mayhem_meta.csv')

# --- Load data
df['avg_cardinality'] = df['avg_cardinality'].fillna(df['avg_cardinality'].median())
df['avg_skewness'] = df['avg_skewness'].fillna(df['avg_skewness'].median())
df['avg_kurtosis'] = df['avg_kurtosis'].fillna(df['avg_kurtosis'].median())
df['pct_missing'] = df['pct_missing'].fillna(df['pct_missing'].median())
df['pct_cat_cols'] = df['pct_cat_cols'].fillna(df['pct_cat_cols'].median())

# --- Drop unused columns
df = df.drop(columns=['dataset_id'])

# --- Encode categorical pipeline columns
df = pd.get_dummies(df, columns=['trans_imputation', 'trans_encoding', 'trans_selection', 'downstream_model'])
#
# # --- Split features/target
X = df.drop(columns=['val_score'])
y = df['val_score']
#
# # --- Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # --- Train RandomForest
model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
model.fit(X_train, y_train)
#
# # --- Predict & evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
#
print(f"R² on test: {r2:.4f}")
print(f"RMSE on test: {rmse:.4f}")

joblib.dump(list(X_train.columns), 'saved_model/feature_columns.pkl')
joblib.dump(model, 'saved_model/meta_mayhem.pkl')