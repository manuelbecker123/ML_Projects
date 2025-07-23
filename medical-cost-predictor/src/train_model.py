import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# --- 1. Load Data ---
df = pd.read_csv('data/processed/insurance_processed.csv')
X = df.drop('log_charges', axis=1)
y = df['log_charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Create a Preprocessing Pipeline ---
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features)
    ], remainder='passthrough')

# --- 3. Define Models and Hyperparameter Grids ---
models = {
    'ElasticNet': ElasticNet(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42)
}
param_grids = {
    'ElasticNet': {'model__alpha': [0.1, 0.5, 1.0], 'model__l1_ratio': [0.1, 0.5, 0.9]},
    'RandomForest': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20]},
    'XGBoost': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1], 'model__max_depth': [3, 5]},
    'LightGBM': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1], 'model__num_leaves': [20, 31]}
}

# --- 4. Train, Tune, and Evaluate Models ---
results = []
all_residuals = {}
for name, model in models.items():
    print(f"--- Tuning and Training {name} ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    results.append({'Model': name, 'R-squared': r2, 'MAE': mae, 'RMSE': rmse, 'Best Estimator': best_model})
    all_residuals[name] = y_test_orig - y_pred_orig

# --- 5. Display and Analyze Results ---
results_df = pd.DataFrame(results).sort_values(by='RMSE', ascending=True)
results_df_display = results_df.drop(columns=['Best Estimator'])
print("\n--- Model Comparison ---")
print(results_df_display.to_string(index=False))
best_model_name = results_df.iloc[0]['Model']
best_model_pipeline = results_df.iloc[0]['Best Estimator']
print(f"\nBest performing model: {best_model_name}")

# --- 6. Advanced Visualizations ---
print("\nGenerating advanced performance plots...")
plt.style.use('seaborn-v0_8-whitegrid')
residuals_df = pd.DataFrame(all_residuals)
plt.figure(figsize=(15, 8))
sns.stripplot(data=residuals_df, jitter=0.3, alpha=0.6, size=5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Jitter Plot for All Models', fontsize=16)
plt.ylabel('Residuals (Actual - Predicted) [$]', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=45)
plt.savefig('reports/figures/residuals_jitter_plot.png')
print("Saved 'residuals_jitter_plot.png' to reports/figures/")

if hasattr(best_model_pipeline.named_steps['model'], 'feature_importances_'):
    feature_names = best_model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model_pipeline.named_steps['model'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title(f'Top 10 Feature Importances for {best_model_name}', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.savefig('reports/figures/feature_importance.png')
    print("Saved 'feature_importance.png' to reports/figures/")

# --- 7. Save Results and Best Model ---
# Save the results summary to a text file
with open('reports/model_evaluation_summary.txt', 'w') as f:
    f.write("--- Model Evaluation Summary ---\n\n")
    f.write(results_df_display.to_string(index=False))
print("\nSaved model evaluation summary to reports/model_evaluation_summary.txt")

# Save the best model pipeline
with open('models/best_model_pipeline.pkl', 'wb') as f:
    pickle.dump(best_model_pipeline, f)
print(f"Best model pipeline ({best_model_name}) saved to models/best_model_pipeline.pkl")