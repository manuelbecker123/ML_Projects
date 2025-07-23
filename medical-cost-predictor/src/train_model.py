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

print("--- Script Start: Running Full Model Comparison ---")

# --- 1. Load Data ---
df_raw = pd.read_csv('data/raw/insurance.csv')
df_processed = pd.read_csv('data/processed/insurance_processed.csv')

# --- 2. Define Feature Sets and Targets ---
X = df_processed.drop('log_charges', axis=1)
targets = {
    "With Log Transform": df_processed['log_charges'],
    "Without Log Transform": df_raw['charges']
}

# --- 3. Define Preprocessing, Models, and Hyperparameter Grids ---
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numerical_features)],
    remainder='passthrough'
)

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

# --- 4. Train, Tune, and Evaluate All Scenarios ---
all_results = []
all_residuals = {}

for scenario_name, y in targets.items():
    print(f"\n===== EVALUATING SCENARIO: {scenario_name} =====")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        print(f"--- Tuning and Training {model_name} ---")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        use_log = "With Log" in scenario_name
        if use_log:
            y_test_eval = np.expm1(y_test)
            y_pred_eval = np.expm1(y_pred)
        else:
            y_test_eval = y_test
            y_pred_eval = y_pred
            
        rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
        mae = mean_absolute_error(y_test_eval, y_pred_eval)
        r2 = r2_score(y_test_eval, y_pred_eval)
        
        result_key = f"{model_name} ({scenario_name.split(' ')[0]} Log)"
        all_results.append({
            'Experiment': result_key,
            'R-squared': r2, 'MAE': mae, 'RMSE': rmse,
            'Best Estimator': best_model
        })
        all_residuals[result_key] = y_test_eval - y_pred_eval

# --- 5. Analyze Results and Find Best Overall Model ---
results_df = pd.DataFrame(all_results).sort_values(by='RMSE', ascending=True)
results_df_display = results_df.drop(columns=['Best Estimator'])
print("\n--- Overall Model Comparison ---")
print(results_df_display.to_string(index=False))

best_experiment = results_df.iloc[0]
best_model_name = best_experiment['Experiment']
best_model_pipeline = best_experiment['Best Estimator']
print(f"\nBest performing model overall: {best_model_name}")

# --- 6. Generate All Visualizations ---
print("\nGenerating performance plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# RMSE Comparison Bar Plot
plt.figure(figsize=(15, 8))
sns.barplot(x='RMSE', y='Experiment', data=results_df, palette='plasma', orient='h')
plt.title('Model RMSE Comparison: With vs. Without Log Transformation', fontsize=16)
plt.xlabel('Root Mean Squared Error (RMSE) in $', fontsize=12)
plt.ylabel('Model and Scenario', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/rmse_comparison_by_scenario.png')
print("Saved 'rmse_comparison_by_scenario.png' to reports/figures/")

# Residuals Jitter Plot
residuals_df = pd.DataFrame(all_residuals)
plt.figure(figsize=(15, 8))
sns.stripplot(data=residuals_df, jitter=0.35, alpha=0.6, size=5, orient='h')
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Residuals Jitter Plot for All Experiments', fontsize=16)
plt.xlabel('Residuals (Actual - Predicted) [$]', fontsize=12)
plt.ylabel('Model and Scenario', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/residuals_jitter_plot_all.png')
print("Saved 'residuals_jitter_plot_all.png' to reports/figures/")

# NEW PLOT: Side-by-Side Feature Importance Comparison for XGBoost
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
fig.suptitle('XGBoost Feature Importance: With vs. Without Log Transformation', fontsize=18, y=1.02)

scenarios_to_plot = ['XGBoost (With Log)', 'XGBoost (Without Log)']
for i, scenario in enumerate(scenarios_to_plot):
    # Find the specific model pipeline from the results
    model_pipeline = results_df[results_df['Experiment'] == scenario].iloc[0]['Best Estimator']
    
    # Extract importances and feature names
    importances = model_pipeline.named_steps['model'].feature_importances_
    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    # Plot
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=axes[i], palette='viridis')
    axes[i].set_title(scenario, fontsize=14)
    axes[i].set_xlabel('Importance Score', fontsize=12)
    axes[i].set_ylabel('') # Remove y-label for cleaner look

axes[0].set_ylabel('Feature', fontsize=12) # Set y-label only on the first plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('reports/figures/feature_importance_comparison.png')
print("Saved 'feature_importance_comparison.png' to reports/figures/")


# --- 7. Save Final Results and Best Model ---
with open('reports/model_evaluation_summary.txt', 'w') as f:
    f.write("--- Overall Model Evaluation Summary ---\n\n")
    f.write(results_df_display.to_string(index=False))
print("\nSaved final evaluation summary to 'reports/model_evaluation_summary.txt'")

with open('models/best_model_pipeline.pkl', 'wb') as f:
    pickle.dump(best_model_pipeline, f)
print(f"Best overall model pipeline ({best_model_name}) saved to models/best_model_pipeline.pkl")

print("\n--- Script End ---")
