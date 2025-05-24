import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/movies_features.csv')

# select features and target
features = ['genre_score', 'movie_writer_quality', 'movie_actor_score', 'budget', 'director_quality']
target = 'averageRating'

# feature labels, will be used for coefficients table/file
feature_labels = {
    'genre_score': 'Genre Score',
    'movie_writer_quality': 'Writer Quality',
    'movie_actor_score': 'Actor Score',
    'budget': 'Budget (Millions USD)',
    'director_quality': 'Director Quality'
}

x = df[features]
y = df[target]

# normalize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# initialize RandomForest model
model = RandomForestRegressor(random_state=42)

# parameters used to find the combination with highest performance
# n_estimators -> number of trees
# max_depth -> tree depth

param_grid = {'n_estimators': [100, 200],'max_depth': [10, 20, None],'min_samples_split': [2, 5]}

# initialize grid search with 3 fold cross validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1, verbose=1)

# perform grid search
grid_result = grid.fit(x_scaled, y)

#extract metrics and data about the best model
best_model = grid_result.best_estimator_
best_params = grid_result.best_params_
best_mse = -grid_result.best_score_
best_rmse = np.sqrt(best_mse)

# split the data 80/20
x_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
best_model.fit(x_train, y_train)
y_pred = best_model.predict(X_test)

# calculate performance metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


feature_importances = pd.DataFrame({ 'Feature': [feature_labels.get(f, f) for f in features], 'Importance': best_model.feature_importances_})

print("\nBest Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

print("\nBest Model Performance:")
print(f"Average MSE: {best_mse:.3f}")
print(f"Average RMSE: {best_rmse:.3f}\n")

print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}\n")

print("\nFeature Importances:")
print(feature_importances.to_string(index=False))

# create dictinoaary for metrics and parameters where values will be stored, and later exported
results = { 'Metric': ['Average MSE ', 'Average RMSE', 'R² Score', 'MSE', 'RMSE'],'Value': [best_mse, best_rmse, r2, mse, rmse]}
params = {'Parameter': ['n_estimators', 'max_depth', 'min_samples_split'],'Value': [best_params['n_estimators'], best_params['max_depth'], best_params['min_samples_split']]}

results_df = pd.DataFrame(results)
params_df = pd.DataFrame(params)

feature_importances_df = feature_importances
results_df['Value'] = results_df['Value'].round(3)
feature_importances_df['Importance'] = feature_importances_df['Importance'].round(3)

# create a latex table for feature imetrics
ltx_results = results_df.to_latex(index=False,float_format="%.3f",caption="Random Forest Model Performance for Predicting IMDb Ratings (2020--2025, 3-Fold Cross-Validation with Grid Search)",label="tab:rf_results",column_format='lr',header=['Metric', 'Value'],escape=True)

# create a latex table for feature importances
ltx_importances = feature_importances_df.to_latex(index=False,float_format="%.3f",caption="Feature Importances for Random Forest Model Predicting IMDb Ratings (2020--2025)",label="tab:rf_feature_importances",column_format='lr',header=['Feature', 'Importance'],escape=True)

# save latex tables for metrics and feature importances
with open('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/rf_results.tex', 'w') as f:
    f.write(ltx_results)
with open('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/rf_feature_importances.tex', 'w') as f:
    f.write(ltx_importances)

# save model metrics, parameters and features importances into respective CSV files
results_df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/rf_results.csv',index=False)
params_df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/rf_params.csv',index=False)
feature_importances_df.to_csv(  'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/rf_feature_importances.csv', index=False)

# Best Hyperparameters:
# max_depth: 10
# min_samples_split: 5
# n_estimators: 100
#
# Best Model Performance (Grid Search, 3-Fold CV):
# Average MSE (CV): 0.237
# Average RMSE (CV): 0.487
#
# Final Model Performance (80/20 Split):
# R² Score: 0.909
# MSE: 0.230
# RMSE: 0.480
#
# Feature Importances:
#               Feature  Importance
#           Genre Score    0.002826
#        Writer Quality    0.112024
#           Actor Score    0.022423
# Budget (Millions USD)    0.002805
#      Director Quality    0.859922