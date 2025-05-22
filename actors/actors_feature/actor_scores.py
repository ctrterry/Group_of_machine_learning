import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error,  r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_movies = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/movie_actor_features_2020_2025_top_5_7.csv')

X = df_movies[['cast_avg_movie_rating', 'cast_avg_votes_per_movie', 'cast_avg_total_movies',
              'cast_avg_rating_std_dev', 'cast_avg_high_rated_movie_count', 'cast_avg_career_span',
              'cast_avg_recent_avg_rating']]
y = df_movies['movie_rating']

# Split data into 80/20 train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train Ridgre regression model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# get and print weights
weights = model.coef_
feature_names = X.columns
weights_df = pd.DataFrame({'feature': feature_names, 'weight': weights})
print("Feature Weights from Ridge Regression (2020–2025, Top 5–7 Actors):")
print(weights_df)

# calculate performance metric
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Ridge Regression Evaluation (2020–2025, Top 7 Actors):")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# actual vs. predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Movie Rating')
plt.ylabel('Predicted Movie Rating')
plt.title('Predicted vs. Actual Movie Ratings (2020–2025, Top 7 Actors)')
plt.show()

# residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals (2020–2025, Top 7 Actors)')
plt.show()


# Normalize the entire dataset
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0)

# 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mse_scores)
mae_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')
r2_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')

print("5-Fold Cross-Validation Results (2020–2025, Top 7 Actors):")
print(f"Mean MSE: {-np.mean(mse_scores):.4f} (±{np.std(mse_scores):.4f})")
print(f"Mean RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
print(f"Mean MAE: {-np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})")
print(f"Mean R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")

# Load the actor-level dataset
df_actors = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/actor_scores.csv')

actor_features = df_actors[['avg_movie_rating', 'avg_votes_per_movie', 'total_movies',
                           'rating_std_dev', 'high_rated_movie_count', 'career_span',
                           'recent_avg_rating']]
actor_features.columns = ['cast_avg_movie_rating', 'cast_avg_votes_per_movie', 'cast_avg_total_movies',
                         'cast_avg_rating_std_dev', 'cast_avg_high_rated_movie_count', 'cast_avg_career_span',
                         'cast_avg_recent_avg_rating']

actor_features_scaled = scaler.transform(actor_features)
actor_scores = np.dot(actor_features_scaled, weights)

# add actor scores to data frame
df_actors['actor_score'] = actor_scores
min_score = df_actors['actor_score'].min()

# set the lowest score to be 0, rather than the negative value
if min_score < 0:
    df_actors['actor_score'] = df_actors['actor_score'] - min_score

# save to a CSV file
df_actors.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/actor_scores_and_features.csv', index=False)


