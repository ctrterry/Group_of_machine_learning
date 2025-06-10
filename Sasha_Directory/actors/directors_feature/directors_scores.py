import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('/actors/directors_feature/director_regression_data.csv')

# extract features which will be used to train the ridge regression model
features = ['director_experience', 'director_quality', 'director_total_votes']
x = df[features]
y = df['target_rating']

# create a random 80/20 test/train split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# normalize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# train the Ridge regression model
model = Ridge(alpha=1.0)
model.fit(x_train_scaled, y_train)

# generate predictions for the test data
y_pred = model.predict(x_test_scaled)

# calculate performance metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# make prediction for the entire dataset, will be exported as CSV, and incorporated into the movies dataset
x_scaled = scaler.transform(x)
df['director_score'] = model.predict(x_scaled)

# export results to CSV
df[['tconst', 'director_id', 'director_score']].to_csv('/actors/directors_features/director_scores_output.csv', index=False)

# print final coefficients for the model
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")

# print the performance metrics (R^2, MSE, RMSE)
print(f"R^2 score: {r2:.4f}")
print(f" MSE: {mse:.4f}")
print(f"Test set RMSE: {rmse:.4f}")