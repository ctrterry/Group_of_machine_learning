import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/data/writer_regression_data.csv')

# features which will be used to generate writer scores
features = ['writer_experience', 'writer_quality']
X = df[features]
y = df['target_rating']

# split the dataset into 80/20 train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize features (values are not in same range)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# train Ridge regression model
model = Ridge(alpha=1.0)
model.fit(x_train_scaled, y_train)

# make prediction for movie rating based on writer features
X_scaled = scaler.transform(X)
df['writer_score'] = model.predict(X_scaled)

y_pred = model.predict(x_test_scaled)

# export to a csv file
df[['writer_id', 'writer_score']].to_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/writer_scores_output.csv', index=False)

# calculate performance metrics (R2, mse, rmse)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# print feature weights and performance metrics
print("\nFeature weights:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")

print(f"R^2 score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

