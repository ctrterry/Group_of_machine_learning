import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('/actors/movies_features.csv')

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

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []
mse_scores = []
rmse_scores = []
coefficients = []

# 5-fold cross validation for linear  regression
for fold, (train_id, test_id) in enumerate(kf.split(x), 1):
    # split the data 80/20
    x_train=  x.iloc[train_id]
    x_test =x.iloc[test_id]
    y_train = y.iloc[train_id]
    y_test =  y.iloc[test_id]

    # normalize train and test feature data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # train the model
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    # calculate model performance metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # collect metrics for avg and std dev calculation
    r2_scores.append(r2)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    coefficients.append(model.coef_)

    print(f"R² Score: {r2:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print()

# calcute average and standard deviation for each metric
avg_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
avg_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
avg_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

# calculate and save average feature weights across 5 folds
avg_coeff = np.mean(coefficients, axis=0).round(3)
coeffs = {'Feature': [feature_labels[feat] for feat in features],'Coefficient': avg_coeff}
coeff_df = pd.DataFrame(coeffs)

# create a latex table with feature weights
latex_table = coeff_df.to_latex(index=False,float_format="%.3f",caption="Linear Regression Coefficients for Predicting IMDb Ratings (2020--2025, 5-Fold Cross-Validation)",label="tab:linear_regression_coeffs",column_format='lr', header=['Feature', 'Coefficient'], escape=True )

# save latex file
with open('/actors/linear_regression/linear_regression_coeffs.tex', 'w') as f:
    f.write(latex_table)

# create a dictionary with performance averages and standard deviations
results = {
    'Metric': ['Average R² Score', 'Std R² Score', 'Average MSE', 'Std MSE', 'Average RMSE', 'Std RMSE'],
    'Value': [avg_r2, std_r2, avg_mse, std_mse, avg_rmse, std_rmse]
}
results_df = pd.DataFrame(results)
results_df['Value'] = results_df['Value'].round(3)
results_df.to_csv(   'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/linear_regression_results.csv',index=False)
coeff_df.to_csv( 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/linear_regression_coeffs.csv',index=False)