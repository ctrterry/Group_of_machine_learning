import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from scikeras.wrappers import KerasRegressor
from keras.src.callbacks import EarlyStopping

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

df = pd.read_csv('../Terry_Directory/Data/Merege_data_with_Sasha/movies_features_with_social_Metadata.csv')

# select features and target
features = ['genre_score', 'movie_writer_quality', 'movie_actor_score', 'budget', 'director_quality','top_3_actors_fb_likes_avg']
target = 'averageRating'

# feature labels, will be used for coefficients table/file
feature_labels = {
    'genre_score': 'Genre Score',
    'movie_writer_quality': 'Writer Quality',
    'movie_actor_score': 'Actor Score',
    'budget': 'Budget (Millions USD)',
    'director_quality': 'Director Quality',
    'top_3_actors_fb_likes_avg' : 'Top 3 Avg Actor Metadata From Facebook'
}

x = df[features]
y = df[target]

# normalize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

def create_model(neuron_count=64, activation='relu', momentum=0.6):
    model = Sequential()
    model.add(Input(shape=(len(features),)))
    model.add(Dense(neuron_count, activation=activation))
    model.add(Dense(1))
    optimizer = SGD(learning_rate=0.01, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mse')
    return model

keras_model = KerasRegressor(model=create_model, activation='relu', verbose=0, momentum=0.6, neuron_count=64)
print(keras_model.get_params().keys())
param_grid = {
    'neuron_count': [32, 64, 128],
    'activation': ['tanh', 'relu'],
    'momentum': [0.6, 0.9],
    'batch_size': [32, 64],
}

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1,  verbose=0)

grid_result = grid.fit(x_scaled, y, callbacks=[early_stopping], validation_split=0.2)

# extract best model data
best_model = grid_result.best_estimator_
best_params = grid_result.best_params_
best_mse = -grid_result.best_score_
best_rmse = np.sqrt(best_mse)

# split train/test data into 80/20 split (final model)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# train the model and make predictions for movie ratings
best_model.fit(x_train, y_train, callbacks=[early_stopping], validation_split=0.2)
y_pred = best_model.predict(x_test)

# calculate performance metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nBest Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print("Epochs: 20 (fixed)")
print("Learning Rate: 0.01 (fixed)")

print("\nBest Model Performance:")
print(f"Average MSE: {best_mse:.3f}")
print(f"Average RMSE: {best_rmse:.3f}")

print("\nFinal Model Performance:")
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")

# create a dictionary to save average and final performance metrics
results = {
    'Metric': ['Average MSE', 'Average RMSE', ' R² Score', 'MSE', 'RMSE'],
    'Value': [best_mse, best_rmse, r2, mse, rmse]
}
# create a dictionary with best hyperparameter combination
params = {'Parameter': ['Neurons', 'Activation', 'Learning Rate', 'Momentum', 'Batch Size', 'Epochs'],
    'Value': [best_params['neuron_count'], best_params['activation'], 0.01, best_params['momentum'], best_params['batch_size'], 20]}

results_df = pd.DataFrame(results)
params_df = pd.DataFrame(params)
results_df['Value'] = results_df['Value'].round(3)

# crreate a latex table with results of the ANN models
latex_table = results_df.to_latex(index=False, float_format="%.3f", caption="ANN Model Performance for Predicting IMDb Ratings (2020--2025, 3-Fold Cross-Validation with Grid Search)", label="tab:ann_results", column_format='lr',header=['Metric', 'Value'], escape=True)


# save results and params as CSV file for later use
results_df.to_csv( '../Terry_Directory/New_work/result/ann_results.csv',index=False)