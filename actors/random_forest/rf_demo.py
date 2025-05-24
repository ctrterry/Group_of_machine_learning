import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


# select features and target
features = ['genre_score', 'movie_writer_quality', 'movie_actor_score', 'budget', 'director_quality']

# load save model and scaler
model = joblib.load('./rf_model.joblib')
scaler = joblib.load('./rf_scaler.joblib')

# mock data
movie_data = {
    'genre_score': 6.246229143988683,
    'movie_writer_quality': 7.700000286102295,
    'movie_actor_score': 8.912113350831817,
    'budget': 45110614.14841664,
    'director_quality': 6.920000171661377
}
target = 6.8
df_movie = pd.DataFrame([movie_data], columns=features)

df_movie_scaled = scaler.transform(df_movie)

prediction = model.predict(df_movie_scaled)
print("Six: The Fully Animated Musical")
print(f"\nActual averageRating: {target:.2f}")
print(f"\nPredicted averageRating: {prediction[0]:.2f}")