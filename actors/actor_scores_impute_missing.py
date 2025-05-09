import pandas as pd

df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/actor_scores_filtered.csv')

# chekc number of data points
total_rows = len(df)
print(f"Total number of rows: {total_rows}")

# create flags for missing features
df['is_rating_std_dev_missing'] = df['rating_std_dev'].isnull().astype(int)
df['is_recent_avg_rating_missing'] = df['recent_avg_rating'].isnull().astype(int)

# impute missing rating_std_dev with the median
median_rating_std_dev = df['rating_std_dev'].median()
df['rating_std_dev'] = df['rating_std_dev'].fillna(median_rating_std_dev)

# impute missing recent_avg_rating with avg_movie_rating
df['recent_avg_rating'] = df['recent_avg_rating'].fillna(df['avg_movie_rating'])

# verify no missing values remain
missing_data = df.isnull().sum()
print("Number of missing values after imputation:")
print(missing_data)

# save as new CSV file
df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/actor_scores.csv', index=False)
