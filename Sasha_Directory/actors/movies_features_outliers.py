import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# load the data
df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Sasha_Directory/actors/movies_features.csv')

# features for outlier analysis
features = ['genre_score', 'movie_writer_quality', 'movie_actor_score', 'budget', 'director_quality']


# function to calculating outliers bounds and counts
def detect_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
    return iqr_outliers,lower_bound, upper_bound


# outlier ananlysis per feature
for feature in features:
    print(f"\nOutlier Analysis for {feature}:")
    iqr_outliers,  lower_bound, upper_bound = detect_outliers(df, feature)
    print(f"Summary Statistics:\n{df[feature].describe()}")
    print(f"IQR Outliers (count: {len(iqr_outliers)}):")
    print(iqr_outliers)
    print(f"IQR Bounds: Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")


# ANALYSIS OF BUDGET OUTLIEER
df['budget_log'] = np.log1p(df['budget'])
z_scores = np.abs((df['budget_log'] - df['budget_log'].mean()) / df['budget_log'].std())
print(f"Z-score Outliers (count: {len(df[z_scores > 3])}):")
print(f"Z-score Documentary Outliers (count: {len(df[z_scores > 3]['budget'] == 4388240.25)}):") #2951


# FURTHER ANALYSIS OF GENRE OUTLIERS
Q1 = 5.800594
Q3 = 6.222951
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# movies below lower bound
below_lower = df[df['genre_score'] < lower_bound]
num_below_lower = len(below_lower)
print("Number of movies below lower bound: ", num_below_lower)

num_doc_below= len(below_lower['genre_score'] == 4.975294)
print("Number of Documentaries below lower bound: ", num_doc_below) #2376
# print(below_lower['genre_score'])

# movies above upper bound
above_upper = df[df['genre_score'] > upper_bound]
num_above_upper = len(above_upper)
print("Number of movies above upper bound: ", num_above_upper)

# horror movies above upper bound
num_horror_above= len(above_upper['genre_score'] == 7.213336)
print("Number of Horrors above upper bound: ", num_horror_above) #3639
# print(above_upper['genre_score'])

