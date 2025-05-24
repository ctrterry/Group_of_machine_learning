import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/movies_features.csv')

# features of the movies_features table/file
features = ['genre_score', 'movie_writer_quality', 'movie_actor_score', 'budget', 'director_quality']

# create a map where statistics for all the features will be collected
stats = {'Feature' : [], 'Q1' : [], 'Median' : [], 'Mean' : [] ,'Q3' : [], 'Std_dev' : []}

# populate statistics for each of the features in 'features'
for feature in features:
        stats['Feature'].append(feature)

        if feature == 'budget':
                data = df[feature] /1_000_000
        stats['Q1'].append(df[feature].quantile(.25).round(2))
        stats['Median'].append(df[feature].median().round(2))
        stats['Mean'].append(df[feature].mean().round(2))
        stats['Q3'].append(df[feature].quantile(.75).round(2))
        stats['Std_dev'].append(df[feature].std().round(2))

stats_df = pd.DataFrame(stats)

# convert statistics to a latex tables
latex_table = stats_df.to_latex(index=False, float_format="%.2f", caption="Summary Statistics of Movie Features (2020–2025)", label="tab:movie_features_stats", column_format='lrrrrr', header=['Feature', 'Q1', 'Median', 'Mean', 'Q3', 'Std. Dev.'],escape=True)

with open('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/movies_features_stats.tex', 'w') as f:
    f.write(latex_table)

# export to a csv file
stats_df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/movies_features_stats.csv', index=False)

print("Feature Statistics:")
print(stats_df.to_string(index=False))
