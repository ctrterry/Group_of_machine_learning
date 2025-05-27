import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/terrychen/Desktop/Group_of_machine_learning/Terry_Directory/Data/Merege_data_with_Sasha/movies_features_with_social_Metadata_with_log.csv')

# features of the movies_features table/file
features = ['genre_score', 'movie_writer_quality', 'movie_actor_score', 'budget', 'director_quality', 'log_top3_fb']

# create a map where statistics for all the features will be collected
stats = {'Feature' : [], 'Q1' : [], 'Median' : [], 'Mean' : [] ,'Q3' : [], 'Std_dev' : []}

# populate statistics for each of the features in 'features'
for feature in features:
        stats['Feature'].append(feature)

        if feature == 'budget':
                data = df[feature] / 1_000_000
        else:
                data = df[feature]
                
        stats['Q1'].append(round(data.quantile(.25), 2))
        stats['Median'].append(round(data.median(), 2))
        stats['Mean'].append(round(data.mean(), 2))
        stats['Q3'].append(round(data.quantile(.75), 2))
        stats['Std_dev'].append(round(data.std(), 2))

stats_df = pd.DataFrame(stats)

# export to a csv file
stats_df.to_csv('/Users/terrychen/Desktop/Group_of_machine_learning/Terry_Directory/New_work/result/movies_features_stats.csv', index=False)

print("Feature Statistics:")
print(stats_df.to_string(index=False))
