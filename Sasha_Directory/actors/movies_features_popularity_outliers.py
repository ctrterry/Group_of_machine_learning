import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# load the data
df = pd.read_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Terry_Directory/Data/Merege_data_with_Sasha/ready_to_train.csv')

# features = ['top_3_actor_scores', 'budget']
features = ['top_3_actor_scores']


results = []
# print(len(df[feature]))
# calculate feature statistics and roound ot 4 decimals
for feature in features:
    stats = df[feature].describe()
    min_val = round(stats['min'], 4)
    q1 = round(stats['25%'], 4)
    median = round(stats['50%'], 4)
    q3 = round(stats['75%'], 4)
    max_val = round(stats['max'], 4)

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr


    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
    total_outliers = len(outliers)

    results.append({
        'Feature': "movie_actor_popularity",
        'Min': min_val,
        'Q1': q1,
        'Median': median,
        'Q3': q3,
        'Max': max_val,
        'IQR_Outliers': total_outliers
    })

summary_table = pd.DataFrame(results)
print(summary_table)


summary_table.to_csv("C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Sasha_Directory/actors/features_outliers_summary.csv",mode='a', index=False)