import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Terry_Directory/Data/Merege_data_with_Sasha/ready_to_train.csv')

features = ['top_3_actor_scores', 'budget']
# features of the movies_features table/file, with target being average movie rating

target = 'averageRating'

# map of feature labels, will be used in the bar plot
feature_labels = {
                  'top_3_actor_scores':'Movie Actor Popularity',
                  'budget': ' Budget (Millions USD)'
                }

correlations = {'Feature':[], 'Correlation':[]}

for f in features:
    correlations['Feature'].append(feature_labels[f])
    # convert budget to millions of USD
    if f == 'budget':
        data = df[f] / 1_000_000
    else:
        data = df[f]
    corr = data.corr(df[target], method='pearson')
    correlations['Correlation'].append(corr)

# create data frame for feature rating correlations
corr_df = pd.DataFrame(correlations)
# round to 3 decimal places
corr_df['Correlation'] = corr_df['Correlation'].round(3)

print(corr_df)
# create bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Feature',  y='Correlation', data=corr_df, palette='coolwarm', hue='Correlation', dodge=False)

plt.ylabel('Pearson Correlation with IMDb Rating')
plt.xlabel('Feature')
plt.title('Correlation of Movie Features with IMDb Ratings (2020–2025)')
plt.ylim(-1, 1)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
# rotate feature labels for readability
# plt.xticks(rotation=45, ha='right')
plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()


# 0  Movie Actor Popularity        0.012
# 1   Budget (Millions USD)       -0.139


# plt.savefig(
#     'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/feature_corr_bar_chart.png',
#     dpi=300,
#     bbox_inches='tight'
# )
plt.close()
# corr_df.to_csv(
#     'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/feature_correlations.csv',
#     index=False
# )