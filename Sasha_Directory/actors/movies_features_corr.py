import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_exp = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Sasha_Directory/actors/movies_features.csv')

# features of the movies_features table/file, with target being average movie rating
features_exp = ['genre_score', 'movie_writer_quality', 'movie_actor_score', 'budget_imputed', 'director_quality']
target = 'averageRating'

# map of feature labels, will be used in the bar plot
feature_labels = {'genre_score': 'Genre',
                  'movie_writer_quality':'Writer',
                  'movie_actor_score':'Actor',
                  'director_quality': 'Director',
                   'top_3_actor_scores': 'Popularity',
                  'budget_imputed': ' Budget (Imp)',
                  'budget': ' Budget'
}

correlations = {'Feature':[], 'Correlation':[]}

for f in features_exp:
    correlations['Feature'].append(feature_labels[f])
    # convert budget to millions of USD
    if f == 'budget_imputed':
        data = df_exp[f] / 1_000_000
    else:
        data = df_exp[f]
    corr = data.corr(df_exp[target], method='pearson')
    correlations['Correlation'].append(corr)


df_pop = pd.read_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Terry_Directory/Data/Merege_data_with_Sasha/ready_to_train.csv')
features_pop = ['top_3_actor_scores', 'budget']

for f in features_pop:
    correlations['Feature'].append(feature_labels[f])
    # convert budget to millions of USD
    if f == 'budget':
        data = df_pop[f] / 1_000_000
    else:
        data = df_pop[f]
    corr = data.corr(df_pop[target], method='pearson')
    correlations['Correlation'].append(corr)


# create data frame for feature rating correlations
corr_df = pd.DataFrame(correlations)
# round to 3 decimal places
corr_df['Correlation'] = corr_df['Correlation'].round(3)

# create bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Feature',  y='Correlation', data=corr_df, palette='coolwarm', hue='Correlation', dodge=False)

plt.ylabel('Pearson Correlation', fontweight='bold', fontsize=12)
plt.xlabel('Feature', fontweight='bold', fontsize=12)
plt.title('Correlation of Movie Features with IMDb Ratings', fontweight='bold', fontsize=12)
plt.ylim(-0.3, 1)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)

# print horizontal dotted lines for each bar representing a feature
for index, row in corr_df.iterrows():
    plt.axhline(y=row['Correlation'], color='gray', linestyle=':', linewidth=1, alpha=0.7)

# for each bar/feature print the correlation value
for index, row in corr_df.iterrows():
    y_offset = 0.02 if row['Correlation'] >= 0 else -0.05
    plt.text(x=index, y=row['Correlation'] + y_offset, s=f"{row['Correlation']:.3f}",
             ha='center', va='bottom' if row['Correlation'] >= 0 else 'top', fontsize=12)

plt.legend([], [], frameon=False)

plt.tight_layout()
plt.show()


# plt.savefig(
#     'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Sasha_Directory/actors/feature_corr_bar_chart2.png',
#     dpi=300,
#     bbox_inches='tight'
# )
plt.close()
corr_df.to_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Sasha_Directory/actors/feature_correlations2.csv',
    index=False
)