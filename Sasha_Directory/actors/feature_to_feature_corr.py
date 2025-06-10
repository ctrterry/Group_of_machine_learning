import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Terry_Directory/Data/Merege_data_with_Sasha/ready_to_train.csv')

features = ['genre_score', 'movie_writer_quality', 'movie_actor_score','budget','director_quality','top_3_actor_scores']
# features of the movies_features table/file, with target being average movie rating
feature_labels = {'genre_score': 'Genre',
                  'movie_writer_quality':'Writer',
                  'movie_actor_score':'Actor',
                  'director_quality': 'Director',
                   'top_3_actor_scores': 'Popularity',
                  'budget': ' Budget'
}
df['budget'] = df['budget'] / 1_000_000

corr_matrix = df[features].corr(method='pearson')

corr_matrix_labeled = corr_matrix.rename(columns=feature_labels, index=feature_labels)

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix_labeled, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
plt.title('Feature-to-Feature Correlation Matrix (POP Set)', fontsize=12, fontweight='bold')
# plt.xlabel('Features', fontsize=12, fontweight='bold')
# plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.xticks( fontsize=12, )
plt.yticks(fontsize=9, )
plt.tight_layout()
plt.show()