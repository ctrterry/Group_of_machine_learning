import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
actor_scores = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Sasha_Directory/actors/actors_feature/actor_scores_and_features.csv')

actors_popularity= pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Sasha_Directory/actors/unique_actors_popularity.csv')


# Merge datasets on nconst (from actors_popularity) and actor (from actor_scores)
common_actors = pd.merge(
    actors_popularity[['actor_id', 'facebook_likes']],
    actor_scores[['actor', 'actor_score']],
    left_on='actor_id',
    right_on='actor',
    how='inner'
)

common_actors = common_actors.dropna(subset=['facebook_likes', 'actor_score'])

# Calculate Pearson correlation
correlation = common_actors['facebook_likes'].corr(common_actors['actor_score'])

print(f"Correlation between facebook_likes and actor_score: {correlation:.4f}")

# # Optional: Save the merged data for further analysis
# common_actors.to_csv('common_actors_correlation.csv', index=False)