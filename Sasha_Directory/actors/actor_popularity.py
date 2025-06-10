
import pandas as pd


data= pd.read_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/Terry_Directory/Data/social_metadata/raw_socialMedian_Metadata.csv')
# Create DataFrames for each actor role
actor1 = data[['actor_1_name', 'actor_1_facebook_likes']].rename(columns={'actor_1_name': 'actor_name', 'actor_1_facebook_likes': 'facebook_likes'})
actor2 = data[['actor_2_name', 'actor_2_facebook_likes']].rename(columns={'actor_2_name': 'actor_name', 'actor_2_facebook_likes': 'facebook_likes'})
actor3 = data[['actor_3_name', 'actor_3_facebook_likes']].rename(columns={'actor_3_name': 'actor_name', 'actor_3_facebook_likes': 'facebook_likes'})

# Concatenate all actor data
all_actors = pd.concat([actor1, actor2, actor3], ignore_index=True)

# Remove rows with missing or empty actor names
all_actors = all_actors.dropna(subset=['actor_name'])
all_actors = all_actors[all_actors['actor_name'].str.strip() != '']

# Group by actor_name and take the maximum facebook_likes
unique_actors = all_actors.groupby('actor_name')['facebook_likes'].max().reset_index()

# Sort by facebook_likes (descending) for readability
unique_actors = unique_actors.sort_values(by='facebook_likes', ascending=False)

print(unique_actors)

# Save to a new CSV file
unique_actors.to_csv('unique_actors_facebook_likes.csv', index=False)
#
# print("Created 'unique_actors_facebook_likes.csv' with unique actors and their Facebook likes.")
