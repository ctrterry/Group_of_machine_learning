import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/actor_scores_updated.csv')

# get actors with movie count between 1 and 10
df_filtered = df[(df['total_movies'] >= 1) & (df['total_movies'] <= 10)]

# get actors per movie count
total_movies_counts = df_filtered['total_movies'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=total_movies_counts.index, y=total_movies_counts.values, color='skyblue')
plt.title('Number of Actors per Movie Count (1–10 Movies)')
plt.xlabel('Total Movies')
plt.ylabel('Number of Actors')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()