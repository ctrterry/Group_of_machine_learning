import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Updated read path
df = pd.read_csv('movies_features_with_social_Metadata.csv')

# ✅ Added new feature: top_3_actors_fb_likes_avg
features = [
    'genre_score',
    'movie_writer_quality',
    'movie_actor_score',
    'budget',
    'director_quality',
    'top_3_actors_fb_likes_avg'
]
target = 'averageRating'

# ✅ Add label for new feature
feature_labels = {
    'genre_score': 'Genre Score',
    'movie_writer_quality': 'Movie Writer Quality',
    'movie_actor_score': 'Movie Actor Score',
    'budget': ' Budget (Millions USD)',
    'director_quality': 'Director Quality',
    'top_3_actors_fb_likes_avg': 'Top 3 Avg Actor Metadata From Facebook'
}

correlations = {'Feature': [], 'Correlation': []}

for f in features:
    correlations['Feature'].append(feature_labels[f])
    if f == 'budget':
        data = df[f] / 1_000_000  # Convert budget to millions
    else:
        data = df[f]
    corr = data.corr(df[target], method='pearson')
    correlations['Correlation'].append(corr)

# Create DataFrame
corr_df = pd.DataFrame(correlations)
corr_df['Correlation'] = corr_df['Correlation'].round(3)

# Create bar chart
plt.figure(figsize=(8, 6))
sns.barplot(
    x='Feature',
    y='Correlation',
    data=corr_df,
    palette='coolwarm',
    hue='Correlation',
    dodge=False
)

plt.ylabel('Pearson Correlation with IMDb Rating')
plt.xlabel('Feature')
plt.title('Correlation of Movie Features with IMDb Ratings (2020–2025)')
plt.ylim(-1, 1)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

# ✅ Save results
plt.savefig(
    'feature_corr_bar_chart.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()

corr_df.to_csv(
    'ann_results.csv',
    index=False
)
