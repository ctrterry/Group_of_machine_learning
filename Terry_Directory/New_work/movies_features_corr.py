import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the new ready_to_train.csv file
df = pd.read_csv('/Users/terrychen/Desktop/Group_of_machine_learning/Terry_Directory/Data/Merege_data_with_Sasha/ready_to_train.csv')

# Define features and target
features = [
    'genre_score',
    'movie_writer_quality',
    'movie_actor_score',
    'budget',
    'director_quality',
    'top_3_actor_scores'  # Updated to match our new feature name
]
target = 'averageRating'

# Feature labels for the plot
feature_labels = {
    'genre_score': 'Genre Score',
    'movie_writer_quality': 'Movie Writer Quality',
    'movie_actor_score': 'Movie Actor Score',
    'budget': 'Budget (Millions USD)',
    'director_quality': 'Director Quality',
    'top_3_actor_scores': 'Top 3 Actor Social Media Score'  # Updated label
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
plt.figure(figsize=(10, 6))
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
plt.title('Correlation of Movie Features about IMDb Ratings with Social Metadata (2020–2025)')
plt.ylim(-1, 1)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend([], [], frameon=False)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save results
plt.savefig(
    '/Users/terrychen/Desktop/Group_of_machine_learning/Terry_Directory/New_work/result/feature_corr_bar_chart.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()

# Save correlation results
corr_df.to_csv(
    '/Users/terrychen/Desktop/Group_of_machine_learning/Terry_Directory/New_work/result/feature_correlations.csv',
    index=False
)

# Print correlation results
print("\nFeature Correlations with IMDb Rating:")
print(corr_df.to_string(index=False))
