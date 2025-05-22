import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/actor_scores_updated.csv')

# create bins per 5-year career span
bins = range(0, 85, 5)
labels = [f'{i}-{i+5}' for i in range(0, 80, 5)]  # '0-5', '5-10', ..., '75-80'
df['career_span_bin'] = pd.cut(df['career_span'], bins=bins, labels=labels, include_lowest=True)

# get number of actors per each bin
career_span_counts = df['career_span_bin'].value_counts().sort_index()

# create a histogram
plt.figure(figsize=(10, 6))
sns.barplot(x=career_span_counts.index, y=career_span_counts.values, color='skyblue')
plt.title('Number of Actors per 5-Year Career Span Interval')
plt.xlabel('Career Span (Years)')
plt.ylabel('Number of Actors')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()