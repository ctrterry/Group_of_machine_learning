import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

metrics = {
    'Model': ['Linear Regression(5-Fold)', 'ANN(3-Fold)', 'Random Forest(3-Fold)'],
    'MSE': [0.322, 0.275, 0.230],
    'RMSE': [0.568, 0.524, 0.480],
    'R²': [0.874, 0.892, 0.909]
}

df = pd.DataFrame(metrics)

models = df['Model']
metrics = ['MSE', 'RMSE', 'R²']
colors = ['blue', 'orange', 'green']
bar_width = 0.25
index = np.arange(len(metrics))


plt.figure(figsize=(10, 6))

for i, model in enumerate(models):
    values = df.loc[df['Model'] == model, metrics].values.flatten()
    plt.bar(index + i * bar_width, values, bar_width, label=model, color=colors[i])

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Model Comparison: Linear Regression(5-Fold), ANN(3-Fold), Random Forest(3-Fold)')
plt.xticks(index + bar_width, metrics)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

plt.savefig('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/model_comparison_bar_graph.png', dpi=300, bbox_inches='tight')

plt.savefig('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/model_comparison_bar_graph.pdf', format='pdf', bbox_inches='tight')

plt.close()

df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/model_comparison_metrics.csv',index=False)