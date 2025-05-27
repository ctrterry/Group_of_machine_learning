import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# After add socail media metadata
# Linear:        R² Score: 0.890 ,MSE: 0.310, RMSE: 0.557
# ANN,           R² Score: 0.905, MSE: 0.240, RMSE: 0.490
# Random Forest: R² Score: 0.905, MSE: 0.240, RMSE: 0.490

metrics = {
    'Model': ['Linear Regression(5-Fold)', 'ANN(3-Fold)', 'Random Forest(3-Fold)'],
    'MSE': [0.310, 0.240, 0.240],
    'RMSE': [0.557, 0.490, 0.490],
    'R²': [0.890, 0.905, 0.905]
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
plt.title('Model Comparison with social Metadata: Linear Regression(5-Fold), ANN(3-Fold), Random Forest(3-Fold)')
plt.xticks(index + bar_width, metrics)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.savefig(
    '/Users/terrychen/Desktop/Group_of_machine_learning/Terry_Directory/New_work/result/model_comparison_bar_graph.png',
    dpi=300,
    bbox_inches='tight'
)
plt.tight_layout()
plt.show()

# Save results
plt.close()

# plt.savefig('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/model_comparison_bar_graph.png', dpi=300, bbox_inches='tight')

# plt.savefig('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/model_comparison_bar_graph.pdf', format='pdf', bbox_inches='tight')

# plt.close()

# df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/project_imdb_rating/Group_of_machine_learning/actors/model_comparison_metrics.csv',index=False)