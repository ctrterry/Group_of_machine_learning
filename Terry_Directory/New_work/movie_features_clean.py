import pandas as pd

df = pd.read_csv('/Users/terrychen/Desktop/Group_of_machine_learning/Terry_Directory/New_work/result/movies_features.csv')

orig_size = len(df)
print(f"Original dataset size: {orig_size}")

df_clean = df.dropna()
clean_size = len(df_clean)
removed_count = orig_size - clean_size

print(f"Cleaned dataset size: {clean_size}")
print(f"Removed datapoints: {removed_count} ({(removed_count/orig_size)*100:.2f}%)")


df_clean.to_csv('/Users/terrychen/Desktop/Group_of_machine_learning/Terry_Directory/New_work/result/movies_features.csv',index=False)

#Original dataset size: 42790
#Cleaned dataset size: 42766
#Removed datapoints: 24 (0.06%)