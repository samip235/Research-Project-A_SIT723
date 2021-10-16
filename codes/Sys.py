import numpy as np
import pandas as pd


df = pd.read_csv('C:/Users/samip/Documents/Trimister5/ResearchProject/data/spambase.csv', header = 0)
print(df.shape)
max_val = max(df)
min_val = min(df)


new_df = pd.DataFrame(columns=list(df.columns))
max_dim = df.shape[1] - 1
for i in range(0,max_dim):
    list_temp = df.iloc[:,i]
    min_val = min(list_temp)
    max_val = max(list_temp)
    synthetic_data = list(np.random.uniform(min_val,max_val,df.shape[0]))
    column_name = df.iloc[:,i].name
    new_df[column_name] = pd.Series(synthetic_data).astype('float').round(6)
    new_df['Class'] = 1

df['Class'] = 0
final_df = pd.concat([df, new_df],ignore_index=True)

print(final_df)

