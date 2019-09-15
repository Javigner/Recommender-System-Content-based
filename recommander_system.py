import numpy as np
import pandas as pd

df = pd.read_json("activities_dataset.json")
df_profile = pd.read_json("profile.json")

dataset = df.iloc[:, :29]
profile = df_profile.iloc[:, :29]
id_title = df.iloc[:, [29,30]]

profile = profile.T
X = profile.sum(axis=1)
recommendation_table_df = (dataset.dot(X)) / X.sum()
Score_csv = pd.concat([id_title, pd.DataFrame(recommendation_table_df)], axis=1)
Score_csv.columns = ['id', 'title', 'score']
Score_csv = Score_csv.sort_values(by='score', ascending=False)
Score_csv.to_csv('recommendation.csv', index=False)

final = pd.read_csv('recommendation.csv')