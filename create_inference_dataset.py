
import pandas as pd


df = pd.read_csv("df_partnership.csv")

df_pos = df.loc[df.word == " Yes"]


for index,value in df_pos.iterrows():
     value.token_values = value.token_values[41:47]
df_pos.token_values = df_pos.token_values.astype(float)
df_pos = df_pos.sort_values(by = ['token_values'], ascending = False).reset_index()
new_df_pos = pd.DataFrame(columns = ['sentence', 'score'] )
for i in range(10):
     new_df_pos = new_df_pos.append({'sentence': df_pos.iloc[i].sentence,'score': df_pos.iloc[i].token_values}, ignore_index = True)

new_df_pos.to_json("tokp_finance_positive.json")


df_neg = df.loc[df.word == " No"]
for index,value in df_neg.iterrows():
     value.token_values = value.token_values[41:47]
df_neg.token_values = df_neg.token_values.astype(float)
df_neg = df_neg.sort_values(by = ['token_values'], ascending = False).reset_index()
new_df_neg = pd.DataFrame(columns = ['sentence', 'score'] )
for i in range(10):
     new_df_new = new_df_neg.append({'sentence': df_neg.iloc[i].sentence,'score': df_neg.iloc[i].token_values}, ignore_index = True)

new_df_neg.to_json("tokp_finance_negative.json")