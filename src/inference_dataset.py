import csv
import pandas as pd
from ast import literal_eval
import random

initial_df = pd.read_csv("data/original/company_sentences_euro.csv")
inference_df = pd.DataFrame(columns = ['sentence'])

uplimit = 11000
local_uplimit = 100

for index, row in initial_df.iterrows():
    if len(inference_df) < uplimit: 
            rand = random.randint(0, 1)
            j = 0
            if rand == 1: 
                for sent in literal_eval(row.sentences):
                    if j < local_uplimit and j < len(literal_eval(row.sentences)): 
                            inference_df = inference_df.append({'sentence' : sent}, ignore_index = True)
                            j+=1
                    else:
                            break
    else:
        break



inference_df.to_csv("inference_data.csv", index = False)
                    

