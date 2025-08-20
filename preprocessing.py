
# Plagiarised Data
import glob
import pandas as pd
import os
all_files =  glob .glob("UPPC Corpus/data/*.xml")
# print(all_files[0])

final_data = []
for i in all_files:
    data = pd.read_xml(i,xpath="/UPPC_document")
    Data = {"Sentence" : data["UPPC_document"].values,
    "Sentiment" : data["classification"].values
    }
    final_data.append(Data)
    # print(Data)


df_1 = pd.DataFrame(final_data)
train = df_1.iloc[:128]
test = df_1.iloc[129:]

train.to_csv("Train_data.csv", index= False)
test.to_csv("Test_data.csv", index= False)
