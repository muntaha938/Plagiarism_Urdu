import csv
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext
# NLP Preprocessing
from gensim.utils import simple_preprocess

dataset = pd.read_csv("Train_data.csv")
ds = pd.read_csv("Test_data.csv")

dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(lambda x: ' '.join(simple_preprocess(x)))
ds.iloc[:, 0] = ds.iloc[:, 0].apply(lambda x: ' '.join(simple_preprocess(x)))

dataset.iloc[:, 1] = dataset.iloc[:, 1].apply(lambda x: '__label__' + x)
ds.iloc[:, 1] = ds.iloc[:, 1].apply(lambda x: '__label__' + x)

dataset.to_csv('train.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")

ds.to_csv('test.txt', 
                                     index = False, 
                                     sep = ' ',
                                     header = None, 
                                     quoting = csv.QUOTE_NONE, 
                                     quotechar = "", 
                                     escapechar = " ")




model = fasttext.train_supervised('train.txt', wordNgrams = 2)
model.test('test.txt')



