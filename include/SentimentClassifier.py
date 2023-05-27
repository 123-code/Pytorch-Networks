import os
from dotenv import load_dotenv
import kaggle 
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 

api = KaggleApi()
api.authenticate()

api.competition_download_file('sentiment-analysis-on-movie-reviews','test.tsv.zip',path='./Datasets')
api.competition_download_file('sentiment-analysis-on-movie-reviews','train.tsv.zip',path='./Datasets')

with zipfile.ZipFile('./Datasets/SentimentClassifier/test.tsv.zip', 'r') as zip_ref:
    zip_ref.extractall('./Datasets')


with zipfile.ZipFile('../Datasets/SentimentClassifier/.tsv.zip', 'r') as zip_ref:
    zip_ref.extractall('./Datasets')


df = pd.read_csv('./Datasets/SentimentClassifier/train.tsv', sep='\t')
df['Sentiment'].value_counts().plot(kind='bar')

seq_len = 512
num_samples = len(df)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokens = tokenizer(df['Phrase'].tolist(),max_length=seq_len,truncation=True,padding='max_length',add_special_tokens=True,return_tensors='np')
tokens.keys()
print(tokens['attention_mask'])

with open('movie-xids.npy','wb') as f:
    np.save(f,tokens['input_ids'])

with open('movie-xmask.npy','wb') as f:
    np.save(f,tokens['attention_mask'])

arr = df['Sentiment'].values
arr.max() + 1

labels = np.zeros((num_samples,arr.max() + 1))
labels[np.arrange(num_samples),arr] = 1

with open('movie-labels.npy','wb') as f:
    np.save(f,labels)


with open('movie-xids.npy','rb') as f:
    Xids = np.load(f, allow_pickle=True)


with open('movie-labels.npy','rb') as f:
    labels = np.load(f, allow_pickle=True)

with open('movie-xmask.npy','rb') as f:
    Xmask = np.load(f, allow_pickle=True)

dataset = tf.data.Dataset.from_tensor_slices((Xids,labels))

dataset.take(1)

{inp_ids,masks},outputs

def map(inp_ids,masks,labels):
  return{'input_ids': inp_ids,'attention_mask':masks},labels

data = dataset.map(map)