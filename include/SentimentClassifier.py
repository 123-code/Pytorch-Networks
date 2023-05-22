import kaggle 
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from transformers import BertTokenizer
#import matplotlib.pyplot as plt

api = KaggleApi()
api.authenticate()

api.competition_download_file('sentiment-analysis-on-movie-reviews','test.tsv.zip',path='./Datasets')
api.competition_download_file('sentiment-analysis-on-movie-reviews','train.tsv.zip',path='./Datasets')

with zipfile.ZipFile('./Datasets/test.tsv.zip', 'r') as zip_ref:
    zip_ref.extractall('./Datasets')


with zipfile.ZipFile('./Datasets/train.tsv.zip', 'r') as zip_ref:
    zip_ref.extractall('./Datasets')


df = pd.read_csv('./Datasets/train.tsv', sep='\t')
df['Sentiment'].value_counts().plot(kind='bar')

seq_len = 512
num_samples = len(df)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokens = tokenizer(df['Phrase'].tolist(),max_length=seq_len,truncation=True,padding='max_length',add_special_tokens=True,return_tensors='np')