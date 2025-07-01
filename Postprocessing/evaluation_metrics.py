import pandas as pd
import numpy as np
import os
from bert_score import BERTScorer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from evaluate import load
import asyncio

stop = stopwords.words('italian')
scorer = BERTScorer(model_type='distilbert-base-multilingual-cased')
meteor = load('meteor')
bleu = load('bleu')
rouge = load('rouge')
print()

def cleaning(text):
    text = re.sub(r'[^\w\s]+', ' ', text.lower()).strip()
    text = ' '.join([word for word in text.split() if word not in stop])
    return text

DIRECTORY = "Datasets"
files = os.listdir(DIRECTORY)
df_oracle = pd.read_csv('Oracle.csv')
df_questions = pd.read_csv('Questions.csv')
csv_files = ['DatasetCustom.csv', 'DatasetFixedSize.csv', 'DatasetNewLine.csv']

oracle = pd.read_csv("Cleaned_Oracle.csv")

print("##################### - BERTSCore on Answer - Oracle")
for file in csv_files:
    path = DIRECTORY + '/' + file
    df = pd.read_csv(path)

    strings_to_search = [
        "",
    ]

    for string in strings_to_search:
        i = df[((df.question == string))].index
        df.drop(i, inplace=True)

    candidate = df['answer'].to_list()
    references = []
    for ref in oracle['oracle'].to_list():
        references.append([ref])

    print(file)
    P, R, F1 = scorer.score(candidate, references)
    print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    print()

print("##################### - METEOR score on Answer - Oracle")
for file in csv_files:
    path = DIRECTORY + '/' + file
    df = pd.read_csv(path)

    strings_to_search = [
        "",
    ]

    for string in strings_to_search:
        i = df[((df.question == string))].index
        df.drop(i, inplace=True)

    candidate = df['answer'].to_list()
    references = []
    for ref in oracle['oracle'].to_list():
        references.append([ref])

    print(file)
    results = meteor.compute(predictions=candidate, references=references)
    print(results)
    print()

print("##################### - BLEU score on Answer - Oracle")



for file in csv_files:
    path = DIRECTORY + '/' + file
    df = pd.read_csv(path)

    strings_to_search = [
        "",
    ]

    for string in strings_to_search:
        i = df[((df.question == string))].index
        df.drop(i, inplace=True)
    
    candidate = df['answer'].to_list()
    references = []
    for ref in oracle['oracle'].to_list():
        references.append([ref])

    print(file)
    results = bleu.compute(predictions=candidate, references=references)
    print(results)
    print()

print("##################### - ROUGE score on Answer - Oracle")
for file in csv_files:
    path = DIRECTORY + '/' + file
    df = pd.read_csv(path)

    strings_to_search = [
        "",
    ]

    for string in strings_to_search:
        i = df[((df.question == string))].index
        df.drop(i, inplace=True)

    candidate = df['answer'].to_list()
    references = []
    for ref in oracle['oracle'].to_list():
        references.append([ref])

    print(file)
    results = rouge.compute(predictions=candidate, references=references)
    print(results)
    print()
