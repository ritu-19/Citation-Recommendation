import torch
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/test_eval.csv', encoding = "latin-1")
df = df.dropna()
df = df.reset_index(drop=True)

query_id = 1       #change this to find similar documents to this input

cosine_sim_df = {'score':[], 'paper_index':[], 'paper_original_id':[], 'gt':[]}

for idx in range(len(df)):
  corpus = [df.loc[query_id].paperabstract1, df.loc[idx].paperabstract2]
  vectorizer = TfidfVectorizer(stop_words='english')
  tfidf = vectorizer.fit_transform(corpus).todense()
  cosine_sim = cosine_similarity(tfidf[0], tfidf[1])
  cosine_sim_df['score'].append(cosine_sim.item())
  cosine_sim_df['paper_index'].append(idx)
  cosine_sim_df['gt'].append(df.loc[idx].label)
  cosine_sim_df['paper_original_id'].append(df.loc[idx].id2)

cosine_sim_df = pd.DataFrame(data=cosine_sim_df)
sorted_df = cosine_sim_df.sort_values(by='score', ascending=False).reset_index(drop=True)
sorted_df_2 = sorted_df[sorted_df.score <= 0.99].reset_index(drop=True)

top_k = 99      #change this to display the top-k results
print('Original:-')
print(df.loc[query_id].paperabstract1)
print('Top {} Matches:-'.format(top_k))
gts = []
preds = []
count = 1
for idx in range(top_k):
  count += 1
  paper_id = sorted_df_2.loc[idx].paper_index
  gts.append(df.loc[paper_id].label)
  if count <=50: preds.append(1)
  else: preds.append(0)
  print('GT: {}, Pred: {}, : {}'.format(df.loc[paper_id].label, preds[idx], df.loc[paper_id].paperabstract2))

print(classification_report(gts, preds))
