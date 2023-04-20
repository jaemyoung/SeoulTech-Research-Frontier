# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:16:51 2021

@author: 82109
"""

import numpy as np
import pandas as pd
import collections
import seaborn as sns
import pickle
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
#pip install wordcloud

#tokenized wordlist 불러오기
with open('data/preprocessing_data(4046).pickle', "rb") as fr:
    tokenized_doc = pickle.load(fr)
#re_cluster id 불러오기
txt =np.loadtxt("C:/Users/82109/GitHub/doc2vec/Re_cluster.txt", delimiter=",")
cluster_id = txt.astype(np.int32)

def apply_stop_words(tokenized_text):
    stop_words = ['society','ieee','korean','use']
    result = [tok for tok in tokenized_text if tok not in stop_words]
    return result
# Wordcloud 실행
cluster_wordlist = pd.DataFrame(columns=['cluster_id', 'word'])
cluster_wordlist['word'] =tokenized_doc
cluster_wordlist['cluster_id'] = cluster_id
for i in range(10,37):
    test1 = cluster_wordlist[cluster_wordlist['cluster_id']==i]['word']
    # Generate a wordcloud
    word_counts = collections.Counter(apply_stop_words(test1.sum()))
    wordcloud = WordCloud(font_path=r'C:/Users/82109/GitHub/doc2vec/글꼴/Helvetica 400.ttf', 
                          background_color='white', 
                          colormap='tab20c', 
                          min_font_size=20,
                          max_font_size=400,
                          max_words = 200,
                          width=1600, height=800,random_state=(1004)).generate_from_frequencies(word_counts)
    plt.figure(figsize=(20, 20))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('temp2.png')
    plt.show()

# 주요연구 상위 20개씩 저장
research = pd.read_csv('C:/Users/82109/GitHub/doc2vec/Cluster Naming_Papers_최종.csv', index_col=0, encoding='CP949')
save_result = pd.DataFrame(columns = research.columns)
a =a.dropna(subset=['writer_korean_name'])

for i in range(1,37):
   save_result=save_result.append(a[a['final_cluster'] ==i].sort_values(by='Cited by',ascending = False).iloc[:20],ignore_index=False)
    
save_result.to_csv('C:/Users/82109/GitHub/doc2vec/cited_by_clsuter.csv')
