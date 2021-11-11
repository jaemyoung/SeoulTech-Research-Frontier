#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle as pickle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


filepath = 'data/similarity_matrix.pickle'
with open(filepath, 'rb') as lf :
        similarity_vector = pickle.load(lf)

with open('data/cluster_topic.pickle',"rb") as lf:
          cluster_topic = pickle.load(lf)
          
cluster_topic = pd.read_csv('data/cluster_topic.csv', index_col=(0))


e = [row.sort_values(ascending = False).index[0] for idx,row in cluster_topic.iterrows()]
e = pd.DataFrame(e)
e.to_csv('data/test.csv')

cluster_topic.loc[0].sort_values(ascending=False).index[0]

# clustering simliarity 구하기
num_topic = 40
cluster_n = 20
simliarity_vetor=[]
for i in range(len(corpus)):
    r=[]
    for w in ldamodel.get_document_topics(corpus[i], minimum_probability=0):
        r.append(w[1])
    simliarity_vetor.append(r)
E= pd.DataFrame(simliarity_vetor)
E.to_csv(file_path +now+ 'Topic='+str(num_topic)+'_simliarity.csv', header= ["topic"+str(i) for i in range(1, num_topic+1)])
print("make topic simliarity complete!")

simliarity_vetor.insert(cluster,0)
kmeans = KMeans(n_clusters= 40).fit(simliarity_vetor)
clusters = kmeans.labels_


# 빈도수 그래프
lis = pd.DataFrame(lis)
a= lis.value_counts()

ax = a.plot(kind='bar', title='Number of cluster', figsize=(12, 4), legend=None)
ax.set_xlabel('Cluster', fontsize=12)          # x축 정보 표시
ax.set_ylabel('Number of documents', fontsize=12)     # y축 정보 표시




cluster = df
#토픽별 가중치 시각화
possibility_vector= pd.read_csv("C:/Users/82109/GitHub/doc2vec/data/possibility_vector.csv")
df = pd.read_csv("C:/Users/82109/GitHub/doc2vec/data/clustered_data.csv")
#각 문서별 상위 토픽 가중치 index 저장
lis = []
for idx,row in possibility_vector.iterrows():
    lis.append(row.sort_values(ascending = False).head(5).index[0])
   
 

#각 군집별 토픽 가중치 저장
cluster_23 = df[df['cluster'] == 23]
lis =[]
for idx, row in cluster_23.iterrows():
    #dic[idx]
    lis.append(dic[idx])
    
# 군집 빈도수 별 이름변경
cluster = pd.read_csv("C:/Users/82109/GitHub/doc2vec/data/clustered_data.csv")
cluster_col =cluster["cluster"]
e = cluster_col.value_counts()
e.to_csv('C:/Users/82109/GitHub/doc2vec/data/test.csv')
