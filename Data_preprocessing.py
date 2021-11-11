import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import pickle
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk import pos_tag
import nltk
nltk.download('wordnet')

# pos 태깅 분류
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    
    elif tag.startswith('V'):
        return wordnet.VERB
    
    elif tag.startswith('N'):
        return wordnet.NOUN
    
    elif tag.startswith('R'):
        return wordnet.ADV
    
    else:
        return None

#import re
#def apply_re(tokenized_text):
    result= []
    for tok_list in tokenized_text:
        list_par = []
        for tok in tok_list:
            text = re.sub('[^a-zA-Z]',' ',tok).strip() # 영어 제외 다 제거.
            if(text != ''): # 빈리스트 제거.
                list_par.append(text)
        result.append(list_par)
    return result
    
# 불용어 사전 적용
from nltk.corpus import stopwords

def apply_stop_words(tokenized_text):
    stop_words = stopwords.words('english') 
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            if tok not in stop_words:
                tok_result.append(tok)
        result.append(tok_result)
    return result  


# scientific 불용어사전 적용
file_path =r"C:/Users/82109/GitHub/doc2vec/stopwords/universal_scientific_stopwords.csv"
def apply_scientificword(tokenized_text):
    scientifiword = pd.read_csv(file_path,header = None) #csv 파일일때만 사용
    stop_words = scientifiword[0].tolist()
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            if tok not in stop_words:
                tok_result.append(tok)
        result.append(tok_result)
    return result  

# 표제어 추출 적용 (pos 태깅 후 사용)
import nltk
nltk.download('wordnet')
 
def apply_lemma(tokenized_text):
    lemma = WordNetLemmatizer()
    result = []
    for tok_list in tokenized_text:
        tok_result =[lemma.lemmatize(tok[0], pos= get_wordnet_pos(tok[1])) for tok in tok_list]
        result.append(tok_result)
    return result 

# 어간 추출(porterstemmer)

def apply_porter(tokenized_text):
    s = PorterStemmer()
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            tok_result.append(s.stem(tok))
        result.append(tok_result)
    return result 

# 어간 추출(lancasterstemmer)

def apply_lancaster(tokenized_text):
    s = LancasterStemmer()
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            tok_result.append(s.stem(tok))
        result.append(tok_result)
    return result 

# data 합치기
data_2016 = pd.read_csv(r"C:/Users/82109/GitHub/doc2vec/data/2016.csv")
data_20172018 =pd.read_csv(r"C:/Users/82109/GitHub/doc2vec/data/2017-2018.csv")
data_20192020 = pd.read_csv(r"C:/Users/82109/GitHub/doc2vec/data/2019-2020.csv")
data_2021 = pd.read_csv(r"C:/Users/82109/GitHub/doc2vec/data/2021.csv")
row_data = pd.concat([data_2016,data_20172018,data_20192020,data_2021])
#row_data 저장
row_data.to_csv("C:/Users/82109/GitHub/doc2vec/data/row_data(2016-2021).csv")

paper_notnull = row_data.fillna("")
data = [row['Title'] + row['Abstract']+ row['Author Keywords'] +row['Index Keywords'] for idx, row in paper_notnull.iterrows()]

#토큰화 및 특수문자 및 공백숫자 제거
tokenized_text = []
for doc in data:
    tokenized_text.append(nltk.regexp_tokenize(doc.lower(), '[A-Za-z]+'))   

# 순서 -> stopword -> scientificword -> pos-tagging -> tagging-J,V,N,R만 살리기 ->lemma
result_stopword = apply_stop_words(tokenized_text)
result_scientific = apply_scientificword(result_stopword)
result_tagging = [pos_tag(tok) for tok in result_scientific]
result_tagging_alpha =[]
for l in result_tagging:
    s= [tagged_word for tagged_word in l if tagged_word[1].startswith(('J', 'V', 'N', 'R'))]
    result_tagging_alpha.append(s)
result_lemma = apply_lemma(result_tagging_alpha)


# pickle 파일로 저장
with open('data/preprocessing_data(4046)_lemma.pickle', 'wb') as f:
    pickle.dump(result_lemma, f, pickle.HIGHEST_PROTOCOL)