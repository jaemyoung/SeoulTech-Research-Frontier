# SEOULTECH 교직원 연구 실적 텍스트 데이터 분석<br></br>

### 필요 라이브러리
- nltk
- gensim
- scikit-learn
- matplotlib<br></br>

### 사용법
1. 사용 전 doc2vec의 하이퍼파라미터를 지정해준다.
2. 터미널에서 `python main.py` 키워드를 통해 해당 파일을 실행해준다.
3. 파서를 변경해가며 결과를 확인한다. <br></br>

### 파서
- --data : string, 분석하고자하는 텍스트 데이터의 path 지정 (예. data/data.csv)
- --parameter : string, doc2vec의 하이퍼파라미터의 path 지정 (예. doc2vec_paramter.txt)
- --mode : string, "both", "train", "embedding" 3종류가 있으며, default는 "both"
  - both : doc2vec 모델의 학습과 텍스트 임베딩을 동시에 진행, --save_model_name 파서 필요
  - train : doc2vec 모델의 학습 및 모델 저장만을 진행, --save_model_name 파서 필요
  - embedding : 저장된 doc2vec 모델을 불러와 텍스트를 임베딩을 진행, --model 파서 필요 
- --model : string, 저장된 doc2vec 모델의 path를 불러옴 
- --save_model_name : string : 학습 후 모델 저장 시, 저장할 모델의 이름을 지정




