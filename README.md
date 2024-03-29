# Django React Contents filtering Recommendation

Django와 React, Pandas, Word2vec, VGG16을 이용한 컨텐츠 필터링 기반 추천 시스템 입니다.  

## 개요 및 소개

**개요**

- 리뷰 기반 토픽 모델링 + 이미지 기반 추천 시스템
- 상품 정보 + 리뷰데이터(토픽 모델링) + 상품 이미지를 활용한 추천시스템 개발
- 총 상품 8745개

- **프로젝트 소개**
    
    - 파이썬 기반의 웹 프레임워크인 ‘Django’를 활용하여 의류 쇼핑 플랫폼 '무신사'에서 직접 크롤링한 ‘패션 데이터’를 이용한 컨텐츠 기반 의류 추천 서비스를 만들었습니다.
    데이터 수집에는 fastapi를 이용하여 수집 API를 만들어 활용했습니다.
    수집할 데이터가 많아 비동기적으로 의류 데이터를 scraping하여 CSV로 변환하여 저장하였습니다.
    모델링에 사용되는 데이터가 30만 건 이상이므로 의류 카테고리 별로 수집 API를 병렬적으로 실행시켜 적재했습니다.
    이후 병합한 CSV와 pandas를 활용하여 추천 시스템 모델링을 진행했습니다.
    - EDA를 거쳐 서비스에 필요한 데이터만을 추출하여 SQLite로 저장하고, pandas를 이용하여 Django 내부에서 학습시킨 모델을 활용하여 프로젝트를 진행했습니다. 
    상품 메타 정보와 리뷰 데이터, 상품의 이미지 유사도 정보를 이용하여 컨텐츠 유사도 기반 추천 시스템을 만들었습니다.
    - 해당 추천 시스템을 만들어낼 때, 상품 정보와 리뷰 데이터를 이용하여 Word2vec을 학습하였고, 원하는 토픽 키워드와 아이템 간의 유사도를 계산하여 컨텐츠 유사도 기반 추천 시스템을 만들어냈습니다. 하지만 리뷰 데이터를 비롯한 상품의 텍스트 데이터 만으로는 추천 성능에 한계가 발생하였습니다.
    이때, 콘텐츠 기반 추천 성능을 높이기 위하여 VGG16을 통해 이미지의 Feature을 추출한 후, 상품의 이미지 유사도 정보를 이용하여 성능을 보완하는 CBIR(Content-based image retrieval)을 사용하였습니다. 이를 통해 텍스트로는 표현하기 어려운 상품의 형태, 색상 등의 특징을 사전훈련된 딥러닝 모델을 통해 추출하여 저희 추천 시스템에 추가하여 성능을 보완하였습니다. 이를 통해, 이전보다 키워드와 유사한 상품을 찾을 수 있게 되었습니다. <br>
    또한, 추천된 서비스를 다시 SQLite에 적재하여 향후 사용자 추천 경향을 분석할 수 있도록 구성했습니다.


### 토픽 추천 시스템 로직

- 아이템을 추천해주는 함수 구성

- 유사도 = Cosine Similarity

- 모든 벡터는 word2vec 모델을 통해 산출된 수치 벡터

- **아이템 유사도 벡터 * 아이템 유사도 벡터**
    
    cosine_similarity(전체 아이템 열 벡터, 전체 아이템 열 벡터)
    
    ⇒ item마다 모든 item들에 관한 유사도 벡터 생성
    
    (8745 * 8745)
    

- **키워드(Input string) 벡터 * 아이템 유사도 벡터**
    
    - 단일 아이템 벡터 : (8745, )

    1. 리뷰를 토대로 토픽 모델링을 거쳐 나온 토픽들을 키워드로 입력
    
    2. 키워드를 인풋으로 넣었을 때 해당 키워드와 아이템들 간 유사도를 계산한 후  키워드와 유사성이 높은 순서대로 정렬
    
    3. 아이템간 유사도를 이용하여 키워드와 유사도가 제일 높은 아이템과 가장 유사한 아이템들을 가져온다.

  
## Word2vec + CNN(VGG16) 이미지 추천

- 토픽 모델링과 동일한 word2vec 모델 사용
    1. 키워드를 인풋으로 넣었을 때 해당 키워드와 아이템들 간 유사도를 계산한 후 키워드와 유사성이 높은 순서대로 정렬   
    2. 키워드와 유사성이 높은 상품 상위 3가지를 가져옵니다.
    3. 아이템마다 vgg16으로 feature를 추출한 정보를 바탕으로 이미지 벡터 유사도가 높은 상품 2000개씩 가져와 교집합에 속하는 상품들을 추출합니다
    4. 해당 상품들에 대한 정보를 가져와 키워드와의 코사인 유사도를 기준으로 내림차순 정렬합니다. 
    5. 보고 싶은 아이템의 개수만큼 슬라이싱합니다. (Top-n)
    6. 슬라이싱한 데이터를 보완한 가중 평점으로 내림차순 정렬하여 사용자에게  보여줍니다.

---------

### Stack
  - Django DRF
  - Bootstrap(Template)
  - React
  - word2vec => (npy 파일 이용)
  - VGG16
  - pandas
  - sqlite

### 구현 및 보고서 

[리뷰 토픽 모델링 기반 추천 시스템 Report](https://hajunyoo.oopy.io/6ec88524-b8a9-488f-84ae-d63abfe67295)
