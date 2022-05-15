#!/usr/bin/env python
# coding: utf-8

# # Intro
# 텍스트 분류란 자연어 처리 기술을 활용해 글의 정보를 추출해서 문제에 맞게 사람이 정한 범주(Class)로 분류하는 문제다. 텍스트 분류의 방법과 예시 등 자세한 내용은 이미 2장에서 알아봤다. 이번 장에서는 실제로 데이터셋을 가지고 텍스트 분류를 실습해보면서 영어 텍스트 분류와 한글 텍스트 분류에 대해 알아보자
# 
# 2장에서 본 것처럼 텍스트 분류에는 여러 활용사례가 있다. 그 중에서 이번 장에서는 감정 분류 문제를 다루겠다. 

# In[1]:


# 언제어디서든
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


import datetime


# [keras Setting](https://kloong.tistory.com/entry/Keras%EC%97%90%EC%84%9C-GPU-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-MacBook-18-pro-Radeon-pro-560X-4G)

# In[3]:


import os 
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras


# # 01 영어 텍스트 분류
# |데이터이름|Bag of Words Meets Bags of PopCorn|
# |:--:|:--:|
# |데이터 용도|텍스트 분류 학습을 목적으로 사용한다.|
# |데이터 권한|MIT권한을 가지고 있으나 캐글에 가입한 후 사용하길 바란다.|
# |데이터 출처|https://www.kaggle.com/c/word2vec-nlp-tutorial/data
# 
# ### [대회 관련정보](https://www.kaggle.com/c/word2vec-nlp-tutorial/)
# 
# ### 문제 소개
# - 이번 절에서는 영어 텍스트 분류 문제 중 캐글의 대회인 워드 팝콘 문제를 활용할 것이다.이 문제를 해결하면서 텍스트 분류 기술을 알아 보겠다. 먼저 워드 팝콘이 어떤 문제인지를 알아보자
# 
# ##### 워드 팝콘
# - 워드 팝콘은 인터넷 영화 데이터베이스(IMDB)에서 나온 영화 평점 데이터를 활용한 캐글 문제다. 영화 평점 데이터이므로 각 데이터는 영화 리뷰 텍스트와 평점에 따른 감정 값(긍정 혹은 부정)으로 구성돼 있다. 이 데이터는 보통 감정 분류(sentiment analysis) 문제에서 자주 활용된다. 그럼 이 데이터를 어떻게 분류할지에 대한 목표를 알아보자.
# 
# ##### 목표
# - 크게 3가지 과정을 거칠 것이다. 첫 번째는 데이터를 불러오는 것과 정제되지 않은 데이터를 활용하기 쉽게 전처리하는 과정이다. 그 다음은 데이터를 분석하는 과정이다. 

# ### 1. 데이터 불러오기

# In[4]:


### 1. 데이터 분석 및 전처리 

##### 데이터 불러오기 및 분석
get_ipython().system('kaggle competitions download -c word2vec-nlp-tutorial')


# In[5]:


import zipfile


# In[6]:


DATA_IN_PATH = './data_in/'

file_list = ['labeledtrainData.tsv.zip', 'unlabeledTrainData.tsv.zip', 
            'testData.tsv.zip']

for file in file_list:
    zipRef = zipfile.ZipFile(DATA_IN_PATH + file, 'r')
    zipRef.extractall(DATA_IN_PATH)
    zipRef.close()


# In[7]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
# 그래프를 주피터 노트북에서 바로 그리게함


# In[8]:


train_data = pd.read_csv(DATA_IN_PATH+"labeledTrainData.tsv", header=0, delimiter='\t', quoting=3)
train_data.head()


# 데이터는 "id", 'sentiment', 'review'로 구분돼 있으며, 각 리뷰('review')에 대한 감정('sentiment')이 긍정('1') 혹은 부정('0')이 나와 있다. 
# 
# 1. 데이터 크기
# 2. 데이터의 개수
# 3. 각 리뷰의 문자 길이 분포
# 4. 많이 사용된 단어
# 5. 긍정, 부정 데이터의 분포
# 6. 각 리뷰의 단어 개수 분포
# 7. 특수문자 및 대문자, 소문자 비율

# In[9]:


print('파일 크기: ')
for file in os.listdir(DATA_IN_PATH):
    if 'tsv' in file and 'zip' not in file:
        print(file.ljust(30)+str(round(os.path.getsize(DATA_IN_PATH+file)/1000000, 2))+ "MB")


# In[10]:


print('전체 학습 데이터의 개수: {}'.format(len(train_data)))


# In[11]:


train_length = train_data['review'].apply(len)
train_length.head()


# In[12]:


# 그래프에 대한 이미지 크기 선언
# figsize: (가로, 세로) 형태의 튜플로 입력
plt.figure(figsize=(12,5))
# 히스토그램 선언
# bins: 히스토그램 값에 대한 버킷 범위
# range: x축 값의 범위
# alpha: 그래프 색상 투명도
# color: 그래프 색상
# label: 그래프에 대학 라벨
plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
plt.yscale('log', nonposy='clip')
# 그래프 제목
plt.title('Log-Histogram of length of review')
# 그래프 x축 라벨
plt.xlabel('Length of review')
# 그래프 y축 라벨
plt.ylabel('Number of review')


# In[13]:


print('리뷰 길이 최댓값: {}'.format(np.max(train_length)))
print('리뷰 길이 최솟값: {}'.format(np.min(train_length)))
print('리뷰 길이 평균값: {:.2f}'.format(np.mean(train_length)))
print('리뷰 길이 표준편차: {:.2f}'.format(np.std(train_length)))
print('리뷰 길이 중간값: {}'.format(np.median(train_length)))
# 사분위의 대한 경우는 0~100 스케일로 돼 있음
print('리뷰 길이 제1사분위: {}'.format(np.percentile(train_length, 25)))
print('리뷰 길이 제3사분위: {}'.format(np.percentile(train_length, 75)))


# In[14]:


plt.figure(figsize=(12,5))
# 박스 플롯 생성
# 첫 번째 인자: 여러 분포에 대한 데이터 리스트를 입력
# labels: 입력한 데이터에 대한 라벨
# showmeans: 평균값을 마크함

plt.boxplot(train_length, labels=['counts'], showmeans=True)
plt.show()


# In[15]:


from wordcloud import WordCloud
cloud = WordCloud(width = 800, height=600).generate(" ".join(train_data['review']))
plt.figure(figsize=(20,15))
plt.imshow(cloud)
plt.axis('off')


# In[16]:


fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6,3)
sns.countplot(train_data['sentiment'])


# In[17]:


print('긍정 리뷰 개수: {}'.format(train_data['sentiment'].value_counts()[1]))
print('부정 리뷰 개수: {}'.format(train_data['sentiment'].value_counts()[0]))


# In[18]:


train_word_counts = train_data['review'].apply(lambda x:len(x.split(' ')))


# In[19]:


plt.figure(figsize=(15,10))
plt.hist(train_word_counts, bins=50, facecolor='r', label='train')
plt.title('Log-Histogram of word count in review', fontsize=15)
plt.yscale('log', nonposy='clip')
plt.legend()
plt.xlabel('Nubmer of words', fontsize=15)
plt.ylabel('Nubmer of reviews', fontsize=15)


# In[20]:


print('리뷰 단어 개수 최솟값: {}'.format(np.max(train_word_counts)))
print('리뷰 단어 개수 최숫값: {}'.format(np.min(train_word_counts)))
print('리뷰 단어 개수 평균값: {:.2f}'.format(np.mean(train_word_counts)))
print('리뷰 단어 개수 표준편차: {:.2f}'.format(np.std(train_word_counts)))
print('리뷰 단어 개수 중간값: {}'.format(np.median(train_word_counts)))
# 사분위의 대한 경우 0-100 스케일로 돼 있음
print('리뷰 단어 개수 제1사분위: {}'.format(np.percentile(train_word_counts, 25)))
print('리뷰 단어 개수 제3사분위: {}'.format(np.percentile(train_word_counts, 75)))


# In[21]:


qmarks = np.mean(train_data['review'].apply(lambda x: "?" in x)) # 물음표가 구두점으로 쓰임
fullstop = np.mean(train_data['review'].apply(lambda x: '.' in x)) # 마침표
capital_first = np.mean(train_data['review'].apply(lambda x:x[0].isupper())) # 첫 번째 대문자
capitals = np.mean(train_data['review'].apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(train_data['review'].apply(lambda x : max([y.isdigit() for y in x]))) # 숫자 개수

print('물음표가 있는 질문: {:.2f}%'.format(qmarks *100))
print('마침표가 있는 질문: {:.2f}%'.format(fullstop *100))
print('첫 글자가 대문자인 질문: {:.2f}%'.format(capital_first *100))
print('대문자가 있는 질문: {:.2f}%'.format(capitals *100))
print('숫자가 있는 질문: {:.2f}%'.format(numbers *100))


# ### 2. 데이터 전처리

# In[22]:


import re
import pandas
import numpy
import json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer


# 먼저 사용할 라이브러리 살펴보자. 우선 데이터를 다루기 위해 판다스를 사용하고, 데이터를 정제하기 위한 re와 BeautifulSoup을 사용한다. 그리고 불용어를 제거하기 위해 NLTK 라이브러리의 stopwords모듈을 사용한다.
# 
# 그리고 불용어를 제거하기 위해 NLTK 라이브러리의 stopwords 모듈을 사용한다.\
# 텐서플로의 전처리 모듈인 pad_sequences와 Tokenizer를 사용하고, 마지막으로 전처리된 데이터를 저장하기 위해 넘파이를 사용한다.

# In[23]:


DATA_IN_PATH = './data_in/'
train_data = pd.read_csv(DATA_IN_PATH + 'labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
print(train_data['review'][0]) # 첫 번쨰 리뷰 데이터


# 리뷰 데이터를 보면 문장 사이에 '<br/>'과 같은 HTML 태그와 '\', 같은 특수문자가 포함된 것을 확인할 수 있 것을 확인할 수 있다. 문장부호 및 특수문자는 일반적으로 문장의 의미에 크게 영향을 미치지 않기 때문에 최적화된 학습을 위해 제거하자.

# In[24]:


review = train_data['review'][0] # 리뷰 중 하나를 가져온다.
review_text = BeautifulSoup(review, 'html5lib').get_text() # HTML 태그 제거
review_text = re.sub("[^a-zA-Z]", " ", review_text) # 영어 문자를 제외한 나머지 모두 공백으로 바꾼다.


# BeautifulSoup 라이브러리의 get_text 함수를 사용하면 HTML 태그를 제외한 나머지 텍스트만 얻을 수 있다. 그리고 다음으로 re 라이브러리의 sub 함수를 사용해 영어 알파벳을 제외한 모든 문자, 즉 숫자 및 특수기호를 공백으로 대체한다. 

# In[25]:


print(review_text)


# 결과를 보면 HTML 태그와 특수문자가 모두 제거된 것을 확인할 수 있다.\
# 다음으로 진행할 과정은 불용어(stopword)를 삭제하는 것이다. 불용어란 문장에서 자주 출현하나 전체적인 의미에 큰영향을 주지 않는 단어를 말한다.

# In[26]:


stop_words = set(stopwords.words('english')) # 여어 불용어 set을 만든다.

review_text = review_text.lower()
words = review_text.split() # 소문자로 변환한 후 단어마다 나눠서 단어 리스트로 만든다.
words = [w for w in words if not w in stop_words] # 불용어를 제거한 리스트를 만든다.


# 진행 과정을 보면 우선 리뷰를 lower 함수를 사용해 모두 소문자로 바꿨다.\
# 이후 split 함수를 사용해 띄어쓰기를 기준으로 텍스트 리뷰를 단어 리스트로 바꾼 후 불용어에 해당하지 않는 단어만 다시 모아서 리스트로 만들었다. 중간에 속도 향상을 위해 set 데이터 타입으로 정의한 후 사용했다.

# In[27]:


print(words)


# 하나의 문자열이었던 리뷰가 단어 리스트로 바뀐 것을 확인할 수 있다.\
# 이를 모델에 적용하기 위해서는 다시 하나의 문자열로 합쳐야 한다. 파이썬의 내장 함수인 join 함수를 사용하면 간단히 단어들을 하나로 붙여서 문자열로 만들 수 있다. 

# In[28]:


clean_review = ' '.join(words) # 단어 리스트를 다시 하나의 글로 합친다.
print(clean_review)


# In[29]:


def preprocessing(review, remove_stopwords = False):
    # 불용어 제거는 옵션으로 선택 가능하다.
    
    # 1. HTML 태그 제거
    review_text = BeautifulSoup(review, 'html5lib').get_text()
    
    # 2. 여어가 아닌 특수문자를 곱액(" ")으로 바꾸기re
    view_text = re.sub("]^a-zA-Z]", " ", review_text)
    
    # 3. 대문자를 소문자로 바꾸고 공백 단위로 텍스트를 나눠서 리스트로 만든다.
    words = review_text.lower().split()
    
    if remove_stopwords:
        # 4. 불용어 제거
        
        # 영어 불용어 불러오기
        stops = set(stopwords.words('english'))
        
        # 불용어가 아닌 단어로 이뤄진 새로운 리스트 생성
        words = [w for w in words if not w in stops]
        
        # 5. 단어 리스트를 공백을 넣어서 하나의 글로 합친다.
        clean_review = ' '.join(words)
        
    else: # 불용어를 제거하지 않을 때
        clean_review = ' '.join(words)
        
    return clean_review


# 함수의 경우 불용어 제거는 인자값으로 받아서 선책할 수 있게 햇다. 이제 정의한 함수를 사용해 전체 데이터에 대해 전처리를 진행한 후 전처리를 진행한 후 전처리된 데이터를 하나 확인해 보자

# In[30]:


clean_train_reviews = []
for review in train_data['review']:
    clean_train_reviews.append(preprocessing(review, remove_stopwords=True))
    
# 전처리한 데이터의 첫 번째 데이터를 출력
clean_train_reviews[0]


# 이제 두가지 전처리 과정이 남았다.\
# 우선 전처리한 데이터에서 각 단어를 인덱스로 벡터화해야 한다. 그리고 모델에 따라 입력값의 길이가 동일해야 하기 때문에 일정 길이로 자르고 부족한 부분은 특정값으로 채우는 패딩 과정을 진행해야 한다. 하지만 모델에 따라 각 리뷰가 단어들의 인덱스로 구성된 벡터가 아닌 텍스트로 구성돼야 하는 경우도 있다. 따라서 지금까지 전처리한 데이터를 판다스의 데이터프레임으로 만들어 두고 이후에 전처리 과정이 모두 끝난 후 전처리한 데이터를 저장할 때 함께 저장하게 된다.
# 

# In[31]:


clean_train_df = pd.DataFrame({'review': clean_train_reviews, 'sentiment': train_data['sentiment']})
clean_train_df


# In[32]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_reviews)
text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)


# In[33]:


print(text_sequences[0])


# In[34]:


word_vocab = tokenizer.word_index
word_vocab["<PAD>"] = 0
print(word_vocab)


# 단어 사전의 경우 앞서 정의한 tokenizer 객체에서 word_index 값을 뽑아보면 사전 형태로 구성돼 있다. 결과를 보면 'movie'라는 단어는 1이고 'film'이라는 단어는 2로 돼있는것을 확인할 수 있다.
# 
# word_index에는 패딩 정보값이 정의돼 있지 않기 때문에 '<PAD>'에 대한 인덱스 값을 0으로 입력한다. 그렇다면 전체 데이터에서 사용된 단어 개수는 총 몇개인지 확인해 보자

# In[35]:


print('전체 단어 개수: ', len(word_vocab))


# 단어는 총 74,000개 정도다. 단어 사전뿐 아니라 전체 단어 개수도 이후 모델에서 사용되기 떄문에 저장해 둔다.  데이터에 저장해둔다. 데이터에 대한 정보인 단어 사전과 전체 단어 개수는 새롭게 딕셔너리 값을 지정해서 저장해두자.

# In[36]:


data_configs = {}

data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab)+1


# In[37]:


MAX_SEQUENCE_LENGTH = 174 # #문장 최대 길이

train_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

print('Shape of train data: ', train_inputs.shape)


# 패딩 처리에는 앞서 불러온 pad_sequences 함수를 사용한다.
# 이 함수를 사용할 때는 인자로 패딩을 적용할 데이터, 최대 길이값, 0 값을 데이터 앞에 넣을 뒤에 넣을지 설정을 한다. 여기서 최대 길이를 174로 설정했는데, 이는 앞서 데이터 분석 과정에서 단어 개수의 통계를 계산했을 때 나왔던 중간 값이다. 보통 평균이 아닌 중간 값을 사용하는 경우가 많은데, 일부 이상치 데이터가 길이가 지나치게 길면 평균이 급격하게 올라갈 수 있기 때문에 적당한 값인 중간 값을 사용하는 것이다. 이렇게 패딩 처리를 통해 데이터의 형태가 25,000개의 데이터가 174라는 길이를 동일하게 가지게 되었음을 확인할 수 있다.

# In[38]:


train_labels = np.array(train_data['sentiment'])
print('Shape of label tensor: ', train_labels.shape)


# 이제 전처리한 데이터를 이후 모델링 과정에서 사용하기 위해 저장하자. 여기서는 다음과 같이 총 4개의 데이터를 저장할 것이다.
# - 정제된 텍스트 데이터
# - 벡터화한 데이터
# - 정답 라벨
# - 데이터 정보(단어 사전, 전체 단어 개수)

# 텍스트 데이터의 경우 CSV 파일로 저장하고, 벡터화한 데이터와 정답 라벨의 경우 넘파이 파일로 저장한다. 마지막 데이터 정보의 경우 딕셔너리 형태이기 때문에 JSON 파일로 저장한다. 우선 경로와 파일명을 설정하고 os 라이브러리를 통해 폴더가 없는 경우 폴더를 생성하자.

# In[39]:


DATA_IN_PATH = './data_in/'
TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TRAIN_CLEAN_DATA = 'train_clean.csv'
DATA_CONFIGS = 'data_configs.json'

import os
# 저장하는 디렉토리가 존재하지 않으면 생성
if not os.path.exists(DATA_IN_PATH):
    os.makedirs(DATA_IN_PATH)    


# In[40]:


# 전처리된 데이터를 넘파이 형태로 저장
np.save(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'wb'), train_inputs)
np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'wb'), train_labels)

# 정제된 텍스트를 CSV 형태로 저장
clean_train_df.to_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA, index=False)

# 데이터 사전을 JSON 형태로 저장
json.dump(data_configs, open (DATA_IN_PATH + DATA_CONFIGS, 'w'), ensure_ascii=False)


# 
# 각 데이터에 맞게 저장 함수를 사용해서 저장하면 해당 경로에 파이들이 저장된 것을 확인할 수 이싸. 이제 전처리 과정이 모두 끝났다. 지금까지 학습 데이터에 대해서만 전처리를 했으므로 평가 데이터에 대해서도 위와 동일한 과정을 진행하면 된다. 다른 점은 평가 데이터의 경우 라벨 값이 없기 때문에 라벨은 따로 저장하지 않아도 되고 데이터 정보인 단어 사전과 단어 개수에 대한 정보도 학습 데이터의 것을 사용하므로 저장하지 않아도 된다는 것이다. 추가로 평가 데이터에 대해 저장해야 하는 값이 있는데 각 리뷰에 대한 'id'값을 저장해야 한다. 나머지 부분은 학습 데이터와 동일하게 전처리를 진행한다.

# In[41]:


# 불러오기
test_data = pd.read_csv(DATA_IN_PATH + 'testData.tsv', header=0, delimiter='\t', quoting=3)
# 빈 리스트 만들기
clean_test_reviews = []

# 한 문장 한문장 전처리 후 리스트에 담기
for review in test_data['review']:
    clean_test_reviews.append(preprocessing(review, remove_stopwords=True))
# 데이터 프레임 화    
clean_test_df = pd.DataFrame({'review': clean_test_reviews, 'id': test_data['id']})
# 정답지 만들기
test_id = np.array(test_data['id'])

# 벡터화
text_sequences = tokenizer.texts_to_sequences(clean_test_reviews)
# 패딩작업
test_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


# 평가 데이터를 전처리할 때 한 가지 중요한 점은 토크나이저를 통해 인덱스 벡터로 만들 때 토크나이징 객체로 새롭게 만드는 것이 아니라. 기존에 학습 데이터에 적용한 토크나이저 객체를 사용해야 한다는 것이다. 
# 
# 만약 새롭게 만들 경우 학습 데이터와 평가 데이터에 대한 각 단어들의 인덱스가 달라져서 모델에 정상적으로 적용할 수 없기 때문이다. 이제 평가 데이터를 전처리한 데이터도 위와 동일하게 저장하자. 경로는 이미 지정했으므로 파일명만 새롭게 정의한 후 앞선 과정과 동일하게 저장하자. 경로는 이미 지정했으므로 파일명만 새롭게 정의한 후 앞선 과정과 동일하게 저장하자.

# In[42]:


TEST_INPUT_DATA = 'test_input.npy'
TEST_CLEAN_DATA = 'test_clean.csv'
TEST_ID_DATA = 'test_id.npy'


np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)
np.save(open(DATA_IN_PATH + TEST_ID_DATA, 'wb'), test_id)
clean_test_df.to_csv(DATA_IN_PATH + TEST_CLEAN_DATA, index=False)


# ### 3. 모델링 소개
# 지금까지 탐색적 데이터 분석 과정과 데이터 전처리 과정을 진행했다. 이번 장에서는 감정이 긍정인지 부정인지 예측할 수 있는 모델을 만들 것이다.

# ##### 3-1 실습할 모델 소개
# 머신러닝 모델 중 선형 회귀 모델과 랜덤 포레스트 모델로 감정 분석을 하는 방법을 이야기한다. 그리고 딥러닝 모델 중에서는 합성곱 신경망(CNN), 순환 신경망(RNN) 모델에 대해 살펴보겠다.
# 
# ##### 3-2 회귀 모델
# 이번 장에서 만들 모델은 ***로지스틱 회귀 모델***이다. 로지스틱 회귀 모델은 주로 이항 분류를 하기 위해 사용되며 분류 문제에서 사용할 수 있는 가장 간단한 모델이다.  
# 
# ##### 3-3 선형 회귀
# 선형 회귀 모델은 ***종속 변수***와 ***독립 변수***간의 상관 관계를 모델링하는 방법이다. 간단히 말하자면 하나의 선형 방정식으로 표현해 예측할 데이터를 분류하는 모델이라 생각하면 된다. 
# 
# $$ y = w_{1}*x_{1} + w_{2}*x_{2} + ... + b $$
# 위 수식에서  $w_{1} w_{2}$,b는 학습하고자 하는 파라미터이고 $x_{1}과 x_{2}$는 입력값이다. 여기서 모델에 입력하는 값은 바로 $x_{1},x_{2}$ 변수에 넣게 된다. $x_{1},x_{2}$에는 주로 단어 또는 문장 표현 벡터를 입력하게된다.
# 
# ##### 3-4 로지스틱 회귀 모델
# 로지스틱 모델은 선형 모델의 결괏값에 로지스틱 함수를 적용해 0~ 1사이의 값을 갖게 해서 확률로 표현한다. 이렇게 나온 결과를 1에 가까우면 정답이 1이라 예측하고 0에 가까울 경우 0으로 예측한다.
# 
# 이제 로지스틱 모델을 가지고 텍스트 분류를 하자. 여기서는 입력값인 단어를 ***word2vec***을 통한 단어 임베딩 벡터로 맏느는 방법과 ***tf-idf***를 통해 임베딩 벡터로 만드는 두가지 방법 두가지를 모두 사용한다.

# ### 4. TF-IDF를 활용한 모델 구현
# 여기서는 3장에서 배운 단어 표현 방법인 TF-IDF를 활용해 문장 벡터를 만든다. 입력값에 대해 TF-IDF 값으로 벡터화를 진행하기 때문에 사이킷런의 TfidfVectorizer를 사용한다. TfidfVectorizer를 사용하기 위해서는 입력값이 텍스트로 이뤄진 데이터 형태여야 한다. 따라서 전처리한 결과 중 넘파이 배열이 아닌 전제된 텍스트 데이터를 사용한다. 우선 데이터를 불러오자.

# In[43]:


DATA_IN_PATH = './data_in/'
TRAIN_CLEAN_DATA = 'train_clean.csv'

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)

reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])


# 판다스를 이용해 전처리한 텍스트 데이터를 불러온 후 리뷰값과 라벨 값을 각각 따로 리스트로 지정해 둔다. 

# ##### 4-1 TF-IDF 벡터화
# 이제 데이터에 대해 TF-IDF 값으로 벡터화를 진행한다. 진행 방법은 앞서 2장에서 했던 것과 동일하게 진행하면 되는데

# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df= 0.0, analyzer='char', sublinear_tf =True,
ngram_range=(1,3), max_features=5000)

X =vectorizer.fit_transform(reviews)


# 객체를 생성할 때 몇가지 인자값을 설정하는데 하나씩 살펴보자. 
# - min_df: 설정한 값보다 특정 토큰의 df 값이 더 적게 나오면 벡터화 과정에서제거
# - analyzer: 분석하기 위한 기준: 'word', 'char'
# - sublinear_tf: 문서의 단어 빈도수(term-frequency)에 대한 스무딩(smoothing)여부를 설정
# - ngram: 빈도의 기본 단위를 어느 범위의 n-gram으로 설정할것인지 
# - max_features: 각 벡터의 최대 길이, 특징의 길이를 설정

# ##### 4-2 학습과 검증 데이터셋 분리
# 이제 해당 입력값을 모델에 적용하면 되는데 그 전에 우선 학습 데이터의 일부를 검증 데이터로 따로 분리한다. 

# In[45]:


from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 42
TEST_SPLIT = 0.2

y = np.array(sentiments)

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=TEST_SPLIT,
                                                    random_state=RANDOM_SEED)


# ##### 4-3. 모델 선언 및 학습
# 선형 회귀 모델을 만들기 위해 사이킷런 라이브러리에서 지원하는 LogisticRegression 클래스의 객체를 생성한다.\
# 이후 이 객체의 fit함수를 호출하면 데이터에 대한 모델 학습이 진행된다.

# In[46]:


from sklearn.linear_model import LogisticRegression

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(X_train, y_train)


# 2장에서 사용했던 모델과 거의 사용법이 비슷하다. 간단하게 모델을 만들고 데이터에 적용하기만 하면된다. 특별한 점은 모델을 생성할 때 인자값을 설정했는데 ***class_weight***를 ***balanced***로 설정해서 각 라벨에 대해 균형 있게 학습할 수 있게 한 것이다. 

# ##### 4-4 검증 데이터로 성능평가

# 검증 데이터를 가지고 학습한 모델에 대해 성능을 확인해보자. 성능 평가는 앞서 학습한 객체의 ***score*** 함수를 이용하면 간단하게 측정할 수 있다.

# In[47]:


print('Accuracy: {}%'.format(lgs.score(X_eval, y_eval))) # 검증 데이터로 성능 측정


# 성능 평가 방법으로 여기서는 정확도(Accuracy)만 측정했는데, 이 외에도 다양한 성능 평가 지표가 있다.
# - 정밀도(Precision)
# - 재현율(Recall)
# - f1-score,
# - auc

# ##### 4-5 데이터 제출하기

# In[48]:


TEST_CLEAN_DATA = 'test_clean.csv'
test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)
test_data


# In[49]:


testDataVecs = vectorizer.transform(test_data['review'])


# 벡터화할 때 평가 데이터에 대해서는 fit을 호출하지 않고 그대로 transform만 호출한다. fit의 경우 학습 데이터에 맞게 설정했고, 그 설정에 맞게 평가 데이터로 변환하면 된다. 이제 이 값으로 예측한 후 예측값을 하나의 변수로 할당하고 출력해서 형태를 확인해보자

# In[50]:


test_predicted = lgs.predict(testDataVecs)
print(test_predicted)


# In[51]:


DATA_OUT_PATH = './data_out/'

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
    
ids = list(test_data['id'])
answer_dataset = pd.DataFrame({'id' : ids, 'sentiment': test_predicted})
answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_tfidf_answer.csv', index=False, quoting=3)


# ### 5. Word2Vec을 활용한 모델구현
# 이번에는 word2vec을 활용해 모델을 구현해보자. 우선 각 단어에 대해 word2vec으로 벡터화해야 한다.\
# word2vec의 경우 단어로 표현된 리스트를 입력값으로 넣어야 하기 때문에 전처리한 넘파이 배열을 사용하지 않는다.\
# 따라서 전처리된 텍스트 데이터를 불러온 후 각 단어들의 리스트로 나눠야 한다.

# In[52]:


DATA_IN_PATH = 'data_in/' # 파일이 저장된 경로
TRAIN_CLEAN_DATA = 'train_clean.csv'

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)

reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])

sentences = []
for review in reviews:
    sentences.append(review.split())


# 전처리한 데이터의 경우 각 리뷰가 하나의 문자열로 이뤄져 있다. 하지만 앞서 말했듯이 word2vec을 사용하기 위해서는 입력값을 단어로 구분된 리스트로 만들어야 한다. 따라서 위와 같이 전체 리뷰를 단어 리스트로 바꿔야한다. 각 리뷰를 split 함수를 사용해서 띄어쓰기 기준으로 구분한 후 리스트에 하나씩 추가해서 입력갔을 만든다.

# ##### 5-1 Word2Vec 벡터화

# In[53]:


# 학습시 필요한 하이퍼파라미터
num_features = 300  # 워드 벡터 특징 값 수
min_word_count = 40 # 단어에 대한 최소 빈도수
num_workers = 4 # 프로세스 개수
context = 10 # 컨텍스트 윈도 크기
downsampling = 1e-3 # 다운 샘플링 비율


# - num_features: 각 단어에 대해 임베딩된 벡터의 차원을 정한다.
# - min_word_count: 모델에 의미 있는 단어를 가지고 학습하기 위해 적은 빈도 수의 단어들은 학습하지 않는다.
# - num_workers: 모델 학습시 학습을 위한 프로세스 개수를 지정한다.
# - context: word2vec을 수행하기 위한 컨텍스트 윈도 크기를 지정
# - downsampling: word2vec학습을 수행할 때 빠른 학습을 위해 정답 단어 라벨에 대한 다운 샘플링 비율을 지정한다. (보통 0.001이 좋은 성능)

# In[54]:


import logging 
logging.basicConfig(format='%(asctime)s : %(levelname)s :  %(message)s',filename='./text.log',level=logging.INFO)


# 로깅을 할 떄 format을 위와 같이 지정하고, 로그 수준은 INFO에 맞추면 word2vec의 학습 과정에서 로그 메시지를 양식에 맞게 INFO 수준으로 보여준다. 이제 본격적으로 학습을 실행해 보자.
# 
# word2vec 학습을 위해서는 word2vec 모듈에 있는 word2vec 객체를 생성해서 실행한다. 이렇게 학습하고 생성된 객체는 model 변수에 할당한다. 이때 학습을 위한 객체의 인자는 입력할 데이터와 하이퍼파라미터를 순서대로 입력해야 원하는 하이퍼파라미터를 사용해 학습할 수 있다.

# In[55]:


from gensim.models import word2vec
print('Training model...')
model = word2vec.Word2Vec(sentences,
                         workers = num_workers,
                          size = num_features,
                          min_count = min_word_count,
                          window =  context,
                          sample = downsampling)


# 학습시킨 모델의 경우 모델을 따로 저장해두면 이후에 다시 사용할 수 있기 때문에 저장해두고 이후에 학습한 값이 추가로 필요할 경우 사용하면 된다.

# In[56]:


# 모델의 하이퍼파라미터를 설정한 내용을 모델 이름에 담는다면 나중에 참고하기에 좋을 것이다.
# 모델을 저장하면 Word2vec.load()를 통해 모델을 다시 사용할 수 있다.
model_name = '300features_40minwords_10context'
model.save(model_name)


# 모델을 저장하려면 앞에서 학습한 model 객체의 save 함수를 사용하면 된다.\
# 여기서 함수의 인자는 모델 이름을 작성해서 입력한다. 저장된 모델은 이미 학습된 것이기 때문에 다른 모델을 불러와서 사용하면 학습 시간 없이 바로 사용할 수 있다.
# 
# 이제 만들어진 Word2Vec 모델을 활용해 선형 회귀 모델을 학습해보자. 우선 학습을 하기 위해서는 하나의 리뷰를 같은 형태의 입력값으로 만들어야 한다. 지금은 word2vec 모델에서 각 단어가 벡터로 표현돼 있다. 그리고 리뷰마다 단어의 개수가 모두 다르기 때문에 입력값을 하나의 형태로 만들어야 한다. 
# 
# 가장 단순한 방법으로는 문장에 있몯느 단어의 벡터값에 대해 평균을 내서 리뷰 하나당 하나의 벡터로 만드는 방법이 있다. 따라서 여기서는 이 방법을 통해 입력값을 만든다. 다음과 같이 하나의 리뷰에 대해 전체 단어의 평균값을 계산하는 함수를 구현한다.

# In[57]:


def get_features(words, model, num_features):
    feature_vector = np.zeros((num_features), dtype = np.float32)
    
    num_words = 0
    # 어휘 사전
    index2word_set = set(model.wv.index2word)
    
    for w in words:
        if w in index2word_set:
            num_words +=1
            #사전에 해당하는 단어에 대해 단어 벡터를 더함
            
            feature_vector = np.add(feature_vector, model[w])
            # model은 단어들에 대한 vector를 다 가지고 있음
            # num_features 만큼 이미 학습할때 정의해서 만들어놓음 
            
    feature_vector = np.divide(feature_vector,num_words)
    
    return feature_vector


# - words: 단어의 모음인 하나의 리뷰가 들어간다.
# - model: word2vec 모델을 넣는 곳이며 우리가 학습한 word2vec 모델이 들어간다.
# - num_features: word2vec으로 임베딩할 때 정했던 벡터의 차원수가 된다.

# 하나의 벡터를 만드는 과정에서 속도를 빠르게 하기 위해 np.zeros를 사용해 미리 모두 0의 값을 가지는 벡터를 만든다. 그리고 문장의 단어가 해당 모델 단어사전에 속하는지 보기 위해 model.wv.index2word를 set 객체로 생성해서 index2word_set 변수에 할당한다. 다음 반복문을 통해 리뷰를 구성하는 단어에 대해 임베딩된 벡터가 있는 단어 벡터의 합을 구한다. 마지막으로 사용한 단어의 전체 개수로 나눔으로써 평균 벡터의 값을 구한다.
# 
# 이렇게 문장에 특징값을 만들 수 있는 함수를 구현했다면 이제 앞에서 정의한 함수를 사용해 전체 리뷰에 대해 각 리뷰의 평균 벡터를 구하는 함수를 정의한다.

# In[58]:


def get_dataset(reviews, model, num_features):
    dataset = list()
 
    
    for s in reviews :
        dataset.append(get_features(s,model,num_features))
    
    reviewFeaturevecs = np.stack(dataset)
    
    return reviewFeaturevecs


# In[59]:


test_data_vecs = get_dataset(sentences,model, num_features)


# 속도 향상을 위해 전체 리뷰에 대한 평균 벡터를 담을 0으로 채워진 넘파이 배열을 미리 만든다. 배열은 2차원으로 만드는데 배열의 행에는 각 문장에 대한 길이를 입력하면 되고 열에는 평균 벡터의 차원 수, 즉 크기를 입력하면 된다. 그리고 각 리뷰에 대해 반복문을 돌면서 각 리뷰에 대해 특징 값을 만든다.

# ##### 5-2 학습과 검증 데이터셋 분리
# 만들어진 데이터를 가지고 학습 데이터와 검증 데이터를 나눠보자. 나누는 방식은 앞서 TF-iDF에서 진행했던 방식과 동일하다

# In[60]:


from sklearn.model_selection import train_test_split
import numpy as np

X = test_data_vecs
y = np.array(sentiments)

RANDOM_SEED = 42
TEST_SPLIT = 0.2

X_train, X_eval, y_train, y_eval = train_test_split(X, y , test_size=TEST_SPLIT,
                                                    random_state=RANDOM_SEED)


# 이제 학습 데이터의 일부를 검증 데이터로 분리했다.
# 

# ##### 5-3 모델 선언 및 학습
# 모델의 경우 TF-IDF 벡터를 사용했을 때와 동일하게 로지스틱 모델을 사용한다. TF-IDF 모델과 비교하면 입력값을 어떤 특징으로 뽑았는지만 다르고 모두 동일하다.

# In[61]:


from sklearn.linear_model import LogisticRegression
lgs = LogisticRegression(class_weight = 'balanced')
lgs.fit(X_train, y_train)


# 모델을 생성할 때 TF-IDF 모델과 동일하게 class_weight라는 인자값을 'balanced'로 설정했다.\
# 이는 역시 각 라벨에 대해 균형 있게 학습하기 위함이다. 이렇게 생성한 데이터에 학습 데이터를 적용하면 이제 다른 데이터에 대해 학습을 진행할 수 있다.

# ##### 5-4 검증 데이터셋을 이용한 성능 평가

# In[62]:


print('Accuracy: {}%'.format(lgs.score(X_eval, y_eval))) #검증 데이터로 성능 측정


# 학습한 결과를 보면 TF-IDF를 사용해서 학습한 것보다 상대적으로 성능이 조금 떨어지는 것을 볼 수 있다.\
# 이런 결과를 보면 word2vec 모델이 생각보다 성능이 떨어지는 것에 의아해할 수 있다. 물론 word2vec이 단어 간의 유사도를 보는 관점에서는 분명히 효과적일 수 있지만 word2vec을 사용하는 것이 항상 가장 좋은 성능을 보장하지는 않는다.

# ##### 5-5 데이터 제출

# In[63]:


TEST_CLEAN_DATA = 'test_clean.csv'
test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)

test_review = list(test_data['review'])


# 폎가 데이터 역시 학습 데이터와 마찬가지로 각 리뷰가 하나의 문자열로 이뤄져 있다.\
# 따라서 평가 데이터도 각 단어의 리스트로 만들어야 한다.

# In[64]:


test_sentences = []
for review in test_data:
    test_sentences.append(review.split())


# 평가 데이터도 단어의 리스트로 만든 후 word2vec으로 임베딩된 벡터값을 갖게 해야 한다.\
# 평가 데이터에 대해 새롭게 word2vec 모델을 학습시키는 것이 아니라 이전에 학습 시킨 모델을 사용해 각 단어들을 벡터로 만들어 각 리뷰에 대한 특징값을 만든다. 그리고 나서 이전에 정의했던 함수에 동일하게 적용하면 된다.

# In[65]:


test_data_vesc = get_dataset(test_sentences, model, num_features)


# 이렇게 평가 데이터에 대해 각 리뷰를 특징 벡터로 만들었다면 이제 학습시킨 로지스텍 모델에 적용해 결과를 확인하면 된다. 해당 데이터의 평가 데이터는 라벨을 가지고 있지 않으므로 예측한 값을 따로 저장해서 캐글에 제출함으로써 성능을 측정해야한다.

# In[66]:


DATA_OUT_PATH = './data_out/'

test_predicted = lgs.predict(test_data_vecs)

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
    
ids = list(test_data['id'])
answer_dataset = pd.DataFrame({'id': ids, 'sentiment':test_predicted})
answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_answer.csv')


# ### 6. 랜덤포레스트 분류 모델

# ##### 6-1. CounterVectorizer를 활용한 벡터화
# 모델을 구현하기에 앞서 모델에 사용할 입력값을 정해야 한다. 이전 선형 회귀 모델에서는 TF-IDF와 Word2Vec을 활용해서 벡터화한 데이터를 입력값으로 사용했다. 랜덤 포레스트 모델에서는 또 다른 방법인 ***CounterVectorizer***를 사용해 모델의 입력값을 만든다.

# In[67]:


import pandas as pd

DATA_IN_PATH = './data_in/'
TRAIN_CLEAN_DATA = 'train_clean.csv'

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
reviews = list(train_data['review'])
y = np.array(train_data['sentiment'])


# 불러온 텍스트 데이터를 특징 벡터로 만들어야 모델에 입력값으로 사용할 수 있다. 앞서 말했듯이 이번 절에서는 모델에 적용하기 위한 특징 추출 방법으로 ***CounterVectorizer***를 이용한 방법을 사용한다. 

# In[68]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = 'word', max_features = 5000)
train_data_features = vectorizer.fit_transform(reviews)


# 벡터화된 값을 변수에 할당한다. 객체를 생성할 때 인자 값을 설정하는데 분석 단위를 하나의 단위로 지정하기 위해 analyzer를 word로 설정하고 각 벡터의 최대 길이를 5000으로 설정했다.

# In[69]:


train_data_features


# ##### 6-2. 학습과 검증 데이터 분리

# In[70]:


TEST_SIZE = 0.2
RANDOM_SEED = 42

train_input, eval_input, train_label, eval_label = train_test_split(train_data_features,
y, test_size=TEST_SIZE, random_state=RANDOM_SEED)


# ##### 6-3 모델 구현및 학습
# RandomForestClassifier 객체를 사용해 구현한다. 

# In[71]:


from sklearn.ensemble import RandomForestClassifier

# 랜덤 포레스트 분류기에 100개의 의사결정 트리를 사용한다.
forest = RandomForestClassifier(n_estimators=100)

# 단어 묶음을 벡터화한 데이터와 정답 데이터를 가지고 학습을 시작한다.
forest.fit(train_input, train_label)


# 트리의 개수를 100개로 설정했다.

# ##### 6-5. 검증 데이터셋으로 성능 평가

# In[72]:


print("Accuracy: {}%".format(forest.score(eval_input, eval_label))) # 검증 함수로 정확도 측정


# ##### 6-6 데이터 제출

# In[73]:


TEST_CLEAN_DATA = 'test_clean.csv'
DATA_OUT_PATH = './data_out/'

test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)

test_reviews = list(test_data['review'])
ids = list(test_data['id'])


# In[74]:


test_data_features = vectorizer.transform(test_reviews)


# In[75]:


if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
    
result = forest.predict(test_data_features)

output = pd.DataFrame(data = {'id': ids, 'sentiment': result})
output.to_csv(DATA_OUT_PATH + 'Bag_of_words_model.csv', index=False, quoting=3)


# ### 7. 순환 신경망 분류 모델(RNN)
# 순환 신경망은 언어 모델에서 많이 쓰이는 딥러닝 모델 중 하나다. 주로 순서가 있는 데이터, 즉 문장 데이터를 입력해서 문장 흐름에서 패턴을 찾아 분류한다. 앞선 모델들과 달리 이미 주어진 단어 특징 벡터를 활용해 모델을 학습하지 않고 텍스트 정보를 입력해서 문장에 대한 특징 정보를 추출한다.
# 
# ##### 7-1 모델 소개
# 순환 신경망(RNN)은 현재 정보는 이전 정보가 점층적으로 쌓이면서 정보를 표현할 수 있는 모델이다.\
# 따라서 시간에 의존적인 혹은 순차적인 데이터에 대한 문제에 활용된다.
# 
# 현재 정보를 입력 상태(input state)라 부르고 이전 정보를 은닉 상태(hidden state)라 부른다. 순환 신경망은 이 두 상태 정보를 활용해 순서가 있는 데이터에 대한 예측 모델링을 가능하게 한다. 마지막 시간 스텝에 나온 은닉 상태는 문장 전체 정보가 담긴 정보로서 이 정보를 활용해 영화 평점을 예측할 수 있도록 로지스틱 회귀 또는 이진 분류를 하면 된다. 

# ##### 7-2 랜던 시드 고정

# In[76]:


import tensorflow as tf


# In[77]:


SEED_NUM = 1234
tf.random.set_seed(SEED_NUM)


# ##### 7-3 학습 데이터 불러오기
# 앞서 4-2절에서 전처리해둔 데이터를 불러온다.

# In[78]:


DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'
TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
DATA_CONFIGS = 'data_configs.json'

train_input = np.load(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'rb'))
train_input = pad_sequences(train_input, maxlen=train_input.shape[1])
train_label = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'rb'))
prepro_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))


# 입력 텍스트에 대해서 모델을 학습할 때 텍스트 길이를 맞추기 해 tensorflow.keras.preprocessing.sequence 모듈에 있는 pad_sequences 함수를 사용했다. 

# ##### 7-4 모델 하이퍼파라미터 정의ㅡ

# In[79]:


model_name = 'rnn_classifier_en'
BATCH_SIZE = 128
NUM_EPOCHS = 5
VALID_SPLIT = 0.1
MAX_LEN = train_input.shape[1]

kargs = {'model_name': model_name,
        'vocab_size': prepro_configs['vocab_size'],
        'embedding_dimension': 100,
        'dropout_rate': 0.2,
        'lstm_dimension': 150,
        'dense_dimension': 150,
        'output_dimension': 1}


# 모델에 대한 하이퍼파라미터는 ***모델 학습***을 위한 설정과 ***모델 레이어의 차원 수*** 설정으로 나뉜다.
# 1. 모델 학습을 위한 설정
#     - 배치크기
#     - 에폭 수
#     - 텍스트 데이터 길이
#     - validation 데이터셋 구성 비율
# 2. 모델 레이어의 차원 수
#     - __init__ 함수 파라미터에 입력하기 위해ㅣdict 객체에서 정의
#     - 각 모델 레이어 차원 수, 드롭아웃 값을 정하는 하이퍼파라미터 명칭은 key
#     - 키에 해당하는 하이퍼파라미터 명칭에 대한 값은은 value에 입력

# ##### 7-6 모델 구현

# In[80]:


from tensorflow import keras


# In[81]:


class RNNClassifier(tf.keras.Model):
    # __init__: 모델 객체를 생성할때마다 실행, 하이퍼파라미터 정보를 dict 객체로 받는다. 
    def __init__(self, **kargs):
        # 클래스를 상속 받는 경우 super함수를 통해 부모 클래스에 있는 __init__함수를 호출해야 한다.
        super(RNNClassifier, self).__init__(name=kargs['model_name'])
        self.embedding = keras.layers.Embedding(input_dim = kargs['vocab_size'],
                                        output_dim = kargs['embedding_dimension'])
        
        self.lstm_1_layer = keras.layers.LSTM(kargs['lstm_dimension'],
                                                return_sequences=True)
        self.lstm_2_layer = keras.layers.LSTM(kargs['lstm_dimension'])
        self.dropout = keras.layers.Dropout(kargs['dropout_rate'])
        
        self.fc1 = keras.layers.Dense(units=kargs['dense_dimension'], 
                               activation=tf.keras.activations.tanh)
        self.fc2 = keras.layers.Dense(units=kargs['output_dimension'],
                               activation=tf.keras.activations.sigmoid)
        
    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.lstm_1_layer(x)
        x = self.lstm_2_layer(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# 클래스로 모델을 구현하려면 ***tf.keras.Model***을 상속 받아야 한다.\
# 그래서 tf.keras.Model을 상속 받기 위해 ***'class RNNClassifier(tf.keras.Model)'*** 로 구현을 시작한다.
# 
# - 가장 먼저 구현할 함수는 ***__init__*** 함수다.
#     - RNNClassifier 모델 객체를 생성할 때마다 실행
#     - 매개변수로 모델 레이어의 입력 및 출력 차원 수를 정의하는 하이퍼파라미터 정보를 dict 객체로 받는다.
#     
#     
# - __init__ 함수의 구현을 시작하면 먼저 super 함수를 확인할 수 있다.
#     - tf.keras.Model 클래스를 상속받는 경우 super 함수를 통해 부모 클래스에 있는 __init__ 함수를 호출해야 한다.
#     - tf.keras.model 클래스를 상속 받는 경우 super 함수를 통해 부모 클래스에__init__함수의 인자에 모델 이름을 전달하면 tf.keras.Model을 상속 받은 모든 자식은 해당 모델의 이름으로 공통적으로 사용
#     - 그 다음 워드 임베딩 벡터를 위해 layers.Embedding 객체를 생성한다. 이 때 파라미터로 데이터 사전 수와 단어 임베딩 차원수를 입력한다.
#     
#     
# - RNN Classificaiton 모델은 
#     - 워드 임베딩에서 ***RNN 레이어***를 거쳐 ***RNN 레이어 시퀀스***의 마지막 은닉 상태 벡터를 사용
#     - 구현할 RNNClassifier 클래스에서는 RNN 계열 모델 중 하나인 LSTM을 2개의 레이어로 활용함
#     - LSTM을 활용하기 위해서는 ***tf.keras.layers.LSTM*** 객체를 생성한다.
#     - 이때 입력 파라미터로 레이어 출력 차원 수와 출력 시퀀스를 전부 출력할지 여부를 묻는
#         - return_sequences를 입력한다.
#         - True로 입력할 경우 시퀀스 형태의 은닉 상태 벡터가 출력됨
#         
#         
# - 두개의 LSTM 레이어를 활용해 마지막 시퀀스의 은닉 상태 벡터를 얻기 위해선
#     - 첫 레이어에서 시퀀스 은닉 상태 벡터를 출력해 다음 레이어에 입력할 시퀀스 벡터를 구성
#     - 마지막 레이어에서는 시퀀스의 마지막 스텝의 은닉 상태 벡터를 출력해야 함
#     - 따라서 첫 번째 레이어인 LSTM 객체에서만 return_sequences=True
#     
#     
# - 피드 포워드 네트워크
#     - tf.keras.layers.Dense를 통해 객체를 생성해 피드 포워드 네트워크를 구성한다.
#     - 객체 생성시 입력 파라미터로 
#         - units: 네트워크 출력할 때 나오는 벡터 차원 수 
#         - activations: 네트워크에서 사용할 활성화 함수
#         
#         
# - 회귀
#     - Dense가 위의 네트워크를 거쳐 나온 상태 벡터에서 회귀하게 할 수 있다.
#     - activation에는 tf.keras.activations.sigmoid를 지정
#     
#     
# - Dropout 선언
#     - 과적합을 방지하기 위한 레이어
#     - tf.keras.layers.Dropout을 활용해 생성 (파라미터는 적용할 비율 값)
#   

# ##### 7-7 모델 생성

# In[82]:


model = RNNClassifier(**kargs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
             loss = tf.keras.losses.BinaryCrossentropy(),
             metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])


# RNNClassifier 클래스를 생성한다.\
# 그리고 model.compile을 통해 학습할 옵티마이저나 손실 함수, 평가를 위한 평가지표 등을 설정 한다.

# ##### 7-8 모델 학습

# In[83]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# overfitting을 막기 위한 earlystop 추가
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)
# min_delta: the threashold that triggers the termination(acc should at least imporve 0.0001)
# patience: no improvement epochs(patience =1 , 1번 이상 상승이 없으면 종료)

checkpoint_path = DATA_OUT_PATH + model_name + '/weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok= True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))
    
cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=True
)

history = model.fit(train_input, train_label, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                    validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])


# 사실 model.fit 함수만 사용해도 모델 학습이 진행된다.
# - 그러나 오버피팅이 발생할 수도 있고 학습 도중 특정 상태의 모델에서 하이퍼 파라미터를 바꿔서 다시 학습을 진행할 수도 있다.
# - 이를 위해 tensorflow.keras.callback 모듈에 있는 ***EarlyStopping***과 ***ModelCheckPoint***라는 클래스를 활용할 수 있다.

# In[84]:


epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[85]:


epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'])
plt.plot(epochs, history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# ##### 7-9 데이터 제출

# In[86]:


DATA_OUT_PATH = './data_out/'
TEST_INPUT_DATA = 'test_input.npy'
TEST_ID_DATA = 'test_id.npy'
SAVE_FILE_NM = 'weights.h5'

test_input = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb'))
test_input = pad_sequences(test_input, maxlen=test_input.shape[1])

model.load_weights(os.path.join(DATA_OUT_PATH, model_name, SAVE_FILE_NM))


# In[87]:


predictions = model.predict(test_input, batch_size = BATCH_SIZE)
predictions = predictions.squeeze(-1)

test_id = np.load(open(DATA_IN_PATH + TEST_ID_DATA, 'rb'), allow_pickle=True)
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
    
output = pd.DataFrame(data={'id':list(test_id), 'sentiment':list(predictions)})
output.to_csv(DATA_OUT_PATH + 'movie_review_result_rnn.csv', index=False, quoting=3)


# 앞서 데이터를 전처리하면서 저장한 테스트 입력 데이터를 불러온다. 데이터를 불러오는 방법은 학습 데이터를 불러올 때와 동일하다. 그리고 가장 좋은 검증 점수가 나온 모델을 model.load_weights로 불러오자.

# 모델 예측의 경우 model.predict 함수를 통해 한 번에 배치 사이즈만큼 입력해서 예측할 수 있게 했다.
# 

# ### 8. 컨볼루션 신경망 분류모델(CNN)
# ##### 8-1 모델 소개
# 합성 신경망(CNN)은 여러 계층의 합성곱(Convolution) 계층을 쌓은 모델인데, 입력 받은 이미지에 대한 가장 좋은 특징을 만들어 내도록 학습하고 추출된 특징을 활용해 이미지를 분류하는 방식
# 
# 텍스트에서도 좋은 효과를 낼 수 있다는 점을 Yoon Kim(2014) 박사가 쓴 "Convolutional Neural Network for sentence Classification"을 통해 입증
# ![ㅇㅇ](https://d3i71xaburhd42.cloudfront.net/06b919f865d0a0c3adbc10b3c34cbfc35fb98d43/4-Figure1-1.png)

# RNN이 단어의 입력 순서를 중요하게 반영한다면 CNN은 문장의 지역 정보를 보존하면서 각 문장 성분의 등장 정보를 학습에 반영하는 구조로 풀어가고 있다. 학습할 때 각 필터 크기를 조절하면서 언어의 특징 값을 추출하게 되는데, 기존의 n-gram(2그램, 3그램) 방식과 유사하다고 볼 수 있다. 

# ##### 8-2 모델 구현
# 모델에 필요한 하이퍼파라미터를 정의해보자. RNN에서 설명한 바와 같이 ***모델 학습 설정***, ***모델 레이어 차원 수*** 등으로 나누고 학습을 위한 배치 크기 등은 변수로 지정하고 모델에 필요한 내용은 모델의 __init__ 함수 파라미터에 입력하기 위해 dict 객체로 정의한다.

# In[88]:


BATCH_SIZE = 512
NUM_EPOCHS = 5
VALID_SPLIT = 0.1
MAX_LEN = train_input.shape[1]

kargs = {'model_name': model_name,
        'vocab_size': prepro_configs['vocab_size'],
         'embedding_size': 128,
         'num_filters': 100,
         'dropout_rate': 0.5,
         'hidden_dimension': 250,
         'output_dimension': 1}


# In[89]:


class CNNClassifier(tf.keras.Model):
    
    def __init__(self, **kargs):
        super(CNNClassifier, self).__init__(name=kargs['model_name'])
        self.embedding = keras.layers.Embedding(input_dim=kargs['vocab_size'],
                                         output_dim = kargs['embedding_size'])
        
        self.conv_list = [keras.layers.Conv1D(filters=kargs['num_filters'],
                                       kernel_size=kernel_size,
                                       padding='valid',
                                       activation=tf.keras.activations.relu,
                           kernel_constraint = tf.keras.constraints.MaxNorm(max_value=3.))
                         for kernel_size in [3,4,5]]
        
        self.pooling = keras.layers.GlobalMaxPooling1D()
        self.dropout = keras.layers.Dropout(kargs['dropout_rate'])
        self.fc1 = keras.layers.Dense(units=kargs['hidden_dimension'],
                               activation = tf.keras.activations.relu,
                               kernel_constraint = tf.keras.constraints.MaxNorm(max_value=3.))
        self.fc2 = keras.layers.Dense(units=kargs['output_dimension'], 
                               activation=tf.keras.activations.sigmoid,
                               kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
        
    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = tf.concat([self.pooling(conv(x)) for conv in self.conv_list], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


# - Yoon Kim의 CNN과 유사하게 구현했다. 
#     - 임베딩 벡터를 생성
#     - 케라스의 Conv1D를 활용해 총 3개의 합성곱 레이어를 사용하고 각각 필터의 크기를 다르게 해서 값을 추출
#     - self.conv_list로 관련 함수를 저장하며 kernel_size의 입력에 리스트 컴프리헨션 형태로 구현
#         - kernel_size가 3,4,5인 Conv1D 객체 리스트로 구현
# 
# - 합성곱 신경망 이후에 맥스 풀링 레이어를 적용
#     - 총 3개의 합성곱 + 맥스 풀링 레이어를 사용
#     - 과적합 방지하기 위한 Dropout과 완전 연결 계층(fully-connected) 2개 층을 쌓아 최종 출력 차원인 kargs['output_dimension']과 출력을 맞춰 모델 구성
# 
# - Call함수를 통해 실행 
#     - 리스트 컴프리헨션 형태의 Conv1D 리스트 값을 각각 다른 필터의 값이 Conv1D를 통해 문장의 각기 다른 표현값들을 추출해서 concat을 통해 출력값들을 합친다. 
#     - 완전 연결 계층(fc)을 통해 분류 모델을 만들기 위한 학습 모델 구조를 완성한다.

# ##### 8-3 모델 생성

# In[90]:


model = CNNClassifier(**kargs)
model.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
             loss=tf.keras.losses.BinaryCrossentropy(),
             metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])


# 모델은 앞에서 구현한 CNNClassifier 클래스를 생성한다. 그리고 model.compile을 통해 학습할 옵티마이저나 손실 함수 및 평가를 위한 평가지표 등을 설정한다. 

# ##### 8-4 모델 학습
# 

# In[91]:


earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)

checkpoint_path = DATA_OUT_PATH + model_name + '/weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))
    
cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=True)

history = model.fit(train_input, train_label, batch_size=BATCH_SIZE, 
                    epochs=NUM_EPOCHS, validation_split = VALID_SPLIT, 
                    callbacks=[earlystop_callback, cp_callback])


# In[95]:


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string],'')
    plt.xlabel('Epochs')
    plt.ylabel('string')
    plt.legend([string, 'val_'+string])
    plt.show()


# In[96]:


plot_graphs(history, 'loss')


# In[97]:


plot_graphs(history, 'accuracy')


# ##### 8-5 데이터 제출
# 

# In[99]:


DATA_OUT_PATH = './data_out/'
TEST_INPUT_DATA = 'test_input.npy'
TEST_ID_DATA = 'test_id.npy'

test_input = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb'))
test_input = pad_sequences(test_input, maxlen=test_input.shape[1])

SVAE_FILE_NM = 'weights.h5'

model.load_weights(os.path.join(DATA_OUT_PATH, model_name , SVAE_FILE_NM))


# In[100]:


predictions = model.predict(test_input, batch_size = BATCH_SIZE)
predictions = predictions.squeeze(-1)

test_id = np.load(open(DATA_IN_PATH + TEST_ID_DATA, 'rb'), allow_pickle=True)

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
    
output = pd.DataFrame(data={'id': list(test_id), 'sentiment': list(predictions)})
output.to_csv(DATA_OUT_PATH + 'movie_review_result_cnn.csv', index=False, quoting=3)      


# In[ ]:




