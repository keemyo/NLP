import os
import re
import json 

import numpy as np 
import pandas as pd 
from tqdm import tqdm

from konlpy.tag import Okt

# 데이터 처리를 위해 활용하는 모듈들

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX =0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25

# PAD: 어떤 의미도 없는 패딩 토큰
# SOS: 시작 토큰
# END: 종료 토큰
# UNK: 사전에 없는 단어


# inputs, outputs에는 question과 answer가 존재
def load_data(path):
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])
    
    return question, answer

# 단어 사전 만들기: 데이터 전처리 후 단어 리스트로 먼저 만들어야 한다
## 특수기호 사용해 특수기호 모두 제거, 공백 문자 기준 단어 나눠서 모든 단어 포함하는 단어리스트 만든다.
def data_tokenizer(data):
    words =[]
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    return [word for word in words if word]

# prepro_like_morphized: 한글 텍스트를 토크나이징 위해 형태소로 분리하는 함수
## 형태소로 분류한 데이터를 받아 morphs 함수를 통해 토크나이징된 리스트 객체를 받고 이를 공백 문자를 기준으로 문자열로 재구성해서 반환
def prepro_like_morphized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphized_seq)
        
    return result_data

def load_vocabulary(path, vocab_path):
    vocabulary_list = [] #경로에 단어 사전 파일이 있다면 불러와서 사용
    if not os.path.exists(vocab_path): # 단어 사전 파일이 없다면 새로 만드는 구조 
        if (os.path.exists(path)):  # 데이터를 불러와서 앞서 정의한 함수로 토크나이징 - 단어리스트
            data_df = pd.read_csv(path, encoding='utf-8')
            question, answer = list(dat_df['Q']), list(data_df['A']) 
            data = []
            data.extend(question)
            data.extend(answer)
            
            words = data_tokenizer(data)
            words = list(set(words)) # set으로 중복 제거 후 단어 리스트 만든다.
            words[:0] = MARKER # MARKER로 특정 토큰들을 단어 리스트 앞에 추가 
        
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')
                
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())
            
    word2idx, idx2word = make_vocabulary(vocabulary_list)
    
    return word2dix, idx2word, len(word2idx)

# 단어리스트를 인자로 받고, 두 개의 딕셔너리를 만든다.
## 단어에 대한 인덱스 + 인덱스에 대한 단어를 나타내도록 
def make_vocabulary(vocabulary_list):
    # 리스트를 키가 단어이고 값이 인덱스인 딕셔너리를 만든다.
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
    
    # 리스트를 키가 인덱스이고 값이 단어인 딕셔너리를 만든다.
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}
    
    # 두 개의 딕셔너리를 넘겨 준다.
    return word2idx, idx2word

word2idx, idx2word, vocab_size = load__vocabulary(PATH, VOCAB_PATH)

# 이제 불러온 데이터를 대상으로 인코더 부분과 디커도 부분에 대해 각각 전처리해야 한다.
## 우선 인코더에 적용될 입력값을 만드는 전처리함수를 확인 
def enc_processing(value, dictionary):
    sequences_input_idex = []
    sequences_length = []
    
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        
        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[UNK]])

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]

        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]

        sequences_input.append(sequence_index)

    return np.asarray(sequences_input_index), sequences_length

# 함수를 보면 2개의 인자를 받는데, 하나는 전처리할 데이터/ 나머지 하나는 단어 사전이다.
# 입력 데이터를 대상으로 전처리 진행하는데, 띄어쓰기를 기준으로 토크나이징 한다.

def dec_output_processing(value, dictionary):
    sequences_output_index = []
    sequences_length = []
    
    for sequence in value:
        sequence = re.sub(CHANGEFILTER, "", 


                               
        
