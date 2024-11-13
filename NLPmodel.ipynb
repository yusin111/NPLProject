import pandas as pd
import urllib.request # 인터넷을 이용하여 데이터 요청
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt # 한국어 불용어 처리시
from tqdm import tqdm # 진행률 바 표기
import numpy as np

#seperate title [1. 리뷰파일 다운로드] ================
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

#seperate title [2. 판다스로 데이터 확인] ================
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
print(train_data.info())
print(test_data.info())
# >>> 출력 결과
# >>> <class 'pandas.core.frame.DataFrame'>
# >>>  #   Column    Non-Null Count   Dtype 
# >>> ---  ------    --------------   ----- 
# >>>  0   id        150000 non-null  int64 
# >>>  1   document  149995 non-null  object
# >>>  2   label     150000 non-null  int64 
# >>> dtypes: int64(2), object(1)
# >>> <class 'pandas.core.frame.DataFrame'>
# >>>  #   Column    Non-Null Count  Dtype 
# >>> ---  ------    --------------  ----- 
# >>>  0   id        50000 non-null  int64 
# >>>  1   document  49997 non-null  object
# >>>  2   label     50000 non-null  int64 
# >>> dtypes: int64(2), object(1)
# *개발자 분석내용 - document 필드의 수량이 다른 필드의 수량과 다르기 때문에 결측 데이터가 존재한다.

#seperate title [3. 결측데이터 수량 확인 및 제거] ================
print("훈련데이터 결측수량:",train_data["document"].isna().sum())
print("테스트데이터 결측수량:",test_data["document"].isna().sum())
#ref 참조    DataFrame.dropna(*, axis=0, how=<no_default>, thresh=<no_default>, subset=None, inplace=False, ignore_index=False)[source]
train_data = train_data.dropna(axis=0,subset="document")
test_data = test_data.dropna(axis=0,subset="document")
print("훈련데이터 결측수량:",train_data["document"].isna().sum())
print("테스트데이터 결측수량:",test_data["document"].isna().sum())
# >>> 출력 결과
# >>> 훈련데이터 결측수량: 5
# >>> 테스트데이터 결측수량: 3
# >>> 훈련데이터 결측수량: 0
# >>> 테스트데이터 결측수량: 0
# *개발자 분석내용 - 최초에 훈련데이터에 5개의 결측데이터가 관측되었고, 테스트데이터에 3개의 결측데이터가 관측되어 pandas의 dropna명령으로 제거하였다.

#seperate title [4. 중복데이터 확인 및 제거] ================
print(train_data["document"].count()) # 총 데이터 수량
print(train_data["document"].nunique()) # 유니크한 데이터의 수량
# >>> 149995
# >>> 146182
# *개발자 분석내용 - 총 데이터 수량과 유니크 데이터 수량의 차이가 있음은 중복된 데이터가 존재하고 있다.

print("중복된 훈련 데이터의 수:",train_data["document"].count()-train_data["document"].nunique())
print(test_data["document"].count())
print(test_data["document"].nunique())
print("중복된 테스트 데이터의 수:",test_data["document"].count()-test_data["document"].nunique())
# >>> 중복된 훈련 데이터의 수: 3813
# >>> 49997
# >>> 49157
# >>> 중복된 테스트 데이터의 수: 840
# *개발자 분석내용 - 테스트 데이터 또한 중복 내용이 존재하고 있으나 훈련 대상 데이터가 아니지만 중복 내용을 제거 하겠다.

#ref 참조   DataFrame.drop_duplicates(subset=None, *, keep='first', inplace=False, ignore_index=False)[source]
train_data=train_data.drop_duplicates(subset="document") # 훈련 데이터 중복 제거
test_data=test_data.drop_duplicates(subset="document") # 테스트 데이터 중복 제거
print("중복된 훈련 데이터의 수:",train_data["document"].count()-train_data["document"].nunique())
print("중복된 테스트 데이터의 수:",test_data["document"].count()-test_data["document"].nunique())
# >>> 중복된 훈련 데이터의 수: 0
# >>> 중복된 테스트 데이터의 수: 0
# *개발자 분석내용 - 모든 데이터의 중복이 제거되어 중복데이터 수가 0으로 표기 되었다.

#seperate title [5. 한글을 제외한 문자 제거와 형태소별로 분류] ================
print(train_data[:5])
# >>>         id                                   document                              label
# >>> 0   9976970                         아 더빙.. 진짜 짜증나네요 목소리                   0
# >>> 1   3819312                흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나      1
# >>> 2  10265843                        너무재밓었다그래서보는것을추천한다                   0
# *개발자 분석내용 - ...? 영문 등은 감성분석에 불필요하므로 제거 대상이다.
# 정규표현식을 이용한 한글과 공백을 제외한 모든 단어는 제거
train_data["document"] = train_data["document"].replace(r"[^\sㄱ-ㅎㅏ-ㅣ가-힣]","",regex=True)
print(train_data[:5])
# >>>         id                                    document                            label
# >>> 0   9976970                          아 더빙 진짜 짜증나네요 목소리                   0
# >>> 1   3819312               흠포스터보고 초딩영화줄오버연기조차 가볍지 않구나             1
# >>> 2  10265843                         너무재밓었다그래서보는것을추천한다                 0
test_data["document"] = test_data["document"].replace(r"[^\sㄱ-ㅎㅏ-ㅣ가-힣]","",regex=True)
print(test_data[:5])
