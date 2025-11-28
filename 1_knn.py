# 10_AI_Prediction/1_knn.py

'''
    * 인공지능 (AI, Artificial Intelligence)
    : 사람처럼 학습하고 추론할 수 있는 기능을 컴퓨터가 할 수 있도록 만드는 기술

    * 머신러닝 (ML, Machine Learning)
    : 인공지능 기술 중 하나이며, 데이터를 기반으로 패턴을 학습해 새로운 데이터를 예측하는 알고리즘

      - 지도 학습
        : 정답이 있는 데이터를 사용하여 학습하는 방식
        : 대표적인 유형
          1) 회귀 (Regression) : 숫자를 예측하는 문제
          2) 분류 (Classification) : 종류를 분류하는 문제

      - 비지도 학습
        : 정답없이 스스로 패턴을 찾도록 학습하는 방식
        : 대표적인 유형
          1) 군집화 (Clustering)
          2) 차원 축소

    * 딥러닝 (DL, Deep Learning)
      : 인공 신경망을 기반으로 한 머신러닝 알고리즘
      : 비정형 데이터(이미지, 음악 등)의 특징을 추출하고 학습
'''
# ------------------------------------------------------------

# 사이킷런(scikit-learn) 라이브러리 설치
# > pip install scikit-learn

import numpy as np
# KNN (K-Nearest Neighbor) : K-최근접 이웃 알고리즘
from sklearn.neighbors import KNeighborsRegressor   # 회귀
from sklearn.neighbors import KNeighborsClassifier  # 분류
from sklearn.model_selection import train_test_split

# - KNN 회귀 : 가까운 K개의 이웃의 평균값으로 결과를 예측
def run_knnr():
  '''
      knn 회귀 연습
      - 목표 : 공부 시간을 입력하면 시험 점수를 예측하는 모델
  '''
  print('* --- Run KNNR --- *')

  # 샘플 데이터
  # X = [1, 2, 3, 4, 5, 6, 7]   # 공부 시간
  # => 입력 데이터는 2차원 배열 형태로 요구함! ValueError!!
  X = np.array([[1], [2], [3], [4], [5], [6], [7]])
  Y = np.array([40, 45, 50, 55, 60, 65, 70])

  # ============ * 데이터 분리 * ==========================
  # 데이터셋 => 훈련용데이터셋 + 테스트용데이터셋
  # train_test_split : 데이터를 '훈련용'과 '테스트용(검증)'으로 나누는 함수. 모델이 공부하는 데이터와 실제 평가하는 데이터를 구분할 때 사용.
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
  # 입력데이터_훈련용데이터, 입력_테스트용데이터, 예측_훈련용데이터, 예측_테스트용데이터 = train_test_split(입력데이터, 예측데이터, test_size=테스트용비율(0~1),random_state=데이터셔플비율)

  # ============ * 학습 * =================================
  # 가장 가까운 5명의 평균을 이용해 점수를 예측한다.
  # => k=5
  model = KNeighborsRegressor(n_neighbors=2)      # 모델 생성
  '''
      KNeighborsRegressor(n_neighbors, weights)

      n_neighbors : 사용할 이웃 수(K). 기본값: 5
      weights : 이웃에 대한 가중치 방식. 기본값: 'uniform'
      - uniform  : 균일한 가중치. 모든 이웃에 동일한 가중치
      - distance : 가까운 이웃에 더 큰 가중치 부여
  '''

  # fit() : 모델을 학습시키는 함수. 훈련용 데이터를 사용해 학습.
  model.fit(X_train, Y_train)                     # 학습

  '''
      fit(X, Y)
      - X : 훈련 데이터(입력)
      - Y : 목표 데이터(결과)
  '''

  # 예측
  test_data = np.array([[5.5]]) # 공부시간
  prediction = model.predict(test_data)   # 5.5를 전달해서 결과 예측
  '''
    Y = predict(X)
    - X : 테스트 샘플(입력)
    - Y : 예측 데이터(예측 결과)
  '''

  print(f'{test_data} 시간 공부 시.. 예측 점수 : {prediction}')

  # 성능 평가 => score(입력_테스트데이터, 예측_테스트데이터)
  r2 = model.score(X_test, Y_test)

  '''
      r2 = score(X, Y)
      - 회귀 모델의 성능 평가 지표를 반환. 1에 가까울수록 정확하게 예측한다는 의미
      - X : 검증용(테스트) 입력 데이터
      - Y : 검증용(테스트) 목표 데이터
      - r2 : 결정 계수
  '''

  print(f'결정 계수(성능 평가) : {r2:.2f}')
  # => 1에 가까울수록 좋은 모델을 의미함!

  # 예측 시 사용된 이웃들의 데이터 확인
  # => 입력값(테스트샘플)과 가장 가까운 K개의 이웃의 거리와 인덱스를 반환
  #    이웃들의 데이터를 직접 확인하여 예측 근거로 이해할 수 있음

  distances, indices = model.kneighbors(test_data)

  print('* --- 이웃 정보 --- *')
  print('* 거리: ')
  print(distances)
  print('* 인덱스: ')
  print(indices)

  print(X[indices])
  print(Y[indices])

# run_knnr()

def run_knnc():
  '''
      knn 분류 연습
      - 목표 : 키와 몸무게를 기반으로 체형 분류
  '''
  print('* --- RUN KNNC --- *')

  # 입력 데이터 [키, 몸무게]
  x = np.array([
    [160, 55], [165, 60], [170, 65]       # 보통
    , [160, 70], [165, 75], [170, 80]     # 과체중
  ])
  # 목표 데이터 (0: 보통, 1: 과체중)
  y = np.array([0, 0, 0, 1, 1, 1])

  # 데이터 세트를 훈련용 70%, 테스트용(검증) 30%로 분리
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

  # 모델 생성 -> KNN 분류 모델
  model = KNeighborsClassifier(n_neighbors=3)
  # => 가장 가까운 이웃 3명의 라벨(0 또는 1) 중 다수결로 분류

  # 학습
  model.fit(x_train, y_train)

  # 예측
  test_data = np.array([[175, 78]]) # 샘플 데이터 [[키, 몸무게]]
  prediction = model.predict(test_data)

  print(type(prediction), prediction)

  labels = ['보통', '과체중']

  print(f'입력한 정보 {test_data} 의 결과는 {labels[prediction[0]]} 입니다.')

  # 성능 평가
  # - 회귀(Regression) : R2(결정계수) -> 적합도를 측정하는 지표
  # - 분류(Classification) : 정확도를 반환 (측정 지표)
  accuracy = model.score(x_test, y_test)
  print(f' 현재 모델의 정확도 : {accuracy}')

  # 이웃 데이터 확인
  distance, indices = model.kneighbors(test_data)

  print('* --- 이웃 데이터 --- *')
  print(distance)
  print(x_train[indices])
  print(y_train[indices])

# run_knnc()
