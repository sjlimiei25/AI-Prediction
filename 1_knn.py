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

# KNN (K-Nearest Neighbor) : K-최근접 이웃 알고리즘
from sklearn.neighbors import KNeighborsRegressor   # 회귀
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
  X = [[1], [2], [3], [4], [5], [6], [7]]
  Y = [40, 45, 50, 55, 60, 65, 70]

  # ============ * 데이터 분리 * ==========================
  # 데이터셋 => 훈련용데이터셋 + 테스트용데이터셋
  # train_test_split : 데이터를 '훈련용'과 '테스트용(검증)'으로 나누는 함수. 모델이 공부하는 데이터와 실제 평가하는 데이터를 구분할 때 사용.
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
  # 입력데이터_훈련용데이터, 입력_테스트용데이터, 예측_훈련용데이터, 예측_테스트용데이터 = train_test_split(입력데이터, 예측데이터, test_size=테스트용비율(0~1),random_state=데이터셔플비율)

  # ============ * 학습 * =================================
  # 가장 가까운 5명의 평균을 이용해 점수를 예측한다.
  # => k=5
  model = KNeighborsRegressor(n_neighbors=2)      # 모델 생성

  # fit() : 모델을 학습시키는 함수. 훈련용 데이터를 사용해 학습.
  model.fit(X_train, Y_train)                     # 학습

  # 예측
  test_data = [[5.5]] # 공부시간
  prediction = model.predict(test_data)   # 5.5를 전달해서 결과 예측

  print(f'{test_data} 시간 공부 시.. 예측 점수 : {prediction}')

  # 성능 평가 => score(입력_테스트데이터, 예측_테스트데이터)
  r2 = model.score(X_test, Y_test)

  print(f'결정 계수(성능 평가) : {r2:.2f}')
  # => 1에 가까울수록 좋은 모델을 의미함!

run_knnr()