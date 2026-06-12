# Cat Classifying with Pre-trained Inception v3 & SVM
본 프로젝트는 사전 학습된 **Inception v3** CNN 모델을 사용하여 고양이 이미지의 품종(Tabby Cat, Tiger Cat)을 예측하고, 해당 예측 확률 값을 피처(Feature)로 활용하여 **SVM(Support Vector Machine)**으로 두 품종 간의 의사결정 경계(Decision Boundary)를 분류 및 시각화하는 파이토치(PyTorch) 및 머신러닝 실습 노트북입니다.

---

## 📌 주요 기능 및 프로젝트 흐름

### 1. 사전 학습된 이미지 분류 모델 로드
- PyTorch Hub를 사용하여 ImageNet 데이터셋으로 사전 학습된 **Inception v3** 모델을 불러옵니다.
- 모델 입력에 맞게 데이터를 정규화하기 위한 커스텀 PyTorch Module인 `Normalize` 클래스를 정의하고 모델 파이프라인의 첫 단계로 구성합니다. (ImageNet 평균: `[0.485, 0.456, 0.406]`, 표준편차: `[0.229, 0.224, 0.225]`)

### 2. 고양이 이미지 전처리 및 예측
- 입력 이미지를 Inception v3 입력 규격인 `299x299` 크기로 조정(`Resize` & `CenterCrop`)하고 텐서로 변환합니다.
- 특정 URL에서 고양이 이미지를 다운로드받아 모델에 입력한 후 소프트맥스(Softmax)를 거쳐 상위 5개의 ImageNet 클래스 및 예측 확률을 출력합니다.
  - ImageNet 클래스 인덱스 **281**: 얼룩고양이 (tabby, tabby cat)
  - ImageNet 클래스 인덱스 **282**: 호랑이고양이 (tiger cat)

### 3. 고양이 예측 데이터셋 구축
- 총 12장의 고양이 이미지(Tabby Cat 이미지 6장, Tiger Cat/Toyger 이미지 6장)를 활용하여 각 이미지별 Tabby Cat(281)과 Tiger Cat(282)의 예측 확률 데이터를 수집합니다.
- 수집된 데이터는 아래와 같은 2차원 피처 공간 상의 좌표로 표현됩니다.
  - **X축**: 얼룩고양이 확률 (Tabby Cat Probability [%])
  - **Y축**: 호랑이고양이 확률 (Tiger Cat Probability [%])
  - **Label**: 얼룩고양이(0), 호랑이고양이(1)

### 4. SVM(Support Vector Machine)을 이용한 선형 분류
- `scikit-learn` 라이브러리의 `svm.SVC` 모델을 사용하여 선형 커널(`kernel='linear'`) 기반의 SVM 분류기를 학습시킵니다.
- 두 클래스를 가장 잘 구분하는 최적의 결정 초평면(Decision Hyperplane)과 마진(Margin)을 계산합니다.

### 5. 결과 시각화
- `matplotlib`를 사용하여 2차원 좌표 평면에 고양이 데이터 포인트를 산점도(Scatter Plot)로 나타냅니다.
- 학습된 SVM의 결정 경계(실선)와 서포트 벡터의 마진 경계(점선)를 평면에 함께 시각화하여 분류 성능을 한눈에 확인합니다.

---

## 🛠️ 개발 환경 및 요구 사양

본 노트북을 실행하기 위해서는 아래의 라이브러리들이 설치되어 있어야 합니다.

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

- **Python** 3.7+
- **PyTorch** & **Torchvision** (CUDA 가속 사용 가능 권장)
- **NumPy**
- **Matplotlib**
- **Pillow (PIL)**
- **Scikit-Learn**

---

## 🚀 실행 방법

1. Jupyter Notebook 또는 Google Colab을 실행합니다.
2. `Cat_classifying.ipynb` 파일을 열어 모든 셀을 순서대로 실행합니다.
3. 이미지 URL 로드 시 네트워크 연결이 필요하므로 인터넷 환경에서 실행해 주세요.
4. 마지막 셀 실행 후 화면에 출력되는 SVM 의사결정 경계 시각화 그래프를 확인합니다.

---

## 📊 결과 그래프 예시
노트북이 정상적으로 실행되면, Tabby Cat과 Tiger Cat 예측 확률의 선형 관계와 이를 나누는 검은색 SVM 결정 초평면(Decision Boundary)이 시각화됩니다. 
- **Tabby Cat** 확률이 높은 영역(주로 우측 하단)과 **Tiger Cat** 확률이 높은 영역(주로 좌측 상단) 사이에 깔끔한 경계가 생성되는 것을 볼 수 있습니다.
