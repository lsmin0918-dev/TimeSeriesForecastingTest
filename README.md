# 📈 비트코인 알고리즘 트레이딩 과제 보고서 (Final Report)

**작성자**: juho@hufs.ac.kr (한국외국어대학교 GBT + Business & AI)

---

## 1. 프로젝트 개요 (Project Overview)

본 프로젝트의 목표는 딥러닝(Deep Learning) 모델을 활용하여 비트코인(BTC)의 가격 등락을 예측하고, 이를 기반으로 **Buy and Hold 벤치마크를 상회하는 수익률**을 달성할 수 있는 알고리즘 트레이딩 전략을 구축하는 것입니다.

단순한 예측 모델을 넘어, 실제 시장에서의 거래 비용(수수료)과 리스크를 고려한 **"Aggressive Hybrid Strategy"**를 제안하며, 기술적 지표(RSI, MACD)와 모델의 예측 확률을 결합하여 최적의 매매 타이밍을 포착하고자 했습니다.

---

## 2. 모델 설계 및 학습 (Model Design & Training)

### 2.1 모델 아키텍처: RobustGRU

금융 시계열 데이터의 높은 변동성과 노이즈에 대응하기 위해, LSTM보다 구조가 단순하여 일반화(Generalization) 성능이 뛰어난 **GRU(Gated Recurrent Unit)**를 기반으로 모델을 설계했습니다. 과적합(Overfitting)을 방지하기 위해 정규화 기법을 적극 도입했습니다.

#### **모델 구조도**
```plaintext
[Input Features] (Sequence Length: 20)
       ⬇
[GRU Layer 1] 
  - Hidden Size: 128
  - Batch Normalization (LayerNorm 효과)
  - Dropout: 0.3
       ⬇
[GRU Layer 2] 
  - Hidden Size: 64
  - Batch Normalization
  - Dropout: 0.3
       ⬇
[Fully Connected Layer]
  - ReLU Activation
       ⬇
[Output Layer] (Linear -> Softmax)
  - Classes: 3 (매도, 관망, 매수)
```

### 2.2 하이퍼파라미터 및 학습 설정
*   **Optimizer**: Adam (Learning Rate: 0.001)
*   **Loss Function**: CrossEntropyLoss (클래스 불균형 고려)
*   **Regularization**: Dropout (0.3) 및 Batch Normalization 적용
*   **Sequence Length**: 20일 (과거 20일 데이터를 보고 다음 날 예측)

---

## 3. 투자 전략 상세 (Investment Strategy: Aggressive Hybrid)

기존의 보수적인 모델들이 예측 확신도가 낮아 **"거래를 거의 하지 않는(Zero Minimum Trades)" 문제**를 해결하기 위해, 딥러닝 모델의 예측값과 전통적인 기술적 분석 지표를 결합한 **하이브리드 전략**을 설계했습니다.

### 3.1 전략 핵심 로직 (Logic Breakdown)

이 전략은 `'assignment_notebook.ipynb'`의 `simulate_custom_strategy` 함수에 구현되어 있으며, 주요 로직은 다음과 같습니다:

#### **① 진입 장벽 완화 (Lower Threshold)**
*   기존 모델은 60~70% 이상의 확신이 있어야 매수했으나, 본 전략은 **50% (probability > 0.5)** 이상의 확률만 나와도 즉시 분할 매수를 검토합니다.
*   **목적**: 상승 추세 초기를 놓치지 않고 적극적으로 포지션을 확보하기 위함.

#### **② RSI 역추세 매매 (Rebound Trading)**
*   **조건**: `RSI(14) < 30` (과매도 구간)
*   **행동**: 모델이 하락을 예측하더라도, **강제로 매수 비중을 확대(target_ratio += 0.2)**합니다.
*   **근거**: 비트코인 시장은 공포(Panic Sell)에 의한 급락 후 기술적 반등(Dead Cat Bounce)이 강하게 나타나는 경향이 있습니다. 이를 수익 기회로 삼습니다.

#### **③ MACD 추세 추종 (Trend Boosting)**
*   **조건**: `MACD > Signal` (골든크로스 상태)
*   **행동**: 목표 투자 비중을 **20% 상향(×1.2)** 조절합니다.
*   **근거**: 상승 추세가 확인된 구간에서는 레버리지를 태우듯 공격적으로 비중을 늘려 수익을 극대화합니다.

#### **④ 변동성 리스크 관리 (Risk Management)**
*   **조건**: 변동성(`Volatility_10`)이 상위 10% 이내인 초고변동성 구간
*   **행동**: 목표 비중을 **50% 축소**합니다.
*   **근거**: 예측 불가능한 급등락 장세에서는 현금 비중을 늘려 자산을 방어합니다.

---

## 4. 결과 분석 (Result Analysis)

### 4.1 성과 비교 (Performance Comparison)

| 구분 | Buy and Hold (벤치마크) | Aggressive Hybrid (제안 전략) |
| :--- | :---: | :---: |
| **전략 특성** | 단순히 매수 후 보유 | 시장 상황에 따라 비중 조절 (Active) |
| **하락장 성과** | 📉 시장 하락폭 그대로 손실 반영 | 🛡️ 현금 비중 확대로 손실 방어 |
| **횡보장 성과** | ➖ 변동 없음 | 📉 잦은 매매 수수료로 인한 소폭 손실 |
| **급반등장 성과** | 📈 시장 상승폭 반영 | 🚀 RSI 매수를 통해 바닥권 진입 성공 |

### 4.2 분석 및 고찰
1.  **"0 Trades" 문제 해결**:
    *   초기 모델은 높은 확신도를 요구하여 거래가 0건인 경우가 많았으나, 임계값을 0.5로 낮추고 보조 지표를 활용함으로써 **활발한 거래(Active Trading)**를 유도했습니다.
2.  **리스크 대비 수익률 (Risk-Adjusted Return)**:
    *   벤치마크는 하락장에서 무기력하게 손실을 입었으나, 제안 전략은 변동성 필터와 RSI 과열(70 이상) 시 매도 로직을 통해 **MDD(Maximum Drawdown)를 효과적으로 방어**했습니다.
3.  **한계점 (Limitation)**:
    *   **Whipsaw(휩소) 현상**: 뚜렷한 추세가 없는 박스권 장세에서는 잦은 매수/매도 시그널로 인해 수수료(0.1%)가 누적되어 수익률을 갉아먹는 현상이 관찰되었습니다.

---

## 5. 결론 (Conclusion)

본 프로젝트를 통해 **"설명 가능한 AI 트레이딩"**의 가능성을 확인했습니다. 단순히 모델의 'Black Box' 예측값에만 의존하는 것이 아니라, 검증된 기술적 지표(RSI, MACD)를 결합했을 때, 하락장 방어와 이익 실현이라는 두 마리 토끼를 잡을 수 있었습니다.

향후에는 **1시간 봉 데이터**를 활용한 정밀 타점 분석이나 **온체인 데이터(On-chain Data)**를 입력 변수로 추가한다면 횡보장에서의 손실을 줄이고 승률을 더 높일 수 있을 것입니다.

---
*© 2025 TimeSeriesForecastingTest Project*
