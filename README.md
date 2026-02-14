# P2P 대출의 부도 시점 예측을 위한 그래프 어텐션 기반 생존분석 모델
## Graph Attention Neural Networks for Survival Analysis of Default in P2P Lending

> **학위**: 석사학위논문 (중앙대학교 대학원 통계데이터사이언스학과)  
> **저자**: 김민석 | **지도교수**: 이주영  
> **사용 언어**: Python 3.8.10 (PyTorch, PyTorch Geometric)

---

## 1. 연구 배경 및 목적

### 배경
- **P2P 대출(Peer-to-Peer Lending)**은 전통 금융시장 접근이 제한된 개인에게 중요한 자금 조달 수단
- 높은 부도율은 플랫폼 지속 가능성과 투자자 보호를 위협하는 핵심 요인
- 기존 생존분석 모형(CoxPH, RSF, CoxMLP)은 차입자를 **독립적 관측치로 가정** → 차입자 간 유사성/관계 구조를 반영하지 못함

### 목적
- **Graph Attention Network(GAT)**와 **Cox 부분우도(Partial Likelihood)**를 결합한 **CoxGAT** 모델 제안
- 유사한 금융 프로필을 가진 차입자 간의 **관계적 구조(homophily)**를 활용하여 부도 시점 예측 성능 향상
- CoxPH, RSF, CoxMLP와의 비교를 통해 제안 모델의 우수성 검증

---

## 2. 핵심 개념 정리

### 2.1 기존 생존분석 모형 비교

| 모형 | 특징 | 한계 |
|------|------|------|
| **CoxPH** | 준모수, 선형 로그-위험 구조, 해석 용이 | 비선형 상호작용 포착 불가 |
| **RSF** | 앙상블 기반, 비선형/교호작용 포착 가능 | 차입자 간 관계 구조 미반영 |
| **CoxMLP** | 심층 신경망으로 Cox 선형 예측자 대체 | 차입자 간 관계 구조 미반영 |
| **CoxGAT (제안)** | 그래프 어텐션 + Cox 부분우도 결합 | 관계 구조 + 비선형성 동시 반영 |

### 2.2 왜 그래프 기반 모델이 필요한가?
- CoxPH의 t-SNE 시각화 결과: Fully Paid와 Unpaid 그룹이 **완전히 혼재** → 선형 모형의 판별 한계
- 신용 리스크는 개인 속성뿐 아니라 **유사한 차입자 집단의 맥락 정보**에 의해 영향
- 재무적으로 위험한 이웃에 둘러싸인 차입자는 더 높은 위험 추정치를 받아야 함 → **위험 전파(risk propagation)** 개념

---

## 3. 제안 모델: CoxGAT 아키텍처

### 3.1 전체 파이프라인

```
Feature Matrix (Tabular Data, 60 features)
    ↓
Feature Engineering → Node Features
    ↓
Graph Construction (Mutual KNN, k=15) → Edge Index (Graph Structure)
    ↓
Input Linear (60 → 128)
    ↓
GAT Block: LayerNorm → GATConv → Multi-Head Attention (M=3)
    ↓
Final Layer: LayerNorm → Linear (128 → 1)
    ↓
Output: Risk Score (r_i)
    ↓
Cox Loss + C-index Evaluation
```

### 3.2 Graph Construction (그래프 구축)

| 항목 | 내용 |
|------|------|
| **노드(Node)** | 각 차입자 |
| **엣지(Edge)** | 특징 공간에서의 유사성 기반 연결 |
| **방법** | K-Nearest Neighbors (KNN) |
| **거리 척도** | 유클리드 거리 (연속형 변수 표준화 후) |
| **k 선택** | k ∈ {5, 10, 15, 20} 실험 → **k=15** 최적 (검증 C-index 기준) |
| **그래프 유형** | **Mutual KNN** (양방향으로 이웃인 경우만 엣지 유지) → 안정적이고 대칭적 구조 |

### 3.3 Attention Mechanism (어텐션 메커니즘)

**비정규화 어텐션 계수:**
```
e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )
```

**정규화 (Softmax):**
```
α_ij = exp(e_ij) / Σ_{k∈N_i} exp(e_ik)
```

**노드 임베딩 업데이트:**
```
h'_i = σ( Σ_{j∈N_i} α_ij · W·h_j )
```

**Multi-Head Attention (M=3):**
```
h'_p = CONCAT_{m=1}^{M} σ( Σ_{s∈N_p} α_ps^(m) · W^(m) · h_s )
```

- 3개의 독립적 어텐션 헤드가 이웃 정보의 **서로 다른 관점**을 학습
- 결과를 **concatenation**하여 더 풍부한 임베딩 생성
- 활성화 함수: **GeLU** (ReLU의 dead neuron 문제 해결, 음수값 정보 보존)

### 3.4 Optimization (최적화)

**Cox 부분우도 손실함수:**
```
L(θ) = - Σ_{i:δ_i=1} [ r_i - log Σ_{k∈R(T_i)} exp(r_k) ]
```

| 하이퍼파라미터 | 설정값 |
|----------------|--------|
| 옵티마이저 | AdamW |
| 초기 학습률 | 5 × 10⁻⁴ |
| Weight Decay | 10⁻³ |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=8) |
| 최대 에폭 | 300 |
| Early Stopping | 검증 C-index 30 에폭 미개선 시 |
| Gradient Clipping | ℓ₂-norm ≤ 1.0 |

---

## 4. 데이터

### 4.1 데이터 출처
- **LendingClub** 데이터셋 (2007–2018)
- 고용 직함이 1,000건 미만인 차입자 제외 후 **210,482명** 분석
- Fully Paid: **185,159명** | Default(Unpaid): **25,323명**
- Train:Test = **7:3** 분할

### 4.2 변수 구성

| 구분 | 변수 예시 |
|------|----------|
| **인구통계** | 고용 직함, 주거 유형, 지역 |
| **재무 속성** | 대출금액, 연소득, 이자율, DTI |
| **신용 이력** | FICO 점수, 연체 이력, 회전 신용 이용률 |
| **계약 정보** | 발행일, 신용등급(Grade), 대출 목적 |

### 4.3 기저 특성 요약 (주요 변수)

| 특성 | Fully Paid (N=185,159) | Unpaid (N=25,323) | P value |
|------|:---:|:---:|:---:|
| 대출금액 (USD) | 15,824 ± 9,311 | 16,782 ± 8,905 | <0.001 |
| 이자율 (%) | 12.19 ± 4.65 | **15.20 ± 4.98** | <0.001 |
| 연소득 (USD) | **84,798** ± 264,640 | 77,703 ± 70,626 | <0.001 |
| DTI (%) | 18.75 ± 12.66 | **20.45 ± 9.39** | <0.001 |
| FICO (하한) | **700.37** ± 33.81 | 687.87 ± 25.94 | <0.001 |
| Grade A 비율 | 23.37% | **6.17%** | <0.001 |
| Grade D–G 비율 | 17.89% | **41.55%** | <0.001 |

→ 부도 그룹: 높은 이자율, 높은 DTI, 낮은 FICO, 고위험 등급 집중

---

## 5. 주요 결과

### 5.1 모델 성능 비교 (C-index)

| 모델 | C-index |
|------|:---:|
| CoxPH | 0.550 |
| RSF | 0.688 |
| CoxMLP | 0.693 |
| **CoxGAT (제안)** | **0.705** |

→ CoxGAT이 모든 비교 모형 대비 **최고 성능** 달성

### 5.2 성능 향상 요인 분석

| 비교 | 성능 차이 | 해석 |
|------|:---:|------|
| CoxPH → RSF | +0.138 | 비선형성 포착의 효과 |
| RSF → CoxMLP | +0.005 | 심층 특징 변환의 미미한 추가 효과 |
| CoxMLP → CoxGAT | **+0.012** | **관계적 구조(그래프) 반영의 효과** |
| CoxPH → CoxGAT | **+0.155** | 비선형성 + 관계 구조의 복합 효과 |

### 5.3 Permutation Importance (변수 중요도)

| 변수 | ΔC (C-index 감소량) |
|------|:---:|
| **연소득 (Annual Income)** | **0.053** |
| FICO 점수 (하한) | 0.015 |
| FICO 점수 (상한) | 0.009 |
| 최근 6개월 신용 조회 수 | 0.005 |

→ **연소득**이 부도 위험 예측에 가장 핵심적 변수 (소득 안정성 = 장기 상환 능력의 근본 동인)  
→ FICO 점수 등 **장기 신용 지표**가 단기 활동 지표보다 중요

### 5.4 t-SNE 시각화 비교

| CoxPH 임베딩 | CoxGAT 임베딩 |
|:---:|:---:|
| Fully Paid / Unpaid **완전 혼재** | Fully Paid / Unpaid **명확 분리** |
| → 선형 모형의 판별력 한계 | → 그래프 어텐션의 관계 구조 학습 효과 |

- CoxGAT: 고위험 차입자끼리 군집, 저위험 차입자끼리 군집 → **이웃 기반 위험 전파** 반영
- 유사한 생존 결과를 가진 차입자들이 임베딩 공간에서 일관된 클러스터 형성

---

## 6. 연구 의의 및 한계

### 의의
- **그래프 기반 생존분석**을 P2P 대출 부도 예측에 최초 적용
- 금융 리스크가 개인 특성뿐 아니라 **차입자 간 관계 구조(homophily)**에 의해 영향받음을 실증
- Attention 메커니즘이 관련성 높은 이웃 정보를 선택적으로 강조하여 **노이즈 완화 + 해석성 향상**

### 한계
- 그래프가 대출 시점의 **정적(static) KNN**으로 구축 → 시간에 따른 차입자 특성 변화 미반영
- **시간적 그래프 동태(temporal graph dynamics)** 미고려 (거시경제 변화, 행동 변화 등)
- Mutual KNN이 특징 공간에서 보이지 않는 **장거리 의존성(long-range dependency)** 포착에 한계

### 향후 연구 방향
- 학습 가능한(adaptive) 그래프 구조 또는 **Temporal Graph Neural Networks** 적용
- 거시경제 지표 통합
- 불확실성 정량화(Uncertainty Quantification) 또는 반사실적 분석(Counterfactual Analysis) 확장

---

## 7. 핵심 키워드 정리

| 키워드 | 설명 |
|--------|------|
| **CoxGAT** | Graph Attention Network + Cox Partial Likelihood를 결합한 생존분석 모델 |
| **GAT (Graph Attention Network)** | 이웃 노드에 어텐션 가중치를 부여하여 메시지를 집계하는 GNN |
| **Cox Partial Likelihood** | 기저 위험 함수를 명시하지 않고 위험비를 추정하는 준모수적 손실함수 |
| **C-index** | 사건 순서를 올바르게 순위화하는 능력을 측정하는 일치 지수 |
| **KNN Graph** | K-최근접 이웃 알고리즘으로 유사 차입자 간 엣지를 생성하는 그래프 구축 방법 |
| **Mutual KNN** | 양방향으로 이웃인 경우만 엣지를 유지하여 안정적 그래프 구조 생성 |
| **Multi-Head Attention** | 여러 독립적 어텐션 헤드로 이웃 정보의 다양한 관점을 동시 학습 |
| **Homophily** | 유사한 특성을 가진 노드가 연결되는 경향; 차입자 군집 형성의 근거 |
| **Permutation Importance** | 특정 변수를 셔플하여 C-index 감소량으로 변수 중요도를 측정 |
| **GeLU** | Gaussian Error Linear Unit; 입력을 확률적으로 게이팅하는 매끄러운 활성화 함수 |
| **RSF** | Random Survival Forest; 부트스트랩 앙상블로 비선형 생존 관계를 포착 |
| **t-SNE** | 고차원 임베딩을 저차원으로 시각화하여 군집 구조를 확인하는 기법 |
