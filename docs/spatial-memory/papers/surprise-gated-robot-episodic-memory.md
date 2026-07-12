# Worth Remembering: Surprise-Gated Robot Episodic Memory

| Field | Value |
|---|---|
| Page ID | SM-PAPER-SURPRISE-EPISODIC-MEMORY |
| Status | Reviewed; recent preprint; official code not published |
| Publication | arXiv:2606.03787 v3, 2026-06-06 |
| Primary source | [Version-pinned arXiv v3](https://arxiv.org/abs/2606.03787v3) |
| Official code | Not published or verified |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-004 |

## 30-second summary

Worth Remembering는 V-JEPA-2 latent space의 예측 놀라움을 사용하여 향후 질문을 알지 못한 채 희박한 시각적 에피소드를 선택한다. 동일한 에피소드 예산에서는 장거리 로봇 QA의 균일하고 무작위 쓰기를 능가한다. 이는 안정적인 지오메트리 코어를 교체하지 않고 작은 깜짝 게이트 evidence reservoir를 지원한다.

## Problem addressed

로봇은 모든 프레임을 유지할 수 없지만 향후 작업은 알 수 없다. 고정 속도 또는 작업별 쓰기는 짧고 중요한 이벤트를 놓치고 장기간의 중복 관찰을 유지할 수 있다. 이 논문에서는 어떤 이벤트를 일시적으로 저장할 가치가 있는지 결정하기 위한 인과적이고 감독되지 않은 신호를 찾고 있다.

## Relevant method

- V-JEPA-2를 사용하여 최근 프레임 64개의 인과 슬라이딩 윈도우를 삽입하고 1,024차원 잠재 데이터를 풀링한다.
- 이전 잠재 창에 대각선 가우스를 맞추고 강력한 정규화된 예측 오류 점수를 계산한다.
- `median + 1.4826 × MAD` 이상의 로컬 최대값에서 트리거한 다음 비최대값 억제를 적용한다.
- 타임스탬프, 로봇 포즈, 깜짝 점수, 8프레임 에피소드 및 검색 embedding을 저장한다.
- 선택한 에피소드를 DAAAM의 4D scene graph 위에 시각적 레이어로 추가한다.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| OC-NaVQA | DAAAM → surprise episodic memory | QA accuracy | 0.711 → 0.796 | Tables 1–2 |
| OC-NaVQA | DAAAM → surprise episodic memory | Position error | 41.75 m → 36.57 m | Tables 1–2 |
| OC-NaVQA | DAAAM → surprise episodic memory | Temporal error | 1.792 min → 1.510 min | Tables 1–2 |
| OC-NaVQA | Surprise gate | Retention rate | 1.28 episodes/min; about 1.7% of frames | Section 4.1 |
| OC-NaVQA | Surprise gate | Reasoning-token cost | About 13% increase | Section 4.1 |
| Kinetics-GEBD | Online unsupervised surprise gate | Mean F1 | 0.833 | Table 3 |

균일하고 무작위적인 에피소드 메모리 기준선은 QA 절제에서 동일한 에피소드 예산을 받는다. 이 논문에서는 DAAAM에 비해 정확도가 12.0%, 위치 오류가 12.4%, 시간 오류가 15.7% 향상되었다고 보고한다.

## What this supports here

- Surprise는 드문 이벤트에 대한 그럴듯한 쿼리 독립적 쓰기 신호이다.
- 작은 시각적 evidence reservoir는 구조화된 spatial memory를 보완할 수 있다.
- 포즈와 타임스탬프는 저장된 에피소드와 함께 제공되어야 한다.
- 균등한 선택과 무작위 선택에 대한 동일 예산 비교가 필요하다.

## What it does not prove

- 놀라움은 정적 기하학 범위를 대체할 수 있다. 중요한 벽, 문, 여유 공간은 놀랄 일이 아닐 수도 있다.
- 시각적 에피소드는 입력된 레코드에 비해 바이트 효율적이다.
- 반복되는 놀라운 주제에 대한 견고성; 이 논문은 습관화 누락을 한계로 식별한다.
- 정확한 1Hz 감지, 기기 내 실행 또는 SuperMemory-VQA 성능.

## Project reproduction status

재현되지 않았다. 현재 저장소에는 선택기 훈련에 놀라운 기능이 있지만 V-JEPA-2 게이트나 시각적 에피소드 저장소는 없다. 프로덕션 모델이 로컬로 다운로드되지 않았다.

## References

- Nicolas Gorlo, Derek K. Wise, Alberto Speranzon 및 Luca Carlone. [Worth Remembering: Surprise-Gated Robot Episodic Memory](https://arxiv.org/abs/2606.03787v3). arXiv:2606.03787 v3.
- [Back to paper index](README.md).
