# EXP-XXXX: 실험 제목

| 항목 | 값 |
| --- | --- |
| Page ID | SM-EXP-XXXX |
| Experiment ID | EXP-XXXX |
| Confluence parent | SM-EXPERIMENTS |
| 상태 | 계획 |
| 근거 수준 | 미실행 |
| 최종 검토 | YYYY-MM-DD |
| 대체 대상 | 없음 |

## 핵심 결론

**대기 / Go / No-Go / Invalid.** 현재 결론과 근거 수준을 한 문장으로 적는다.

## 다음 결정

다음 실행 또는 ADR 변경을 한 문장으로 적는다. 결과가 `Invalid`이면 설계 결론을
내리지 않는다.

## 근거

미실행.

실행 후 variant별 수치, confidence interval 또는 반복 run 분산, 실패한 invariant를
기록한다. local mock 수치는 `Local sanity only; not benchmark`로 표시한다.

## 의사결정 gate

| Metric 또는 invariant | Scale | Go 조건 |
| --- | --- | --- |
| QA-Acc | 0-100 | Run 전 정의 |
| QA-MRR | 0-100 | Run 전 정의 |
| Ans-F1 | 0-100 | Run 전 정의 |
| Actual serialized bytes | bytes | Run 전 budget과 scope 정의 |
| Causal violations | count | 0 |

## 비교안

| Variant | 변경 요소 | 고정 input |
| --- | --- | --- |
| A | Baseline | 목록 작성 |
| B | 제안 변경 | 목록 작성 |

## 가설

검증 가능한 한 문장으로 작성한다. 어떤 입력과 budget에서 무엇보다 좋아야 하는지
명시한다.

## 실행 contract

실행 전에 아래 값을 확정한다. 확정하지 못한 값은 `TBD`로 두며 해당 상태에서는
benchmark를 시작하지 않는다.

| 항목 | 고정값 |
| --- | --- |
| Code revision | TBD |
| Dataset과 split | TBD |
| Dataset, source, question, label digest | TBD |
| Sensor-frame manifest digest | TBD |
| Model과 checkpoint digest | TBD |
| Config와 config digest | TBD |
| Random seed | TBD |
| Byte-budget scope와 값 | TBD |
| QA backend와 prompt schema | TBD |
| Run ID | TBD |

## 추적성

| 유형 | Link | 관련성 |
| --- | --- | --- |
| Claim | Claim link | 해결하려는 문제 |
| Decision | ADR link | 검증하거나 변경할 설계 결정 |
| Paper | Paper page link | 논문 근거; 논문 결과와 project result는 분리 |

## 실행 provenance

| 항목 | 값 |
| --- | --- |
| Run ID | 미할당 |
| Code revision | 미기록 |
| Slurm job ID 또는 process reference | 없음 |
| Company artifact path | 없음 |
| Metrics artifact | 없음 |
| Log | 없음 |
| 로컬 복사 | 없음 |
