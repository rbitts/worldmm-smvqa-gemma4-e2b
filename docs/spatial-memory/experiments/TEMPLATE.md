# EXP-XXXX: Experiment title

| Metadata | Value |
| --- | --- |
| Page ID | SM-EXP-XXXX |
| Experiment ID | EXP-XXXX |
| Confluence parent | Spatial Memory / Experiments |
| Status | Planned |
| Evidence level | Not run |
| Last reviewed | YYYY-MM-DD |
| Supersedes | None |

## Hypothesis

검증 가능한 한 문장으로 작성한다. 어떤 입력과 budget에서 무엇보다 좋아야 하는지
명시한다.

## Linked claims, decisions, and papers

| Type | Link | Relevance |
| --- | --- | --- |
| Claim | Link to claim | 해결하려는 문제 |
| Decision | Link to ADR | 검증하거나 변경할 설계 결정 |
| Paper | Link to paper page | 논문 근거; 논문 결과와 우리 결과는 분리 |

## Fixed contract

실행 전에 아래 값을 확정한다. 확정하지 못한 값은 `TBD`로 두며 해당 상태에서는
benchmark를 시작하지 않는다.

| Item | Fixed value |
| --- | --- |
| Code revision | TBD |
| Dataset and split | TBD |
| Dataset, source, question, and label digests | TBD |
| Sensor-frame manifest digest | TBD |
| Model and checkpoint digest | TBD |
| Config and config digest | TBD |
| Random seed | TBD |
| Byte-budget scope and values | TBD |
| QA backend and prompt schema | TBD |
| Run ID | TBD |

## Compared variants

| Variant | Only changed factor | Inputs held constant |
| --- | --- | --- |
| A | Baseline | List |
| B | Proposed change | List |

## Metrics and go/no-go

| Metric or invariant | Scale | Go condition |
| --- | --- | --- |
| QA-Acc | 0-100 | Define before run |
| QA-MRR | 0-100 | Define before run |
| Ans-F1 | 0-100 | Define before run |
| Actual serialized bytes | bytes | Define budget and scope before run |
| Causal violations | count | 0 |

## Results

Not run.

실행 후 variant별 수치, confidence interval 또는 반복 run 분산, 실패한 invariant를
기록한다. local mock 수치는 `Local sanity only; not benchmark`로 표시한다.

## Run provenance

| Item | Value |
| --- | --- |
| Run ID | Not assigned |
| Code revision | Not recorded |
| Slurm job ID or process reference | None |
| Company artifact path | None |
| Metrics artifact | None |
| Logs | None |
| Copied locally | None |

## Conclusion

Pending.

## Decision impact

어느 ADR을 유지, 변경, 기각할지 실행 결과 이후 기록한다. 결과가 invalid이면 설계
결론을 내리지 않는다.
