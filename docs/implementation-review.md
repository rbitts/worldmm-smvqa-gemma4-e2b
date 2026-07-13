# SuperMemory-VQA Spatial Memory 구현 검토

| 항목 | 값 |
|---|---|
| Page ID | SM-IMPLEMENTATION-REVIEW |
| 문서 역할 | 과거 검토 요약 |
| 검토일 | 2026-07-11 |
| 현재 source of truth | [현재 상태](spatial-memory/status.md) |
| 고정 검토 기록 | [2026-07-11 로컬 준비 상태](spatial-memory/reviews/2026-07-11-local-readiness.md) |

## 핵심 결론

로컬 준비는 완료됐지만 learned end-to-end reproduction은 완료되지 않았다.
Heuristic explicit-memory 경로는 tiny fixture에서 동작했으나, 학습된 student가
QA에 사용되는 persistent evidence를 생성하지 못했다.

## 검토 시점 근거

- Causal source inventory, explicit memory, actual-byte limit, retrieval,
  deterministic proof, four-choice QA, 0–100 metric이 로컬에서 동작했다.
- Typed schema, external teacher/cache contract, DDP candidate-head training,
  hard typed-record validation이 구현돼 있었다.
- 341개 test와 Ruff, basedpyright가 통과했고 tiny QA의 causal violation은 0이었다.
- 실제 model download, training, benchmark evaluation, SSH, Slurm 작업은 없었다.

## 차단 결론

Pinned checkpoint가 canonical typed record를 내부 생성하고, 해당 record가
actual-byte·grounding contract를 충족하며, QA evidence/proof가 checkpoint에
transitive하게 연결되기 전까지 learned-method 또는 공식 benchmark 결과를
보고하지 않는다.

검토 당시 미완료 항목은 production G-CUT3R extraction, raw sensor encoding,
type-specific inference decode, open-world association, 배포 write gate의 QA
supervision, matched 공식 ablation이었다.

이 레거시 문서는 갱신하지 않는다. 당시 근거는 고정 검토 기록, 현재 go/no-go는
[현재 상태](spatial-memory/status.md)를 사용한다.
