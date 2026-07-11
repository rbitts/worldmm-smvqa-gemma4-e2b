# Spatial Memory Experiments

| Metadata | Value |
| --- | --- |
| Page ID | SM-EXPERIMENTS |
| Confluence parent | Spatial Memory |
| Page role | Experiment index and import contract |
| Status | Active |
| Last reviewed | 2026-07-11 |

이 페이지를 Confluence의 `Spatial Memory` 아래 `Experiments` 부모 페이지로
가져온다. 각 `EXP-*` 문서는 이 페이지의 직접 자식으로 둔다. 파일명과 H1의
실험 ID를 유지하면 Markdown 링크를 Confluence page link로 바꾸기 쉽다.

| ID | Experiment | Status | Evidence level | Result |
| --- | --- | --- | --- | --- |
| EXP-0001 | [Source-compact baseline](exp-0001-source-compact-baseline.md) | Local sanity completed | Tiny synthetic fixture and mock QA | Pipeline gate passed; not a benchmark |
| EXP-0002 | [Typed-memory bridge](exp-0002-typed-memory-bridge.md) | Planned | Contract checks only | Not run |
| EXP-0003 | [Byte Pareto](exp-0003-byte-pareto.md) | Planned | Design only | Not run |
| EXP-0004 | [G-CUT3R provider](exp-0004-gcut3r-provider.md) | Planned | Adapter contract checks only | Not run |

## Status vocabulary

| Status | Meaning |
| --- | --- |
| Planned | 계약은 작성됐지만 실행하지 않음 |
| Running | run ID와 실행 위치가 배정되고 작업이 진행 중 |
| Local sanity completed | tiny fixture 또는 mock으로 코드 경로만 검증 |
| Benchmark completed | 고정된 split, digest, checkpoint, run ID로 사내 평가 완료 |
| Invalid | leakage, digest mismatch, artifact 누락 등으로 결과 사용 금지 |
| Superseded | 새 실험이 결정을 대체함 |

## Result authority

- `Local sanity` 수치를 SuperMemory-VQA benchmark 결과로 인용하지 않는다.
- `Benchmark completed` 결과만 모델 간 성능 결론에 사용한다.
- 논문이 보고한 수치는 실험 결과가 아니다. 해당 [paper page](../papers/README.md)에만 기록한다.
- 완료된 benchmark 페이지는 수정해 덮어쓰지 않는다. 조건이 바뀌면 새 EXP ID를 만든다.
- planned 페이지에는 가짜 run ID, checkpoint hash, artifact path, metric을 넣지 않는다.

## Confluence import rules

- 페이지당 H1은 하나만 사용한다. H1은 고유한 EXP ID로 시작한다.
- YAML front matter, Mermaid, raw HTML, GitHub 전용 callout을 사용하지 않는다.
- metadata는 첫 번째 Markdown 표에 둔다.
- 내부 근거는 relative Markdown link로 연결한다. import 후 같은 제목의 Confluence
  page link로 치환한다.
- 표가 너무 넓어지면 새 표를 추가한다. 셀 안에 긴 로그나 JSON을 넣지 않는다.
- 대용량 artifact를 첨부하지 않는다. 승인된 사내 저장 경로와 digest만 기록한다.

새 실험은 [TEMPLATE](TEMPLATE.md)를 복사해 실제 실행 직전 생성한다. 먼 미래의
빈 페이지는 만들지 않는다.
