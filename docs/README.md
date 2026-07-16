# 문서

| 항목 | 값 |
|---|---|
| Page ID | SM-DOCS |
| Confluence parent | SPACE-HOME |
| 문서 역할 | 임원 보고용 문서 목록 |
| 상태 | 활성 |

## 핵심 요약

- **제안 framing:** 제품 단위를 QA 모델이 아닌 QA·Visual Routine이 공유하는
  온디바이스 멀티모달 메모리 런타임으로 둔다.
- **현재 증거:** 로컬 spatial contract와 heuristic baseline은 준비됐지만,
  lifecycle·routine·실기기와 공식 learned-method 결과는 없다.
- **즉시 실행:** [현재 상태](spatial-memory/status.md)의 sensor audit와
  teacher-oracle gate부터 수행한다. 해당 gate 통과 전 benchmark 성능을 주장하지
  않는다.
- **Source of truth:** 아래 canonical Spatial Memory 문서만 사용한다. 레거시
  문서는 안내 용도로만 유지한다.

## 의사결정 문서

| 확인 목적 | 시작 문서 |
|---|---|
| 임원 제안과 승인 요청 | [온디바이스 메모리 런타임 제안](spatial-memory/on-device-memory-runtime-proposal.md) |
| 공통 write/read runtime과 QA·Visual Routine 구조 | [멀티모달 메모리 런타임 아키텍처](spatial-memory/memory-runtime-architecture.md) |
| 현재 go/no-go와 다음 gate | [현재 상태](spatial-memory/status.md) |
| Spatial component와 geometry control | [Spatial 아키텍처](spatial-memory/architecture.md) |
| 실행 순서 | [연구 로드맵](spatial-memory/roadmap.md) |
| 결정별 근거 | [추적성](spatial-memory/traceability.md) |
| 측정 결과 | [실험](spatial-memory/experiments/README.md) |
| 회사 compute 실행 | [Company-compute handoff](../HANDOFF.md) |

현재 구현 프로젝트 홈: [SuperMemory-VQA Spatial Memory](spatial-memory/README.md).

## 관리 원칙

- 각 canonical Markdown 파일은 Confluence page 하나에 대응하며 H1은 하나만 둔다.
- `SPACE-HOME`만 external parent sentinel로 사용하고, 나머지는 stable `SM-*`
  Page ID를 사용한다.
- 논문 근거, 프로젝트 추론, 프로젝트 결과를 분리한다.
- Metadata는 Markdown table로 작성하고, import 전까지 relative link를 유지한다.
- PDF, dataset, model, checkpoint, 대형 artifact는 Git에 저장하지 않는다.

## 레거시 문서

- `spatial-token-compression.md`
- `spatial-token-research-roadmap.md`
- `implementation-review.md`

새 결정과 결과는 `docs/spatial-memory/` 아래에 기록한다.
