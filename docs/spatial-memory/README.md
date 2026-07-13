# SuperMemory-VQA Spatial Memory

| 항목 | 값 |
|---|---|
| Page ID | SM-ROOT |
| Confluence parent | SM-DOCS |
| 상태 | 로컬 준비 진행 |
| 최종 갱신 | 2026-07-13 |
| 목표 | Geometry-grounded QA용 explicit compressed spatial memory |

## 핵심 결론

| 항목 | 결정 |
|---|---|
| 기술 방향 | Explicit typed spatial record를 유지하고 dense geometry는 transient하게 사용 |
| 현재 준비도 | 로컬 contract와 heuristic baseline 준비 완료 |
| 공식 보고 | **No-Go**: learned path end-to-end 미실행 |
| 다음 승인 대상 | 명시적 승인 후 1-node × 1-GPU contract probe |
| Scale-up 조건 | Probe가 checkpoint-to-evidence lineage를 닫고 budget 내 valid typed record 생성 |

지금은 latent codec이나 더 큰 model architecture를 추가하지 않는다. 먼저 하나의
학습 checkpoint가 감사 가능하고 byte-bounded인 spatial evidence를 생성하며,
그 결과가 downstream QA에 반영되는지 입증한다.

## 근거

- Tiny-fixture source-compact memory는 diagnostic source representation의 216개
  record·96,456 JSONL byte 대신 15개 record·6,050 byte를 유지했다(15.94× 축소).
  이는 pipeline sanity이며 benchmark 근거가 아니다.
- Causal retrieval, actual-byte limit, typed schema, deterministic proof,
  four-choice QA, DDP training contract, production artifact lineage가 구현돼 있다.
- Production G-CUT3R extractor, raw RGB/IMU/VIO student encoder,
  repository-owned type-specific decoder, learned open-world association은 없다.
- 이 개발 host에서 실제 dataset, model download, training, evaluation, SSH,
  Slurm job을 실행하지 않았다.

## 즉시 실행 방향

1. 회사 compute에 pinned external teacher와 inference executable을 준비한다.
2. 고정 causal frame inventory로 승인된 contract probe를 실행한다.
3. Probe 통과 후 별도 full learned-E1 run 승인을 요청한다.
4. 공식 비교 전 matched E2/E3 identity를 추가한다.
5. Learned bridge가 valid해진 뒤 QA-versus-bytes를 측정한다.

## 시스템 범위

```text
1 Hz RGB + native-rate IMU/VIO + optional depth
    -> transient guided geometry
    -> typed record candidates
    -> value / actual-byte writer
    -> explicit persistent memory
    -> causal retrieval
    -> deterministic geometry proof
    -> four-choice QA or abstention
```

Persistent memory는 object, plane, portal, free space, landmark, event,
uncertainty, validity, provenance, evidence reference를 저장한다. Language model은
질문 해석과 답변 표현만 담당하며 metric fact를 생성하지 않는다.

## 의사결정 문서

| 질문 | 문서 |
|---|---|
| 문제와 성공 기준은 무엇인가? | [문제 정의와 연구 질문](problem.md) |
| 어떤 시스템과 control을 사용할 것인가? | [아키텍처](architecture.md) |
| 어떤 claim에 근거가 있는가? | [추적성](traceability.md) |
| 다음 실행은 무엇인가? | [연구 로드맵](roadmap.md) |
| 현재 go/no-go는 무엇인가? | [현재 상태](status.md) |
| 어떤 외부 근거가 중요한가? | [논문 근거 목록](papers/README.md) |
| 어떤 결정이 채택됐는가? | [아키텍처 결정](decisions/README.md) |
| 어떤 결과가 측정됐는가? | [실험](experiments/README.md) |
| 어떤 검토가 완료됐는가? | [날짜별 검토](reviews/README.md) |
| 회사 실행 절차는 어디에 있는가? | [운영](operations/README.md) |

## Confluence import 부록

| Source 범위 | Import Page ID | Parent ID |
|---|---|---|
| `docs/README.md` | `SM-DOCS` | `SPACE-HOME` |
| `docs/spatial-memory/README.md` | `SM-ROOT` | `SM-DOCS` |
| `{problem,architecture,traceability,roadmap,status}.md` | Page metadata | `SM-ROOT` |
| `source/README.md`와 non-template child | Page metadata | `SM-ROOT` / `SM-SOURCE` |
| `papers/README.md`와 non-template child | Page metadata | `SM-ROOT` / `SM-PAPERS` |
| `decisions/README.md`와 non-template child | Page metadata | `SM-ROOT` / `SM-DECISIONS` |
| `experiments/README.md`와 non-template child | Page metadata | `SM-ROOT` / `SM-EXPERIMENTS` |
| `reviews/README.md`와 non-template child | Page metadata | `SM-ROOT` / `SM-REVIEWS` |
| `operations/README.md` | `SM-OPERATIONS` | `SM-ROOT` |
| Repository `HANDOFF.md` | `SM-OPERATIONS-HANDOFF` | `SM-OPERATIONS` |

Template과 `docs/` 아래 레거시 문서 3개는 제외한다. Page ID, relative link,
완료된 experiment와 날짜별 review의 불변성을 유지한다.

[문서 목록으로 돌아가기](../README.md)
