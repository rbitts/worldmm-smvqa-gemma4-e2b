# ADR-0002: Use G-CUT3R as an external geometry teacher

| Metadata | Value |
|---|---|
| Page ID | SM-ADR-0002 |
| ADR | ADR-0002 |
| Project claims | C-001, C-006 |
| Status | Accepted design; implementation blocked |
| Date | 2026-07-11 |
| Traceability | [Sparse 1 Hz geometry requirement](../traceability.md) |
| Architecture | [Teacher and student boundary](../architecture.md) |

## Context and problem

AI-glass 입력은 약 1 Hz라 연속 frame overlap이 낮을 수 있다. 영구 typed record를
만들려면 frame-level pose, geometry, uncertainty, association supervision이
필요하지만 글래스용 student를 처음부터 ground truth만으로 학습할 준비는 되어
있지 않다. 큰 geometry model을 온디바이스 persistent memory로 직접 운영하거나
그 hidden state를 장기 저장하는 것도 목표가 아니다.

## Decision

G-CUT3R-compatible model을 회사 환경에서 실행하는 외부 offline teacher로
사용한다. Teacher에는 1 Hz RGB observation과 가능한 pose/depth guidance를 주고,
출력은 causal prefix와 digest에 묶인 typed-record cache로 materialize한다.
Repository는 provider protocol과 cache contract를 소유하지만 G-CUT3R code나
checkpoint를 자동 설치, 다운로드, 포함하지 않는다.

Teacher state는 다음 observation을 위한 transient state다. 영구 memory는
ADR-0001의 typed record이며 teacher state snapshot이 아니다.

## Evidence

- [CUT3R](../papers/cut3r.md)는 recurrent scene-state 기반 geometry prediction의
  관련 근거다.
- [G-CUT3R](../papers/g-cut3r.md)는 pose/depth guidance가 있는 sparse-view geometry
  teacher 선택의 직접적인 연구 근거다.
- `src/worldmm_smvqa/worldmm/gcut3r_teacher.py`는 external
  provider, ordered causal state, request/response/prefix digest 계약을 구현한다.
- `src/worldmm_smvqa/teacher_materializer.py`는 검증된
  cache record와 외부 supervision을 training row로 결합한다.
- [local readiness review](../reviews/2026-07-11-local-readiness.md)는 실제 extractor와
  checkpoint inference가 없는 상태를 P0 blocker로 기록한다.

## Alternatives considered

| Alternative | Why not selected |
|---|---|
| G-CUT3R를 온디바이스 최종 모델로 직접 사용 | 글래스 자원 조건을 입증하지 못했고 persistent typed DB 요구와 별개임 |
| CUT3R/G-CUT3R recurrent state를 장기 저장 | explicit entity와 temporal proof를 제공하지 않으며 snapshot 저장량이 누적됨 |
| Monocular per-frame detector만 사용 | 1 Hz wide-baseline에서 공통 metric frame과 association supervision이 부족함 |
| Repository가 모델을 자동 다운로드 | 로컬 개발 host 규칙과 회사 artifact 관리 경계를 위반함 |
| Teacher 없이 student를 바로 학습 | 현재 준비된 raw geometry supervision과 end-to-end encoder가 없음 |

## Implementation

- Provider는 한 observation과 직전 opaque state만 받는다. 동일 run에서 video ID는
  하나이며 timestamp는 strictly increasing이어야 한다.
- Cache는 request, response, provider, causal prefix digest를 보존하고 future
  validity를 거부한다.
- Materializer는 cache의 모든 teacher record가 supervision row와 정확히 대응하고
  train/validation group이 교차하지 않는지 검사한다.
- `src/worldmm_smvqa/spatial_train.py`는 materialized vector를
  입력받는 candidate head와 DDP training/checkpoint 골격을 제공한다.
- Blocked: 실제 G-CUT3R provider, raw RGB/IMU/VIO feature encoder, teacher
  pseudo-label 생성, type-specific target encoding, trained checkpoint inference.

## Verification

- `tests/test_gcut3r_teacher.py`는 provider contract,
  causality, cache digest와 fail-closed configuration을 검사한다.
- `tests/test_teacher_materializer.py`는 cache와
  supervision의 완전한 join 및 split invariants를 검사한다.
- `tests/test_spatial_train.py`는 tiny tensor 기반
  candidate head, loss, checkpoint path를 검사한다.
- 실제 G-CUT3R inference, model download, training, benchmark evaluation은 로컬에서
  수행하지 않았다. 따라서 이 ADR은 설계 채택 상태이며 재현 완료 상태가 아니다.

## Consequences

- 외부 model dependency와 repository-owned contract가 분리된다.
- Teacher 교체 시에도 typed cache contract가 유지되면 downstream training을 비교할
  수 있다.
- Provider ID, checkpoint, input manifest, cache digest를 experiment provenance에
  반드시 기록해야 한다.
- 회사 환경에서 provider 구현과 checkpoint 접근이 준비되지 않으면 learned lane은
  checkpoint 이전에서 중단된다.
- Teacher가 출력한 inferred geometry는 observed fact와 동일한 provenance로 승격하지
  않는다.

## Supersession

None.
