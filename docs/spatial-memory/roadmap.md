# 연구 로드맵

| 항목 | 값 |
|---|---|
| Page ID | SM-ROADMAP |
| 상태 | 활성 |
| 최종 갱신 | 2026-07-14 |
| 우선순위 원칙 | Student 투자 전에 teacher-oracle QA utility 검증 |

## 실행 방향

G-CUT3R는 offline teacher/oracle이다. 현재 P0는 prepared sensor signal을 audit하고
teacher object/place record의 causal, byte-matched QA ceiling을 측정하는 일이다.
Raw RGB student, all-type decoder, mobile claim은 이 gate 뒤에 둔다.

## 우선순위와 의사결정 gate

| 우선순위 | 목표 결과 | 다음 단계 gate |
|---|---|---|
| P0 | Sensor audit + causal teacher-oracle object/location ceiling | EXP-0004 provider valid, EXP-0005가 동일 byte E0보다 target slice 개선 |
| P1 | Minimal hybrid object/place student | Raw RGB semantics + native geometry가 oracle gain의 사전 정의 비율 유지 |
| P2 | Target-device 실행 가능성 | 실기기 latency, peak memory, energy, thermal, calibration 통과 |
| P3 | Revisit identity와 change memory | Existing/NEW false merge, duplicate, stale-location 목표 통과 |
| P4 | Matched official result와 byte Pareto | Immutable E1/E2/E3, QA-versus-bytes, causal violation 0 |
| P5 | 추가 spatial operator/type | Object/location 이후 held-out utility가 입증된 type만 추가 |

## P0 실행

```text
prepared sources
    -> RGB asset / intrinsics / native pose / depth coverage audit
    -> pinned offline G-CUT3R-compatible provider
    -> external semantic mask and place assignment
    -> selected points -> evidence-bound object/place records
    -> hard actual-byte writer -> retrieval -> deterministic proof -> QA
    -> E0 versus teacher-oracle T0/T1 report
```

P0 필수 조건:

- readable selected RGB와 actual sensor coverage report;
- pinned provider, checkpoint, semantic mask/place ontology, input digest;
- question, choice, label, annotated evidence가 memory construction에 들어가지 않음;
- `model_inferred` record가 selected frame, validity, uncertainty, confidence,
  complete-index 조건을 만족;
- source별 30초 window당 동일 canonical byte budget;
- company resource 실행 전 명시적 승인.

핵심 experiment는 [EXP-0004](experiments/exp-0004-gcut3r-provider.md)와
[EXP-0005](experiments/exp-0005-teacher-oracle-ceiling.md)다.

## 후속 작업

- **P1:** [ADR-0005](decisions/adr-0005-hybrid-on-device-compiler.md)에 따라 작은
  detector/semantic encoder와 native causal VIO/depth projection을 결합한다. Legacy
  supplied-vector head는 architecture baseline이 아니다.
- **P2:** 같은 executable을 target glass hardware에서 profile하고 calibration drift와
  missing-sensor fallback을 검증한다. Server GPU latency로 대체하지 않는다.
- **P3:** Causal existing/`NEW` association, movement event, complete entity index,
  relocalization을 추가한다.
- **P4:** EXP-0002 student, matched E1/E2/E3, EXP-0003 byte Pareto를 실행한다.
- **P5:** Plane, portal, free-space, landmark, place/submap type은 새로운 QA utility와
  byte value가 확인될 때 별도 experiment로 추가한다.

## 보류

VQ/FSQ codec, dense neural-scene storage, custom ANN, all-type generative decoder,
G-CUT3R on-device 실행, photorealistic reconstruction은 보류한다. P0/P1에서 확인된
bottleneck이 요구할 때만 추가한다.

운영 세부 절차는 repository `HANDOFF.md`이며 Confluence에서는
[운영](operations/README.md) 아래에 둔다.

[프로젝트 홈으로 돌아가기](README.md)
