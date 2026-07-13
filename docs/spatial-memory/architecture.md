# 아키텍처

| 항목 | 값 |
|---|---|
| Page ID | SM-ARCHITECTURE |
| 상태 | 갱신형 specification |
| 최종 갱신 | 2026-07-13 |
| 현재 구현 | Heuristic baseline과 external typed-inference bridge |
| 목표 | Guided transient geometry에서 learned typed persistent memory까지 연결 |

## 핵심 아키텍처 결정

| 영역 | 결정 |
|---|---|
| Persistent representation | Dense map/recurrent snapshot 대신 explicit typed record 사용 |
| Geometry provider | G-CUT3R-compatible model을 transient external teacher/front-end로 사용 |
| Storage control | Canonical serialized byte에 hard limit 적용 |
| Spatial answer | Deterministic executor가 proof 또는 abstention 생성 |
| 현재 경계 | Repository checkpoint 이후 external type-specific decode 필요 |

이 아키텍처는 contract probe 준비 상태이며 production 품질을 입증한 상태가 아니다.
핵심 검증 대상은 실제 checkpoint가 question/label leakage 없이 valid하고 유용한
byte-bounded record를 생성하는지 여부다.

## 시스템 흐름

```text
1 Hz RGB + native-rate IMU/VIO + optional calibrated depth
    -> transient guided geometry state
    -> typed candidates
       object / plane / portal / free-space / landmark / event / no-write
    -> utility and actual-byte selection
    -> local-frame persistent database
    -> causal retrieval
    -> deterministic geometry proof
    -> four-choice answer or abstention
```

현재 local baseline은 learned geometry와 selection을 structured source field와
heuristic으로 대체한다. 목적은 contract 검증이다.

## Memory layer 구성

| Layer | 수명 | 목적 |
|---|---|---|
| Fast pose memory | 짧음 | Alignment, motion, IMU/VIO state |
| Transient geometry | Processing window 단위 | Point map, depth, confidence, candidate slot |
| Persistent map | 장기 | 감사 가능한 place, structure, object, free-space, event fact |
| Relocalization memory | 장기·sparse | Position, viewing ray, descriptor, uncertainty |
| Evidence reservoir | 장기·bounded | Text, appearance, surprise, uncertain observation |

Tracking state는 persistent fact를 덮어쓸 수 없다. Dense geometry는 verified
record를 쓴 뒤 폐기한다.

## Persistent record contract 정의

| Type | 최소 geometry | 의사결정 용도 |
|---|---|---|
| Place/submap | Local SE(3) anchor, covariance, topology | Coordinate ownership, loop correction |
| Plane | Normal, offset, extent, uncertainty | Wall, floor, ceiling, support |
| Portal | Position, orientation, extent, connected frame | Transition, topology |
| Free space | Coarse polygon/tile, height, validity | Reachability, visibility |
| Object | Entity/instance ID, centroid, extent, orientation | Metric·semantic object QA |
| Landmark | Position, viewing ray, descriptor, quality | Relocalization, association |
| Event | Kind, entity, before/after state, validity | Movement, state change |
| No-write | Decision trace만 보유 | Rejected candidate이며 persist 금지 |

모든 persisted record는 source video, local frame, temporal validity,
confidence/uncertainty, provenance, evidence reference를 포함한다.

## Production bridge 구조

```text
spatial_student.pt + sanitized sources + selected frames + sensor manifest
    -> WORLDMM_SPATIAL_INFER_EXE (`worldmm-spatial-infer-v1`)
    -> canonical typed_memory.jsonl + typed_memory.inference.json
    -> repository schema, byte, digest, retrieval, proof, and QA validation
```

Executable에는 question과 label을 전달하지 않는다. Input은 selected source
identity/time, pose/gaze, frame reference, frame timestamp만 포함한다. Transcript,
caption, OCR, object annotation, unselected frame은 제거한다. Executable은
type-specific decode, open-world association, candidate selection을 담당하고,
repository code는 validation과 downstream evidence를 담당한다.

## 필수 control

| Control | 요구사항 |
|---|---|
| Sampling | 모든 variant가 하나의 causal at-most-1-Hz frame inventory 공유 |
| Byte budget | Canonical UTF-8 JSONL, 기본 4,096 byte per `(source_video_id, floor(first_seen_time / 30s))` window |
| Artifact validity | Empty/noncanonical output, duplicate ID, persisted `no_write`, 1 MiB 초과 row, manifest mismatch, over-budget window 거부 |
| Evidence grounding | Observed/fused/human record는 validity 내 selected same-video frame을 인용하고 min/max evidence time과 first/last seen, unique count와 `observation_count`가 일치 |
| Identity/time | Explicit ID 우선, observation별 one-to-one association, move 시 이전 validity 종료, latest causal non-conflicting state retrieval |
| Coordinate | Record는 local frame에 속하고 cross-frame operation은 explicit transform 요구 |
| Retrieval | Question-video scope와 question-time cutoff 필수 |
| Geometry proof | Entity role, operation, value, frame, unit, uncertainty, provenance, evidence, behavior-affecting option을 stable hash에 포함 |
| Completeness | `count`/`last_seen`은 complete-index certificate, label-only pair는 certified uniqueness가 없으면 abstain |
| QA boundary | Prompt에는 raw geometry dictionary가 아닌 verified proof만 제공하며 answer choice와 cited proof가 일치하고 real frame을 load해야 함 |
| Lineage | Checkpoint, executable, typed memory, memory manifest, episodic/semantic/visual store, frame, sensor, config, data, prompt, prediction, metric hash 재계산·결속 |
| Finalization | Remote manifest를 completion marker로 쓰기 전에 QA completion과 finalization input seal |

현재 deterministic operation은 `distance`, `near`, `relative_direction`,
`last_seen`, `count`다. Missing frame, ambiguous entity, incompatible frame,
excessive uncertainty, incomplete index는 abstention을 유발한다.

## Writer 목표

실제 저장 byte당 future QA·geometry value가 가장 큰 record를 선택한다.

```text
utility = QA loss reduction + geometry coverage + uncertainty reduction
        + relocalization value + event surprise - redundancy - byte cost
```

현재 production validation은 byte를 강제하지만 external decoder가 이 utility를
학습했는지는 입증하지 못한다. 이는 회사 run에서 검증할 empirical question이다.

## 현재 gap과 실행 방향

Repository는 typed schema, teacher/cache contract, DDP candidate-head training,
hard byte validation, retrieval, proof, QA check, lineage, report generation을
소유한다. Production G-CUT3R extractor, raw sensor encoder, type-specific
inference decoder, learned open-world association은 소유하지 않는다.

다음 실행은 승인된 contract probe다. Probe가 decode, association, byte
selection, grounding, lineage의 구체적 실패를 확인하기 전에는 새 memory
abstraction을 추가하지 않는다.

## 채택한 결정

- [ADR-0001: Explicit typed memory](decisions/adr-0001-explicit-typed-memory.md)
- [ADR-0002: G-CUT3R teacher](decisions/adr-0002-gcut3r-as-teacher.md)
- [ADR-0003: Actual byte당 value](decisions/adr-0003-value-per-byte-writer.md)
- [ADR-0004: Deterministic geometry proof](decisions/adr-0004-deterministic-geometry-proof.md)

현재 go/no-go는 [현재 상태](status.md), 승인된 실행 절차는 repository
`HANDOFF.md`를 따른다.

[프로젝트 홈으로 돌아가기](README.md)
