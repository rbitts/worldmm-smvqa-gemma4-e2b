# ADR-0001: Persistent spatial memory를 explicit typed record로 저장

| 항목 | 값 |
|---|---|
| Page ID | SM-ADR-0001 |
| ADR | ADR-0001 |
| 프로젝트 claim | C-002, C-006, C-012 |
| 상태 | 채택; 일부 구현 |
| 결정일 | 2026-07-11 |
| 추적성 | [Explicit·queryable memory 요구사항](../traceability.md) |
| 아키텍처 | [Typed persistent memory](../architecture.md) |

## 핵심 결론

Dense geometry와 recurrent state는 일시적인 추론 중간값으로만 사용한다. 영구
메모리는 `object`, `plane`, `portal`, `free_space`, `landmark`, `event`의 typed
record로 저장하고, 쓰지 않을 후보는 `no_write`로 표현한다. 모든 writable
record는 source video, entity와 instance ID, local coordinate frame, validity,
uncertainty, observation count, provenance, evidence reference를 포함한다.

이 결정은 전체 submap graph, loop closure, ray-aware landmark 교체 정책이 현재
구현되었다는 뜻이 아니다. 현재 구현은 schema, 직렬화, retrieval 진입,
heuristic source-compact baseline까지다.

## 근거와 판단 이유

- [Point3R](../papers/point3r.md)는 spatially indexed explicit memory 설계의 관련
  근거다. 이 프로젝트는 generic high-dimensional pointer를 영구 record로 그대로
  채택하지 않는다.
- [ConceptGraphs](../papers/conceptgraphs.md)는 object-centric scene representation의
  관련 근거다. 이 프로젝트는 object 외에 metric structure와 temporal event를
  별도 type으로 둔다.
- [LONG3R](../papers/long3r.md)는 제한된 장기 memory 선택의 관련 근거다. 이
  프로젝트의 선택 기준은 reconstruction attention에 한정하지 않는다.
- 현재 schema와 writer는 `src/worldmm_smvqa/worldmm/typed_memory.py`에
  구현되어 있다.
- 현재 구현 범위와 미연결 learned lane은
  [local readiness review](../reviews/2026-07-11-local-readiness.md)에 구분되어 있다.

## 구현 방향

- `src/worldmm_smvqa/worldmm/typed_memory.py`는 typed schema,
  validity 검사, covariance 검사, canonical JSONL 직렬화와 production
  source/sensor grounding 검사를 제공한다. Observed/fused/human-confirmed
  record의 bare `frame_ref` evidence는 같은 source video의 선택 frame이며
  min/max timestamp가 first/last seen과 같고 unique ref 수가
  `observation_count`와 같아야 한다. Window budget은 backdating을 막기 위해
  `first_seen_time`을 사용한다. Production reader는 streaming하며 canonical
  row 하나를 1 MiB로 제한한다.
- `src/worldmm_smvqa/worldmm/spatial_compression.py`는
  현재 source metadata에서 object, relation, zone, trajectory record를 만드는
  heuristic baseline이다.
- Retrieval은 flat typed record의 geometry를 evidence로 변환할 수 있다.
- 미구현: raw RGB/IMU/VIO encoder, checkpoint-specific typed decoder, open-world
  association, submap graph optimization, learned checkpoint에서 persistent record로
  이어지는 end-to-end 경로.

## 검증 결과와 남은 과제

- `tests/test_typed_memory.py`는 record validation과
  schema invariants를 검사한다.
- `tests/test_typed_memory_writer.py`는 persistent
  artifact와 byte budget을 검사한다.
- `tests/test_spatial_compression.py`는 heuristic
  source-compact 경로를 검사한다.
- 검증은 tiny fixture와 로컬 단위 검사 범위다. 실제 G-CUT3R 출력, 장기간 반복
  방문, official SuperMemory-VQA benchmark 성능은 아직 검증되지 않았다.

## 배경

1 Hz 영상 스트림의 dense pointmap, recurrent hidden state, patch feature를 장기간
그대로 저장하면 관측 프레임 수에 따라 저장량이 증가한다. 또한 generic latent
feature만으로는 object identity, coordinate frame, validity, uncertainty,
provenance를 직접 검사하기 어렵다. SuperMemory-VQA 질의에는 시간 범위와
근거가 필요하므로 영구 메모리는 query-time geometry 연산이 가능한 형태여야
한다.

## 대안

| 대안 | 미선택 이유 |
|---|---|
| 원본 frame 또는 keyframe 장기 저장 | frame 수에 따라 증가하고 explicit metric query를 직접 실행할 수 없음 |
| Dense pointmap 또는 mesh 저장 | 장기 byte budget과 반복 방문 consolidation 목표에 맞지 않음 |
| CUT3R recurrent state snapshot 저장 | entity, validity, provenance가 explicit하지 않고 state snapshot을 누적 저장할 수 없음 |
| Generic spatial pointer와 latent feature만 저장 | spatial index는 얻지만 QA executor가 필요한 typed facts가 feature 안에 숨음 |
| Object-only scene graph | plane, portal, free space, relocalization, change event가 누락됨 |

## Trade-off 검토

- Geometry와 temporal facts를 deterministic executor가 직접 읽을 수 있다.
- Schema validation과 provenance를 trust boundary에서 강제할 수 있다.
- Record type별 decoder와 association logic이 필요해진다.
- Typed core가 보존하지 않는 OCR, 미세 외형, novel event에는 별도 evidence
  reservoir가 필요할 수 있다.
- Schema 변경은 저장 artifact 호환성에 영향을 주므로 versioning 또는 migration
  결정이 필요할 때 새 ADR을 작성한다.

## 대체 이력

없음.
