# ADR-0005: Hybrid on-device spatial compiler 경계

| 항목 | 값 |
|---|---|
| Page ID | SM-ADR-0005 |
| Confluence parent | SM-DECISIONS |
| ADR | ADR-0005 |
| 상태 | 채택; local contract 일부 구현 |
| 결정일 | 2026-07-14 |
| 추적성 | [Device cost와 transient geometry 요구사항](../traceability.md) |
| 아키텍처 | [Teacher-oracle과 device compiler 경계](../architecture.md) |

## 핵심 결론

- **결정:** G-CUT3R는 company compute의 offline teacher/oracle로만 사용한다.
  최종 device compiler는 작은 semantic perception과 native IMU/VIO/depth를 결합한
  hybrid pipeline으로 만든다.
- **초기 범위:** SuperMemory-VQA의 object/location 효용을 검증할 수 있는
  object, semantic place, movement event, no-write만 우선한다. Plane, portal,
  free-space, landmark를 한 번에 생성하는 generic decoder는 만들지 않는다.
- **다음 gate:** [EXP-0005](../experiments/exp-0005-teacher-oracle-ceiling.md)가
  동일 byte budget에서 object/location QA 효용을 입증해야 raw student distillation을
  시작한다.

## 근거와 판단 이유

- [G-CUT3R](../papers/g-cut3r.md)는 large external geometry model 후보이며 glass
  runtime 또는 persistent-memory 구현 근거가 아니다.
- [SuperMemory-VQA](../papers/supermemory-vqa.md)는 long-horizon object/location
  memory 효용을 먼저 확인할 benchmark context를 제공한다.
- Native pose/depth를 deterministic geometry에 직접 사용하는 편이 큰 teacher의
  hidden state와 all-type geometry decoder를 device에 복제하는 것보다 책임 경계가
  작고 감사 가능하다.
- 기존 `build_student()`는 supplied feature vector를 받는 candidate head다. Raw
  RGB/IMU encoder, open-world association, mobile executable이 아니며 device result로
  보고하지 않는다.

## 구현 방향

- `src/worldmm_smvqa/worldmm/spatial_sensor.py`가 camera intrinsics, optional depth,
  unit gaze ray, trusted causal pose를 하나의 observation contract로 검증한다.
- `src/worldmm_smvqa/worldmm/gcut3r_teacher.py`는 camera intrinsics를 depth 유무와
  독립적으로 offline teacher request에 보존한다.
- `src/worldmm_smvqa/worldmm/spatial_teacher_targets.py`는 external semantic mask로
  선택된 teacher points를 evidence-bound object target으로 변환한다. Mask/detector
  자체는 구현하지 않는다.
- Device MVP는 작은 pretrained detector/semantic encoder, native VIO/depth의
  deterministic projection, causal association, typed writer 순서다. Oracle 이득이
  확인된 target만 distill한다.

## 검증 결과와 남은 과제

- Tiny local tests는 calibration 보존, offline pose 거부, non-future certificate,
  point-to-object target, extent/uncertainty floor를 검사한다.
- Real RGB detector, mask-to-point correspondence, native sensor ingest, identity
  association, mobile profile은 미구현이다.
- Latency, peak memory, energy, thermal throttling, hardware calibration을 실측하기
  전에는 on-device 가능성을 주장하지 않는다.

## 배경

이전 Issue #3 범위는 raw RGB/IMU encoder, learned projector, 모든 record type의
decoder, open-world pointer, submap, compaction, mobile profile을 한 번에 요구했다.
현재 dataset signal과 oracle utility가 확인되지 않은 상태에서 이 범위를 구현하면
학습 대상과 device 구조를 동시에 추측하게 된다.

## 대안

| 대안 | 미선택 이유 |
|---|---|
| G-CUT3R를 glass에서 직접 실행 | 공개된 scale과 runtime은 device budget 근거가 아니며 persistent typed DB와도 다름 |
| G-CUT3R hidden state를 장기 저장 | explicit entity, byte budget, proof contract를 제공하지 않음 |
| 모든 record type을 생성하는 end-to-end student | 현재 benchmark utility와 supervision coverage가 확인되지 않음 |
| Pose/depth까지 neural student가 재예측 | device native signal을 중복 계산하고 calibration failure를 숨김 |

## Trade-off 검토

- 초기 coverage는 좁지만 object/location slice에서 실패 원인을 분리할 수 있다.
- Device pipeline은 sensor calibration과 platform-specific profiling이 필요하다.
- Missing calibration, untrusted pose, low-confidence inference는 `no_write` 또는
  proof abstention으로 처리한다.

## 대체 이력

없음. [ADR-0002](adr-0002-gcut3r-as-teacher.md)의 external teacher 결정을
유지하면서 production compiler 경계를 추가한다.
