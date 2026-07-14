# 현재 상태

| 항목 | 값 |
|---|---|
| Page ID | SM-STATUS |
| 기준일 | 2026-07-14 |
| 현재 Goal | Teacher-oracle object/location ceiling 검증 준비 |
| 로컬 준비 | Sensor/target/proof vertical slice 구현 |
| 실제 training/evaluation | 미실행 |

## 핵심 결론

| 결정 | 결과 |
|---|---|
| 공식 benchmark 또는 on-device 주장 | **No-Go** |
| 다음 승인 요청 | Sensor coverage 확인 후 EXP-0005 bounded teacher-oracle run |
| 핵심 이유 | Oracle QA utility, real provider, source calibration/frame availability 미검증 |
| Student 시작 조건 | EXP-0005가 동일 byte E0보다 object/location slice 개선 |

G-CUT3R는 offline teacher/oracle로만 사용한다. 기존 DDP `build_student()`는 supplied
feature vector candidate head이며 raw RGB/IMU encoder, open-world association,
mobile model이 아니다. 따라서 legacy checkpoint probe를 student architecture
검증으로 보고하지 않는다.

## 로컬 구현 근거

| 영역 | 구현 | 한계 |
|---|---|---|
| Sensor | Camera intrinsics, optional depth/gaze, trusted causal IMU/VIO observation | Prepared real signal coverage 미확인 |
| Teacher | External provider/cache + selected point object target compiler | G-CUT3R와 semantic mask/place provider 미연결 |
| Typed memory | Optional object `place_label`, hard canonical byte writer | 별도 Place/submap type 없음 |
| Geometry QA | Self-reference를 거부하는 evidence/confidence-gated `model_inferred`, `last_location`, proof-to-choice 검사 | Real confidence calibration과 ontology 미검증 |
| Training scaffold | Global-normalized DDP loss, checkpoint/resume | Feature-level closed-set candidate head뿐 |
| Lineage | External inference, artifact, evidence, QA digest validation | Real executable/checkpoint 미검증 |

## 핵심 blocker

1. Prepared source의 readable RGB, camera intrinsics, native IMU/VIO, depth coverage를
   실제 company data에서 확인하지 않았다.
2. Repository-owned G-CUT3R extractor와 semantic mask/place provider가 없다.
3. Teacher object/place record가 same-byte E0보다 QA utility를 높이는지 미측정이다.
4. Oracle Go 이후 필요한 raw RGB semantic encoder와 causal existing/`NEW`
   association, target-device executable이 없다.
5. 공식 보고용 matched E1/E2/E3와 byte-Pareto run이 없다.

## EXP-0005 통과 조건

1. Selected RGB asset coverage 100%이며 available calibration/pose/depth 비율을
   누락 없이 보고한다.
2. Teacher request, semantic selection, typed record가 question/label을 입력받지 않는다.
3. `model_inferred` record가 same-video selected frame, causal validity, uncertainty,
   confidence, complete entity index에 결속된다.
4. E0/T0/T1이 같은 causal frame inventory, QA backend, serialized-byte budget을 쓴다.
5. Invalid record, future/off-scope evidence, duplicate ID, accepted low-confidence
   proof가 0이다.
6. Object/location slice 개선폭과 selective-risk 기준을 run 전에 고정한다.

## Local sanity 결과

| Metric | Spatial 사용 | Spatial 미사용 |
|---|---:|---:|
| Ans-F1 | 100.00 | 100.00 |
| QA-Acc | 66.67 | 50.00 |
| QA-MRR | 83.33 | 72.22 |
| Relation F1 | 1.00 | 해당 없음 |

위 synthetic 값은 기존 plumbing만 검증한다. Teacher-oracle, learned student,
benchmark, device 근거가 아니다. 상세 내용은
[EXP-0001](experiments/exp-0001-source-compact-baseline.md)을 본다.

## Remote 상태

- SSH session: 없음.
- Slurm submission 또는 job ID: 없음.
- Company artifact: 없음.
- 로컬로 복사한 dataset, model, checkpoint: 없음.

실행 source of truth는 repository `HANDOFF.md`이며 Confluence에서는
[운영](operations/README.md) 아래에 둔다.

[프로젝트 홈으로 돌아가기](README.md)
