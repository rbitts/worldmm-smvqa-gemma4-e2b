# 2026-07-14 Issue #3 개편·구현 검토

| 항목 | 값 |
|---|---|
| Page ID | SM-REVIEW-2026-07-14-ISSUE-3 |
| Confluence parent | SM-REVIEWS |
| 검토 유형 | Architecture, implementation, fail-closed behavior |
| 범위 | Goal, ADR, experiments, sensor/teacher target, typed proof, tests |
| Remote 작업 | 없음 |
| 결론 | Local contract Go; teacher-oracle/student/device claim No-Go |

## 핵심 결론

기존 Issue #3의 raw RGB/IMU all-type student 범위는 현재 근거보다 컸다. G-CUT3R를
offline teacher/oracle로 제한하고, object/location teacher-oracle utility를 먼저
측정한 뒤 최소 hybrid device compiler를 결정하는 순서로 개편했다.

Local vertical slice는 통과했다. Real provider, semantic mask/place adapter,
teacher-oracle benchmark, raw student, target-device profile은 없으므로 learned 또는
on-device 완료로 해석하지 않는다.

## Review finding과 조치

| Finding | 조치 | 상태 |
|---|---|---|
| `build_student()`가 raw student처럼 오해될 수 있음 | Internal class를 `TypedCandidateHead`로 바꾸고 feature-level scaffold임을 명시 | 해결 |
| Camera intrinsics가 depth guidance에만 종속 | 공용 sensor schema로 분리하고 teacher request에 독립 calibration 추가 | 해결 |
| Single-frame target cutoff가 evidence timestamp보다 이를 수 있음 | `observed_through_time == timestamp` 강제, contextual grounding test 추가 | 해결 |
| `model_inferred` proof가 evidence/confidence 없이 막히거나 자기참조로 우회 가능 | Selected frame evidence, self-reference 거부, confidence threshold, uncertainty/provenance 유지 | 해결 |
| Object/location task를 timestamp `last_seen`으로 대신할 위험 | Optional `place_label`, `last_location`, complete-index/stale-event/missing-time gate, choice consistency 추가 | 해결 |
| `where`와 `when` 복합 intent 및 non-unit gaze가 임의 해석될 수 있음 | 복합 intent abstain, gaze unit-vector contract 추가 | 해결 |
| Canonical docs의 RQ anchor와 paper reverse metadata가 import test와 불일치 | Metadata/anchor/traceability를 일괄 정합화 | 해결 |

## 검증 결과

| 검사 | 결과 |
|---|---|
| `ruff check .` | 통과 |
| `basedpyright` | 0 errors, 0 warnings |
| `pytest -q` | 466 passed, 1 environment-specific skip |
| `git diff --check` | 통과 |
| Spatial-memory documentation contract | 3 passed |

Skip은 local environment에 `transformers`가 설치된 경우 제외되는 backend test다.
Real model/data, training, evaluation, SSH, Slurm은 실행하지 않았다.

## 남은 blocker

1. Company prepared data의 selected RGB asset, camera intrinsics, native pose/depth
   coverage audit.
2. Pinned G-CUT3R-compatible provider와 semantic mask/place adapter.
3. [EXP-0005](../experiments/exp-0005-teacher-oracle-ceiling.md)의 same-byte
   E0/T0/T1 object/location result와 confidence calibration.
4. Oracle Go 이후 raw RGB semantic encoder, causal existing/`NEW` association,
   target-device latency/memory/energy profile.

## 의사결정 영향

- Issue #3는 open epic으로 유지한다.
- P0는 student probe가 아니라 sensor audit + teacher-oracle ceiling이다.
- Legacy DDP candidate head와 external inference lineage는 재사용 가능한 scaffold지만
  production student evidence가 아니다.
- Plane, portal, free-space, landmark, 별도 Place/submap type은 object/location utility
  이후로 보류한다.

[검토 목록으로 돌아가기](README.md)
