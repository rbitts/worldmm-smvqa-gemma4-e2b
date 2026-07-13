# 현재 상태

| 항목 | 값 |
|---|---|
| Page ID | SM-STATUS |
| 기준일 | 2026-07-13 |
| 로컬 준비 | 완료 |
| Heuristic baseline | Tiny fixture에서 동작 |
| Learned bridge | Contract 구현, 미실행 |
| 실제 training/evaluation | 미실행 |

## 핵심 결론

| 결정 | 결과 |
|---|---|
| 공식 benchmark 주장 | **No-Go** |
| 다음 승인 요청 | 1-node × 1-GPU contract probe만 허용 |
| 핵심 이유 | External typed inference가 회사 checkpoint·frame·data에서 미실행 |
| Full-scale 조건 | Probe가 valid non-empty byte-bounded typed evidence와 완결된 lineage 생성 |

Repository는 production handoff 검증 준비가 됐지만 learned-method 품질 보고 준비는
되지 않았다. Probe 성공 결과도 `contract_probe` / `PROBE`이며, 별도 승인된
`full` run만 `student` / `E1`을 생성한다.

## 근거

| 영역 | 현재 근거 | 한계 |
|---|---|---|
| Explicit memory | Typed schema, causal retrieval, hard serialized-byte writer | Tiny fixture만 검증 |
| Geometry QA | Deterministic proof, proof-to-choice 검사, abstention | 지원 operator 제한 |
| Training | Global-normalized DDP loss, atomic checkpoint, resume | Candidate head가 supplied vector 사용 |
| Production handoff | Sanitized input, typed JSONL validation, lineage, finalization seal | External executable 미검증 |
| Reporting | Profile-bound identity와 report contract | Matched E2/E3 identity 없음 |

Learned 경로:

```text
teacher cache + supervision
    -> DDP typed candidate head
    -> spatial_student.pt
    -> WORLDMM_SPATIAL_INFER_EXE
    -> canonical typed memory
    -> repository validation, retrieval, proof, QA, report
```

## 핵심 blocker

1. Repository-owned G-CUT3R extractor와 raw RGB/IMU/VIO student encoder가 없다.
2. Repository-owned type-specific inference decoder와 learned open-world
   association이 없어 production은 `WORLDMM_SPATIAL_INFER_EXE`에 의존한다.
3. External inference와 `worldmm-spatial-infer-v1` output을 회사 artifact로
   검증하지 않았다.
4. 공식 보고에 필요한 immutable matched E2 spatial-ablation, E3
   retrieval-ablation identity가 없다.
5. Counterfactual QA utility가 deployed typed write gate에 연결되지 않았다.
   이는 probe input이 아닌 후속 연구 milestone이다.

## Local sanity 결과

| Metric | Spatial 사용 | Spatial 미사용 |
|---|---:|---:|
| Ans-F1 | 100.00 | 100.00 |
| QA-Acc | 66.67 | 50.00 |
| QA-MRR | 83.33 | 72.22 |
| Relation F1 | 1.00 | 해당 없음 |

위 synthetic 값은 plumbing만 검증한다. Compact artifact는 causal window 전체에서
6,050 byte였으며 limit는 lifelong cap이 아닌 window별 cap이다. 상세 내용은
[EXP-0001](experiments/exp-0001-source-compact-baseline.md)을 본다.

## Probe 통과 조건

다음 조건을 모두 통과한 뒤 full-run 승인을 요청한다.

1. Non-empty canonical typed record가 source별 30초 window당 4,096 byte 이하다.
2. Repository validation이 checkpoint, executable, model, frame, sensor, memory,
   evidence, prompt, prediction, metric digest를 연결한다.
3. QA가 실제 selected frame을 load하고 typed memory에서 evidence를 재구성한다.
4. Geometry answer가 matching deterministic proof를 인용한다.
5. Causal, off-scope, duplicate, manifest violation이 0이다.
6. Probe identity와 report가 `contract_probe` / `PROBE`로 seal된다.

## Remote 상태

- SSH session: 없음.
- Slurm submission 또는 job ID: 없음.
- Company artifact: 없음.
- 로컬로 복사한 dataset, model, checkpoint: 없음.

실행 source of truth는 repository `HANDOFF.md`이며 Confluence에서는
[운영](operations/README.md) 아래에 둔다.

[프로젝트 홈으로 돌아가기](README.md)
