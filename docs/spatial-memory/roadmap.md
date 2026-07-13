# 연구 로드맵

| 항목 | 값 |
|---|---|
| Page ID | SM-ROADMAP |
| 상태 | 활성 |
| 최종 갱신 | 2026-07-13 |
| 우선순위 원칙 | 최적화 전에 end-to-end correctness 완결 |

## 실행 방향

Codec, 더 큰 architecture, device claim을 추가하기 전에 learned
checkpoint-to-QA path를 완성하고 검증한다. 현재 우선순위는 승인된 contract
probe 하나이며 공식 benchmark 보고는 차단 상태다.

## 우선순위와 의사결정 gate

| 우선순위 | 목표 결과 | 다음 단계 gate |
|---|---|---|
| P0 | Learned checkpoint가 canonical typed evidence와 proof-grounded QA 생성 | 승인된 1×1 probe의 lineage·byte·grounding·causality 검사 통과 |
| P1 | Learned E1 baseline과 QA-versus-bytes curve | Full E1과 immutable matched E2/E3 identity |
| P2 | QA utility가 deployed write gate를 supervise | Equal-byte geometry-novelty baseline 대비 개선 |
| P3 | Revisit 간 stable identity와 relocalization | False merge, duplicate ID, loop-closure 목표 통과 |
| P4 | 미지 질문 coverage | Held-out operator/category coverage와 calibrated abstention 유지 |
| P5 | On-device student 가능성 | 측정 latency, energy, memory, hardware calibration 통과 |

## P0 실행

```text
teacher extraction -> record-derived supervision -> DDP training
    -> WORLDMM_SPATIAL_INFER_EXE -> typed records + hard byte writer
    -> repository validation -> retrieval -> proof -> real-frame QA
    -> sealed PROBE identity and report
```

Scale-up 전 필수 조건:

- pinned G-CUT3R-compatible teacher와 production inference executable;
- variable geometry를 포함한 type-specific decode와 existing-pointer 또는 `NEW`
  association;
- source별 30초 window당 4,096 byte 이하 canonical typed JSONL;
- 완결된 checkpoint, executable, data, frame, sensor, evidence, QA, report lineage;
- causal/off-scope violation 0;
- 회사 GPU submission 전 명시적 승인.

핵심 experiment: [EXP-0004](experiments/exp-0004-gcut3r-provider.md),
[EXP-0002](experiments/exp-0002-typed-memory-bridge.md).

## 후속 작업

- **P1:** QA-Acc, QA-MRR, Ans-F1, geometry grounding, actual byte, revisit
  growth, latency, resource use를 보고하고 matched E2/E3 run을 만든다.
- **P2:** 별도 selector abstraction을 추가하지 않고 기존 typed write logit을
  counterfactual deletion utility로 supervise한다.
- **P3:** Causal pointer association, viewing-ray compatibility, submap loop
  correction, retain-or-replace landmark를 추가한다.
- **P4:** Structural fact를 보호하고 text, appearance, surprise, uncertainty용
  small bounded evidence reservoir를 추가한다.
- **P5:** P0–P4에서 record contract가 안정화된 뒤 distill·profile한다.

## 보류

VQ/FSQ codec, dense neural-scene storage, custom ANN infrastructure,
end-to-end QA-model training, photorealistic reconstruction은 보류한다. Explicit
baseline이 측정된 bottleneck일 때만 추가한다.

운영 세부 절차는 repository `HANDOFF.md`이며 Confluence에서는
[운영](operations/README.md) 아래에 둔다.

[프로젝트 홈으로 돌아가기](README.md)
