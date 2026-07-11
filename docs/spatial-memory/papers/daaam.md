# DAAAM: Describe Anything Anywhere At Any Moment

| Field | Value |
| --- | --- |
| Page ID | SM-PAPER-DAAAM |
| Status | Reviewed from primary sources; not reproduced in this project |
| Publication | CVPR 2026, pp. 35002-35013 |
| Primary source | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Gorlo_Describe_Anything_Anywhere_At_Any_Moment_CVPR_2026_paper.html), [arXiv:2512.00565](https://arxiv.org/abs/2512.00565) |
| Official code | [MIT-SPARK/DAAAM](https://github.com/MIT-SPARK/DAAAM) |
| Last checked | 2026-07-11 |
| Project links | [Papers index](README.md), [Problem](../problem.md), [Architecture](../architecture.md), [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003, C-004 |

## 30-second summary

**Paper claim:** RGB-D stream을 hierarchical 4D scene graph로 reconcile하고 entity별 detailed description history와 timestamp를 저장하면, frame database보다 spatial·temporal consistency가 높은 real-time embodied memory를 만들 수 있다.

## Problem addressed

Frame-level VLM memory는 semantic detail이 풍부하지만 multi-view object identity와 metric geometry가 약하다. Metric-semantic map은 geometry가 강하지만 open-vocabulary detail이 부족하거나 per-object model inference가 비싸다. DAAAM은 두 장점을 4D explicit graph에서 결합한다.

## Relevant method

- RGB-D와 pose에서 FastSAM·BotSort로 fragment를 track하고 Khronos/Hydra로 4D metric-semantic graph를 만든다.
- Greedy set cover와 binary optimization으로 모든 fragment를 덮는 최소·고품질 frame을 선택한다.
- 선택 frame의 localized mask를 DAM으로 batch captioning하고 CLIP·sentence embedding을 함께 부여한다.
- Backend factor graph가 fragment를 global optimize·reconcile하며 description history와 timestamp를 보존한다.
- Place와 region hierarchy를 만들고 tool-calling agent가 object, region, agent의 spatial-temporal field를 조회한다.

## Paper-reported evidence

**Paper-reported results:**

- OC-NaVQA에서 question accuracy `0.711`, positional error `41.75 m`, temporal error `1.792 min`을 보고했다. 출처: Table 2.
- Abstract 기준 경쟁 baseline 대비 question accuracy `+53.6%`, position error `-21.9%`, temporal error `-21.6%`의 상대 개선을 보고했다. 출처: abstract와 Table 2.
- SG3D에서 sub-task accuracy `22.16%`, full-task accuracy `11.22%`; 비교한 ASHiTA는 `21.7%`, `8.78%`였다. 출처: Table 3.
- Batched semantic frontend가 per-object 방식 대비 order-of-magnitude speedup을 제공하며 geometry tracking/reconstruction은 sensor rate `10 Hz`로 동작한다고 보고했다. 출처: Sections 3과 4.4.

## What this supports here

**Project inference:** persistent entity ID, change-aware description history, timestamp, hierarchical place/object graph, tool-based deterministic retrieval을 결합하는 구조의 근거다. 모든 frame을 caption하지 않고 representative frame set을 선택하는 evidence reservoir 설계에도 직접 연결된다.

## What it does not prove

- 1 Hz monocular low-overlap geometry
- Ground-truth pose 없이 얻는 동일한 결과
- Actual serialized byte budget 또는 learned QA-value writer
- Generic geometry core의 최소 충분성
- SuperMemory-VQA 또는 AI-glass on-device 성능

모든 평가에 ground-truth pose를 제공했다. Small object와 limited tool coverage에서 실패가 보고됐고, 원 NaVQA 210 samples에는 noisy annotation과 in-context leakage 문제가 있어 저자들이 OC-NaVQA를 재구성했다.

## Project reproduction status

**Project result:** Not reproduced. DAAAM, Khronos, DAM과 공식 benchmark를 실행하지 않았다. 이 저장소의 event/validity schema와 geometry tools는 설계 연결일 뿐 CVPR 결과 재현이 아니다.

## References

- [CVF publication](https://openaccess.thecvf.com/content/CVPR2026/html/Gorlo_Describe_Anything_Anywhere_At_Any_Moment_CVPR_2026_paper.html)
- [Official project page](https://nicolasgorlo.com/DAAAM_25/)
- [arXiv paper](https://arxiv.org/abs/2512.00565)
- [Official code](https://github.com/MIT-SPARK/DAAAM)
- [Official ROS 2 interface](https://github.com/MIT-SPARK/DAAAM-ROS)
- [Parent papers index](README.md)
