# FARM: Find Anything using Relational Spatial Memory

| 항목 | 값 |
| --- | --- |
| Page ID | SM-PAPER-FARM |
| 상태 | Primary source 검토 완료; official code 부재로 reproduction 차단 |
| 출판 | arXiv preprint 2606.15476v2, 2026 |
| 1차 출처 | [arXiv:2606.15476](https://arxiv.org/abs/2606.15476), [official project](https://goldengait.github.io/farm/) |
| 공식 code | 2026-07-11 기준 미공개; official project에 `Code coming soon` 표기 |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [논문 목록](README.md), [문제 정의](../problem.md), [아키텍처](../architecture.md), [추적성](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-002 |

## 핵심 결론

**프로젝트 추론:** object OBB/Gaussian, 소수 representative crop, compact descriptor를 entity record로 저장하고 모든 pairwise relation을 영구 저장하지 않는 설계의 강한 근거다. Language model은 query parsing에 쓰고 metric predicate는 deterministic executor가 계산하는 분리도 직접 참고할 수 있다.

## 근거 상태

**프로젝트 결과:** 미재현; official code 사용 불가. 현재 프로젝트의 geometry executor는 같은 설계 방향을 갖지만 FARM 결과가 아니다.

## 논문 핵심

**논문 주장:** object마다 single 3D Gaussian, compact visual-language evidence, lightweight base relations만 저장하고, `left-of`, `between`, `near` 같은 관계는 query time에 executable predicates로 계산하면 large-scale relational object retrieval을 real-time으로 지원할 수 있다.

## 근거

**논문 보고 결과:**

- 67개 indoor·outdoor scene, `44,031` language query, `15-15,000 m²` 범위를 평가했다. 출처: Section 3.1.
- Prior method 대비 Recall@5 `+164%`, Recall@10 `+224%`, final VLM reranking으로 Accuracy@1 `+35%`의 상대 개선을 보고했다. 출처: abstract와 Table 1.
- Mapping은 약 `8 Hz`, saved representation은 ScanNet scene당 약 `23 MiB`, HM3D scene당 약 `125 MiB`로 보고했다. 출처: Table 1(b).
- Quadruped robot의 onboard closed-loop deployment를 제시했다.

## 판단 한계

- Posed RGB-D가 아닌 1 Hz monocular RGB+IMU 조건
- General geometry QA, temporal QA, navigation QA 전체
- Learned write-time selection 또는 QA utility 기반 hard byte budget
- Lifelong revisit consolidation과 dynamic change history
- `0.1-0.4 MB/submap` 목표 달성
- SuperMemory-VQA 성능

논문은 hand-specified·uncalibrated predicate와 fixed weights를 한계로 명시한다. 주된 query 구조도 target-anchor star relation이며 anchor-anchor compositional reasoning은 지원하지 않는다.

## 문제 배경

Category나 appearance가 유사한 object가 많은 환경에서는 label lookup만으로 목표 instance를 찾기 어렵다. 사용자는 landmark와 주변 object의 관계로 목표를 설명하므로, compact object memory와 compositional spatial retrieval이 함께 필요하다.

## 관련 방법

- Streaming posed RGB-D에서 detect, 3D lift, associate, fuse loop를 수행한다.
- Entity마다 single 3D Gaussian, association feature, 최대 `k`개 posed crop, caption, text·image retrieval embedding을 저장한다.
- Mapping time에는 covisibility와 adjacency만 저장한다. Containment, left-of, between 같은 고차 관계는 query time에 계산한다.
- Caption과 embedding 생성은 asynchronous worker로 mapping critical path에서 분리한다.
- LLM이 query를 typed target-anchor-predicate graph로 compile하고, closed-form soft spatial predicates로 binding을 평가한 뒤 top candidate만 VLM으로 rerank한다.

## 참고문헌

- [arXiv paper](https://arxiv.org/abs/2606.15476)
- [공식 project page](https://goldengait.github.io/farm/)
- [상위 논문 목록](README.md)
