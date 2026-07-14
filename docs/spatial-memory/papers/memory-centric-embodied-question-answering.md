# Memory-Centric Embodied Question Answering

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-MEMORYEQA |
| 상태 | 검토 완료; preprint; code 사용 가능 |
| 출판 | arXiv:2505.13948 v2, 2025-12-13; peer-reviewed venue 미확인 |
| 1차 출처 | [Version-pinned arXiv v2](https://arxiv.org/abs/2505.13948v2) |
| 공식 code | [memory-eqa/MemoryEQA](https://github.com/memory-eqa/MemoryEQA) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-002, C-004, C-008 |

## 핵심 결론

- 기억은 최종 답변뿐만 아니라 검색, 중지 및 계획을 알려야 한다.
- 중복 관찰에는 명시적인 업데이트 결정이 필요하다.
- 선택적 모듈별 검색은 메모리를 추가하는 것보다 더 중요할 수 있다.
- 적응형 검색 크기는 바이트당 값 기록기에 대한 관련 기준이다.

## 근거 상태

재현되지 않았다. 공식 프로젝트와 저장소에서 충돌하는 헤드라인 번호를 보고했기 때문에 향후 비교에서는 arXiv v2와 해당 테이블을 고정해야 한다.

## 논문 핵심

MemoryEQA는 응답 시간에만 메모리를 참조하는 대신 계획, 중지 및 응답에 메모리를 사용할 수 있도록 한다. 절제는 선택적 검색과 적응형 검색 크기가 업데이트 게이팅보다 더 많은 기여를 한다는 것을 보여준다. 이는 메모리 인식 제어 및 검색을 지원하지만 구조화된 텍스트 및 밀집 벡터 라이브러리는 명시적인 metric geometry 또는 실제 바이트 압축이 아니다.

## 근거

아래의 모든 값은 arXiv v2에 고정되어 있다. 다른 공식 프로젝트 표면에는 오래되고 상충되는 가치가 포함되어 있어 논문 결과로 사용되지 않는다.

| Dataset | 조건 | Metric | 논문 보고 결과 | 위치 |
|---|---|---|---|---|
| MT-HM3D | Dataset | Scale | 1,587 QA samples over 500 scenes | Section 4 |
| MT-HM3D | GPT-4o ExploreEQA → MemoryEQA | Success | 33.21 → 43.11, +9.90 percentage points | Table 2 |
| MT-HM3D | No strategy → update → update + retrieval → all + adaptive k | Success | 33.18 → 33.41 → 39.69 → 41.95 | Table 3 |
| MT-HM3D | No module injection → stop → stop + answer → stop + answer + planner | Success | 30.22 → 35.10 → 40.99 → 41.95 | Table 4 |
| HM-EQA | GPT-4o ExploreEQA → MemoryEQA | Success | 47.40 → 61.40 | Table 2 |

## 판단 한계

- 명시적인 유형의 메트릭 레코드 또는 결정론적 기하학 증명.
- 실제 직렬화된 바이트 최적화 또는 제한된 평생 성장.
- 1Hz 희소 단안 감지, IMU/VIO 안내 또는 AI-안경 실행.
- 그 엔트로피는 미래의 기하학 기반 질문에 대한 최선의 선택이다.
- SuperMemory-VQA 성능.

## 문제 배경

플래너 중심의 구현된 QA 시스템은 응답 모듈에서만 메모리를 사용하기 때문에 중복 탐색을 수행하거나 너무 일찍 중지할 수 있다. 여러 공간에 걸친 다중 대상 질문에는 에이전트가 관찰 내용을 유지하고, 올바른 하위 집합을 검색하고, 해당 하위 집합을 모든 결정 모듈에 노출해야 한다.

## 관련 방법

- 관찰 내용을 벡터 라이브러리에 저장된 구조화된 텍스트 및 다중 모드 기능으로 변환한다.
- 동일한 환경의 이후 작업에서 재사용할 수 있도록 장면 메모리를 유지한다.
- 위치, 방향, 구조적, 의미적 유사성, 시야 검사를 사용하여 게이트를 업데이트한다.
- 플래너, 중지, 쿼리 응답을 위한 다양한 메모리를 검색한다.
- 쿼리 기능 엔트로피를 사용하여 검색 임계값과 상위 k 크기를 조정한다.
- 여러 객체와 지역에 대한 비교, 관계, 계산 및 속성 질문을 사용하여 MT-HM3D를 구축한다.

## 참고문헌

- Mingliang Zhai, Zhi Gao, Yuwei Wu, Yunde Jia. [Memory-Centric Embodied Question Answering](https://arxiv.org/abs/2505.13948v2). arXiv:2505.13948 v2.
- [공식 project page](https://memory-eqa.github.io/).
- [공식 repository](https://github.com/memory-eqa/MemoryEQA).
- [Official MT-HM3D dataset](https://huggingface.co/datasets/zmling/MT-HM3D).
- [논문 근거 목록으로 돌아가기](README.md).
