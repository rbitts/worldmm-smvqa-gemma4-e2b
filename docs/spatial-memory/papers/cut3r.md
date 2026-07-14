# CUT3R: Continuous 3D Perception Model with Persistent State

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-CUT3R |
| 상태 | Primary source 검토 완료; project 미재현 |
| 출판 | CVPR 2025 Oral, pp. 10510–10522 |
| 1차 출처 | [CVPR Open Access paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html) |
| 공식 code | [CUT3R/CUT3R](https://github.com/CUT3R/CUT3R) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [논문 목록](README.md), [아키텍처](../architecture.md), [ADR-0002](../decisions/adr-0002-gcut3r-as-teacher.md) |
| Project claims | [Traceability](../traceability.md): C-001, C-006 |

## 핵심 결론

**논문 주장:** 반복적인 이미지-상태 상호 작용은 희소 보기 조건을 포함하여 공통 프레임에서 온라인 카메라 및 dense geometry 추정치를 제공할 수 있다.

**프로젝트 추론:** CUT3R는 희박한 관찰을 유형이 지정된 개체, 구조, 랜드마크 및 이벤트 후보로 변환하는 데 적합한 임시 geometry teacher 또는 프런트 엔드이다. 해당 상태는 생성 중에 소비되어야 하며 모든 타임스탬프에서 long-term memory로 직렬화되지 않아야 한다.

공통 프레임 포인트 맵, 포즈 추정 및 신뢰도 출력은 계획된 유형의 decoder 및 연관 모델에 유용한 교사 신호이다.

## 근거 상태

프로젝트는 CUT3R 파생 외부 교사에 대한 공급자/캐시 경계를 정의하지만 CUT3R를 로컬로 공급, 다운로드 또는 실행하지 않는다. 아직 프로젝트 체크포인트, 공식 데이터세트 재구성 결과 또는 CUT3R에서 입력된 레코드로의 재생산은 없다.

## 논문 핵심

CUT3R는 온라인 반복 3D 인식 모델이다. 들어오는 각 RGB 이미지는 영구 latent state에서 읽고 업데이트한 다음 카메라와 공유 세계 좌표 모두에서 camera pose 및 밀집 지점 맵을 예측한다. 가상 광선 맵 쿼리는 상태를 읽어 관찰되지 않은 뷰를 예측할 수 있다. 이 모델은 제공된 camera intrinsics 또는 포즈 없이 정렬된 비디오, 정렬되지 않은 사진, 정적 또는 동적 장면을 처리한다.

## 근거

이는 이 저장소의 결과가 아닌 논문 결과이다.

| Dataset 또는 조건 | Metric | 논문 보고 결과 | 출처 위치 |
|---|---|---:|---|
| KITTI video depth, 512 by 144, A100 | Online throughput | 16.58 FPS | Table 2, paper p. 5 |
| KITTI video depth, per-sequence scale | Abs Rel / delta below 1.25 | 0.118 / 88.1 | Table 2, paper p. 5 |
| 7-Scenes, sparse 3–5 frames | Mean accuracy / completeness / normal consistency | 0.126 / 0.154 / 0.727 | Table 4, paper p. 7 |
| NRGBD, sparse 2–4 frames | Mean accuracy / completeness / normal consistency | 0.099 / 0.076 / 0.837 | Table 4, paper p. 7 |
| 7-Scenes, online then frozen-state revisit | Mean accuracy, before versus after revisit | 0.126 versus 0.113 | Table 5, paper p. 8 |

재방문 실험에서는 7-Scenes의 평균 완성도가 0.154에서 0.107로 향상되었다고 보고했다. 이는 recurrent state가 추가 관찰 후 예측을 개선할 수 있다는 논문의 더 좁은 주장을 뒷받침한다.

## 판단 한계

- latent state는 엔터티, 관계 또는 시간적 유효성 간격에 대한 명시적인 데이터베이스가 아니다.
- 가상 뷰 출력은 직접 관찰이 아닌 모델 추론이므로 다른 provenance를 전달해야 한다.
- 고정 상태 형태는 안정적인 평생 유지나 제한된 의미 망각을 증명하지 못한다.
- 게시된 GPU 처리량은 AI-glasses 배포 가능성을 설정하지 않는다.
- 이 논문는 SuperMemory-VQA, 1Hz 연간 메모리, 실제 바이트 저장소 또는 기하학적 기반 QA를 평가하지 않는다.

## 문제 배경

쌍별 재구성 모델은 많은 뷰를 결합하기 위해 전역 정렬이 필요한 반면, 기존 SfM 및 SLAM는 희박한 중첩, 동적 콘텐츠 또는 퇴화 동작으로 인해 실패할 수 있다. CUT3R는 사전에 학습된 장면과 고정 모양 recurrent state를 사용하여 지속적인 온라인 재구성을 목표로 한다.

## 관련 방법

image encoder는 현재 프레임에 대한 토큰을 생성한다. 상호 연결된 두 개의 transformer 디코더는 교차 주의를 통해 state update 및 상태 판독을 구현한다. 출력 헤드는 카메라 프레임 포인트 맵, 월드 프레임 포인트 맵, 신뢰도 및 포즈를 예측한다. 별도의 광선 맵 encoder는 가상 뷰 예측을 위해 업데이트하지 않고 상태를 쿼리한다.

게시된 구현에서는 ViT-L image encoder, ViT-B 디코더, 16 x 16 패치 및 차원 768의 768개 상태 토큰을 사용한다. 교육은 32개 데이터세트에 걸쳐 4개 뷰 시퀀스에서 64개 뷰까지 시퀀스로 진행된다.

## 참고문헌

- [CVPR 2025 Open Access record](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html)
- [공식 project page](https://cut3r.github.io/)
- [공식 code](https://github.com/CUT3R/CUT3R)
- [공식 arXiv record](https://arxiv.org/abs/2501.12387)

[논문 근거 목록으로 돌아가기](README.md)
