# Spann3R: 3D Reconstruction with Spatial Memory

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-SPANN3R |
| 상태 | 검토 완료; code 사용 가능 |
| 출판 | IEEE 3DV 2025, pp. 78–89; arXiv:2408.16061 v1 |
| 1차 출처 | [DOI](https://doi.org/10.1109/3DV66043.2025.00013) · [arXiv](https://arxiv.org/abs/2408.16061) · [Project page](https://hengyiwang.github.io/projects/spanner) |
| 공식 code | [HengyiWang/spann3r](https://github.com/HengyiWang/spann3r) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-002 |

## 핵심 결론

- 상세한 단기 기하학을 희박한 장기 맥락에서 분리한다.
- 기하학의 참신함과 미래의 QA 유틸리티에 대한 주의 기반 유지를 테스트한다.
- 공간 전환 및 긴 시퀀스 전반에 걸쳐 메모리 및 재구성 실패를 측정한다.

## 근거 상태

재현되지 않았다. 릴리스된 코드는 이를 실용적인 임시 메모리 기준으로 만듭니다. 모든 재생산에서는 게시된 모델을 저장소의 이후 v1.01 체크포인트 및 교육 변경 사항과 구별해야 한다.

## 논문 핵심

Spann3R는 최근의 조밀한 working memory 및 희박한 장기 기능 메모리를 관리하면서 전체적으로 정렬된 포인트맵을 점진적으로 예측한다. 이는 "최근 세부 사항과 선택된 기록"에 대한 초기의 구체적인 기준을 제공한다. 주의 순위가 높은 일반 토큰은 명시적인 기하학적 기반 QA 레코드가 아닌 재구성을 지원한다.

## 근거

**보고된 주장.** 메모리 제거에서는 working memory만 있는 `0.2554/0.1470/0.5964`의 Acc/Comp/NC와 전체 메모리가 있는 `0.0342/0.0241/0.6635`를 보고한다. 이 논문에서는 대부분의 테스트 장면에 4,000개의 장기 토큰이 충분하며 하나의 RTX 4090에 약 11개의 GB VRAM가 있는 약 65개의 FPS가 있다고 보고한다.

**프로젝트 추론.** 검색 주의가 QA 유틸리티로 잘못 표시되지 않는 한 최근의 조밀한 버퍼와 선택된 과거 저장소는 임시 기하학에 대한 유용한 기준선이다.

**프로젝트 결과.** 없음. 이 저장소는 Spann3R를 재현하지 않았다.

## 판단 한계

- 명시적 객체, 평면, 포털, 이벤트, 좌표계 또는 provenance 저장소.
- 재구성 주의는 미래의 QA 값 또는 바이트당 직렬화된 값을 예측한다.
- 강력한 loop closure 또는 멀티룸 평생 매핑; 이 논문은 표류와 관련된 한계를 보고한다.
- SuperMemory-VQA 개선, 1Hz 웨어러블 성능 또는 AI-유리 타당성.
- 저장소의 이후 v1.01 체크포인트는 3DV 문서 결과로 나타납니다. 그것들은 논문 이후의 릴리스 노트이다.

## 문제 배경

쌍별 DUSt3R 포인트맵은 전체 장면을 사용하기 전에 전역 정렬이 필요하다. 대신 Spann3R는 하나의 전역 좌표계에서 각 프레임을 예측하고 카메라 입력이나 테스트 시간 전역 정렬 없이 이전 3D 정보를 온라인으로 검색한다.

## 관련 방법

- 최근 5개 프레임에 대해 조밀한 working memory를 유지한다.
- 오래된 정보를 희소 long-term memory로 이동한다.
- 유사성이 임계값 미만인 경우에만 작업 메모리 토큰을 삽입한다.
- 누적된 검색 관심을 기준으로 장기 토큰의 순위를 매기고 임계값에 도달한 후에도 상위 k를 유지한다.
- 두 메모리를 모두 쿼리하여 전역적으로 정렬된 다음 포인트맵을 예측한다.

## 참고문헌

- 왕헝이(Hengyi Wang)와 루르드 아가피토(Lourdes Agapito). [3D Reconstruction with Spatial Memory](https://doi.org/10.1109/3DV66043.2025.00013). IEEE 3DV 2025, 78-89페이지.
- [공식 arXiv record](https://arxiv.org/abs/2408.16061).
- [공식 project page](https://hengyiwang.github.io/projects/spanner).
- [공식 repository](https://github.com/HengyiWang/spann3r).

[논문 근거 목록으로 돌아가기](README.md)
