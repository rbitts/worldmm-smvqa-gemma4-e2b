# TTT3R: 3D Reconstruction as Test-Time Training

| 항목 | 값 |
|---|---|
| Page ID | SM-PAPER-TTT3R |
| 상태 | 검토 완료; code 사용 가능 |
| 출판 | ICLR 2026; arXiv:2509.26645 v4 |
| 1차 출처 | [OpenReview](https://openreview.net/forum?id=aMs6FtNaY5) · [arXiv](https://arxiv.org/abs/2509.26645) · [Project page](https://rover-xingyu.github.io/TTT3R/) |
| 공식 code | [Inception3D/TTT3R](https://github.com/Inception3D/TTT3R) |
| 최종 확인 | 2026-07-11 |
| 프로젝트 연결 | [상위: 논문 목록](README.md) · [아키텍처](../architecture.md) · [추적성](../traceability.md) |
| 프로젝트 claim | [추적성](../traceability.md): C-006 |

## 핵심 결론

- 포즈 및 정렬 신뢰도를 사용하여 작업 상태 업데이트를 규제한다.
- 내구성 있는 QA 레코드를 별도로 컴파일하는 동안 일정한 크기의 임시 상태를 유지한다.
- 동일한 교사 아래에서 균일하고 신뢰도가 높은 시간적 공간적 이벤트 인식 업데이트 정책을 테스트한다.

## 근거 상태

재현되지 않았다. 작업 메모리 업데이트 기준으로만 사용한다. 재설정하면 장기적인 결과의 의미가 변경되므로 재설정 정책을 명시적으로 기록한다.

## 논문 핵심

TTT3R는 CUT3R의 지속 상태를 테스트 시간 빠른 가중치로 해석하고 상태 관찰 정렬 신뢰도에서 토큰별 업데이트 속도를 도출한다. 이는 경량 작업 상태 안정화 기준이다. 이는 명시적인 long-term memory를 대체하지 않으며 1,000프레임을 초과하는 작성자의 시연을 위해 여전히 재설정이 필요하다.

## 근거

**보고된 주장.** 초록에서는 수천 개의 이미지를 처리하는 동안 기준선, `20 FPS` 및 `6 GB` GPU 메모리에 비해 전역 pose estimation의 `2×` 개선이 보고됐다. 프로젝트 페이지에는 1,000프레임을 초과하는 시연이 100프레임마다 상태를 재설정하고 결과 청크를 예측된 전역 메트릭 카메라 포즈와 정렬한다고 명시되어 있다.

**프로젝트 추론.** 신뢰도 규모 업데이트는 특히 1Hz 간격으로 인해 정렬이 불확실해지는 경우 과도 recurrent state에 대한 유효한 기준선이다.

**프로젝트 결과.** 없음. 이 저장소는 TTT3R를 재현하지 않았다.

## 판단 한계

- 무제한 보존: 저자는 망각이 1,000프레임 이상으로 유지된다고 말한다.
- 엔터티 ID, 측정항목 증명, 시간적 유효성 또는 provenance가 포함된 명시적인 공간 데이터베이스이다.
- 지속적인 직렬화된 바이트 압축 또는 미래 QA-aware 선택.
- SuperMemory-VQA 성능, 1Hz AI-유리 견고성 또는 기기 내 전력 타당성.

## 문제 배경

고정 상태 반복 재구성에는 선형 추론 비용이 있지만 훈련 컨텍스트 길이 이상으로 성능이 저하된다. 새로운 관찰이 제대로 정렬되지 않은 경우 균일 업데이트가 기록을 덮어씁니다. TTT3R는 새 모델을 교육하지 않고 신뢰도가 낮은 업데이트를 억제한다.

## 관련 방법

- recurrent state 업데이트를 온라인 연관 학습으로 재해석한다.
- 상태-관측 cross-attention 정렬을 신뢰도로 사용한다.
- 상태 전환에 대한 폐쇄형 토큰별 학습률을 도출한다.
- 학습된 매개변수를 추가하지 않고 frozen model 가중치를 사용하여 정방향 패스에서 업데이트를 적용한다.

## 참고문헌

- Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger 및 Anpei Chen. [TTT3R: 3D Reconstruction as Test-Time Training](https://openreview.net/forum?id=aMs6FtNaY5). ICLR 2026.
- [공식 arXiv record](https://arxiv.org/abs/2509.26645).
- [공식 project page](https://rover-xingyu.github.io/TTT3R/).
- [공식 repository](https://github.com/Inception3D/TTT3R).

[논문 근거 목록으로 돌아가기](README.md)
