# VisionZip: Longer is Better but Not Necessary in Vision Language Models

| Field | Value |
|---|---|
| Page ID | SM-PAPER-VISIONZIP |
| Status | Reviewed |
| Authors | Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, Jiaya Jia |
| Publication | CVPR 2025 |
| Version checked | arXiv:2412.04467v2, 2026-03-15 |
| Primary source | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html) · [arXiv](https://arxiv.org/abs/2412.04467) |
| Official code | [JIA-Lab-research/VisionZip](https://github.com/JIA-Lab-research/VisionZip) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-011 |

## 30-Second Summary

VisionZip는 언어 모델 처리 전에 중복된 시각적 토큰을 줄이다. language model이 더 짧은 시각적 시퀀스를 수신하도록 지배적인 토큰을 유지하고 상황별 정보를 병합한다. 이는 조기 중복성 감소를 지원하지만 출력은 명시적이고 감사 가능한 공간 데이터베이스가 아닌 VLM 입력 표현으로 유지된다.

## Problem Addressed

최신 비전 언어 모델은 시각적 토큰 길이를 늘려 부분적으로 품질을 향상시켜 사전 채우기 및 생성 비용을 높이다. CLIP 및 SigLIP 파생 시각적 시퀀스에는 상당한 중복성이 포함되어 있다.

## Relevant Method

- 전 세계적으로 중요한 시각적 정보를 전달하는 주요 토큰을 식별한다.
- 단순히 비주위 토큰을 모두 삭제하는 대신 다른 토큰 콘텐츠를 병합하여 상황별 정보를 보존한다.
- 이미지, 비디오, 멀티턴 VLM 용도로 압축 표현을 적용한다.

## Paper-Reported Evidence

이 논문에서는 거의 모든 테스트된 압축 설정에서 이전 기술 수준에 비해 최소 5%의 이득이 있고 사전 충전 시간이 8배 향상되었다고 보고한다. 또한 압축된 LLaVA-NeXT-13B가 LLaVA-NeXT-7B보다 빠르게 실행되면서 더 나은 평가 결과를 얻을 수 있다고 보고한다.

이는 이 저장소에서 재현한 결과가 아닌 논문 결과이다.

## What This Supports Here

- 입력된 후보 decoder 이전에 중복 공급자 기능을 줄이다.
- 상황별 증거에 대한 병합 및 요약 정책과 삭제를 비교한다.
- persistent memory 바이트와 별도의 비용으로 VLM 미리 채우기를 포함한다.

## What It Does Not Prove

- 지배적인 시각적 토큰은 object identity 또는 미터법 좌표를 유지한다.
- token merging는 uncertainty 및 provenance 구성을 유지한다.
- VLM 벤치마크 품질은 기하학적 기반 QA 품질을 예측한다.
- 일시적인 입력 압축은 평생 메모리 증가를 제한한다.

## Project Reproduction Status

재현되지 않았다. 저장소는 VisionZip를 구현하지 않는다. 현재 압축은 형상 추출 후 명시적 레코드 후보에 대해 작동하므로 다른 표현과 목적을 테스트한다.

## References

- [Paper index](README.md)
- [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html)
- [arXiv:2412.04467](https://arxiv.org/abs/2412.04467)
- [Official code](https://github.com/JIA-Lab-research/VisionZip)
