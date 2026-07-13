# Spatial Memory 운영

| 항목 | 값 |
|---|---|
| Page ID | SM-OPERATIONS |
| Confluence parent | SM-ROOT |
| Child Page ID | SM-OPERATIONS-HANDOFF |
| Child source | Repository `HANDOFF.md` |
| 상태 | 실행 준비 중 |
| 최종 검토 | 2026-07-13 |

## 실행 방향

Repository `HANDOFF.md`가 유일한 runbook이다. Confluence에서는
`SM-OPERATIONS-HANDOFF`로 import하며 이곳에 절차를 복사하거나 분기하지 않는다.

현재 운영 결론은 공식 training/evaluation **No-Go**다. 명시적으로 승인된
1-node × 1-GPU contract probe만 다음 실행 요청으로 허용한다.

## 실행 전 gate

1. [현재 상태](../status.md)에서 live decision을 확인한다.
2. [아키텍처](../architecture.md)에서 control을 확인한다.
3. [실험](../experiments/README.md)에서 exact experiment contract를 확인한다.
4. 승인, submission, monitoring, cancellation, artifact 처리는 `HANDOFF.md`를
   따른다.

[프로젝트 홈으로 돌아가기](../README.md)
