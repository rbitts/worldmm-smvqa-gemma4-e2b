# 온디바이스 멀티모달 메모리 런타임 제안

| 항목 | 값 |
|---|---|
| Page ID | SM-ONDEVICE-RUNTIME |
| 상태 | 제안 |
| 기준일 | 2026-07-16 |
| 요청 | 공통 runtime framing과 제한된 1단계 검증 승인 |

## 한 줄 결론

새 범용 모델보다 먼저, 기존 인식·OCR·VLM·QA 모델이 공통으로 사용하는
**온디바이스 멀티모달 메모리 런타임**을 검증한다. 핵심은 두 가지다.

1. `Write`: 제한된 byte·전력 안에서 무엇을 원본, 증거, 사건, 지식으로 남길지 결정
2. `Read`: 질문이나 현재 이벤트에 맞춰 필요한 기억을 routing·retrieval·검증

QA는 첫 consumer 중 하나일 뿐이다. Visual Routine, recall/search, 향후 navigation도
동일 runtime을 사용한다. 승인 대상은 제품 전체가 아니라 고정 byte 조건의
1단계 가치 검증이다.

## 왜 필요한가

지속 관측 기기는 모든 frame과 sensor를 장기간 보관할 수 없다. 즉시 caption이나
semantic summary만 남기면 미래 질문·routine에 필요한 작은 물체, 정확한 글자,
행동 순서, 마지막 위치가 사라진다. Visual embedding은 원본의 가역 압축이 아니므로
삭제된 frame을 다시 OCR할 수 없다.

반대로 모든 처리를 cloud로 보내면 개인정보, 네트워크 의존성, 비용과 지연이
늘어난다. 필요한 것은 또 하나의 답변 모델이 아니라 원본의 짧은 보존, 장기 증거
선별, 파생 정보의 근거 연결, consumer별 검색 권한을 함께 운영하는 runtime이다.

## 제품 정의

| Runtime이 책임지는 것 | Consumer가 책임지는 것 |
|---|---|
| 관측 ingest, 시간·source·privacy 표기 | 질문 해석과 최종 문장 생성 |
| byte/energy 기반 보존·승격·통합·삭제 | Routine 정의와 task state |
| visual·episodic·semantic·spatial index | 알림·확인·외부 행동 결정 |
| request별 routing, retrieval, reranking | Consumer별 UI와 사용자 경험 |
| causal·권한·원본 존재·완전성 검증 | 고위험 행동에 대한 추가 승인 |
| evidence와 부족 정보를 담은 ContextPack | 답변, 침묵, 알림 또는 abstention |

Routine rule을 semantic memory에 저장하지 않는다. `Routine Registry`라는 별도
control state로 두고, runtime에서 사실 evidence만 읽는다. Memory read 권한과
외부 action 권한도 분리한다.

## Overall architecture

```text
Camera · Mic · IMU/VIO · App event
               |
        [Sensor Gateway]
               |
      +--------+------------------+
      |                           |
[Hot Evidence Ring]        [Fast Trigger/Perception]
 bounded raw source          scene·object·OCR·motion·routine cue
      |                           |
      +------------+--------------+
                   v
            [Memory Compiler]
        Visual | Episodic | Semantic | Spatial
                   |
       [Retention/Promotion Controller]
       keep · promote · merge · expire · delete
                   |
     temporal · vector · graph · spatial · provenance index
                   ^
                   |
 Question or event -> [Request Gateway + Query Planner]
                   -> parallel retrieval / refine / STOP
                   -> [Evidence Verifier]
                   -> ContextPack
                         +--> QA: answer / abstain
                         +--> Visual Routine: silent / notify / confirm
                         +--> Future: search / navigation / summary
```

항상 켜지는 것은 bounded buffer와 저비용 trigger다. Heavy captioning, embedding,
consolidation과 reasoning은 이벤트가 있거나 충전·idle 조건일 때만 실행한다.
접근제어와 causal cutoff는 learned router보다 먼저 deterministic하게 적용한다.

상세 component, record/request contract, failure control은
[멀티모달 메모리 런타임 아키텍처](memory-runtime-architecture.md)에 정의한다.

## 두 동작 pipeline

### 1. Pull consumer: QA

```text
질문 -> 시간/entity/operation 추출 -> 관련 memory route
     -> visual·episodic·semantic·spatial 병렬 검색
     -> 원본 존재·인과성·충돌·완전성 검증
     -> evidence pack -> cited answer 또는 근거 부족
```

정확한 글자나 외형 질문은 source pixel 또는 보존된 OCR span이 필요하다. 요약만
남았다면 “관련 정보는 있으나 정확한 답은 복구할 수 없음”으로 끝낸다.

### 2. Event-driven consumer: Visual Routine

```text
현재 관측 -> cheap trigger -> 후보 routine match
         -> 필요한 과거 기억만 retrieve
         -> condition: satisfied | violated | unknown
         -> action gate: silent | notify | ask confirmation
```

예: “현관을 나갈 때 열쇠를 두고 가면 알려줘.” 문 통과 신호가 routine을 깨우고,
최근 pickup event, 열쇠의 마지막 spatial state, 필요한 visual evidence만 검색한다.
현재 frame에서 열쇠가 보이지 않는다는 이유만으로 부재를 단정하지 않는다.
근거와 관측 coverage가 충분할 때만 알린다.

MVP는 `silent`, `notify`, `ask confirmation`만 허용한다. 메시지 발송, 구매, 문 제어
같은 외부 action은 초기 범위에서 제외한다.

## Retention 정책 가설

| Tier | 초기 horizon | 남기는 정보 | 보장 경계 |
|---|---:|---|---|
| H0 Hot | 0~1시간 | 허용된 raw/sample frame, audio, sensors | 원본이 실제 남은 범위만 재-OCR·재인식 가능 |
| H1 Evidence | 1시간~1일 | 선택 clip/crop, OCR/transcript, 사건, 위치 상태 | 선택 evidence 안에서 explicit 확인 가능 |
| H2 Consolidated | 1일~1주 이상 | 사건 요약, supported fact, 안정된 spatial state, 소수 핵심 evidence | 삭제된 세부는 unknown |
| Pinned | 사용자 정책까지 | 사용자가 명시적으로 고정한 source/record | 별도 권한·용량 정책 적용 |

`1시간/1일/1주`는 제품 약속이 아니라 검증 시작값이다. 별도 세 DB가 아니라 같은
record 위 retention profile로 구현한다. Migration은 파생 record와 support hash를
검증한 뒤에만 원본을 만료한다.

## 근거와 프로젝트 해석

- [WorldMM](https://arxiv.org/abs/2512.02425)은 episodic·semantic·visual memory와
  adaptive multi-step retrieval로 저자 보고 기준 5개 long-video QA benchmark에서
  이전 SOTA 대비 평균 8.4% 향상을 보였다. On-device 저장·전력 결과는 제공하지
  않아 알고리즘 참고선으로만 사용한다.
- [MemoLens, MobiSys 2026](https://sigmobile.org/mobisys/2026/accepted_papers/)은
  selective write, hierarchical memory, coarse-to-fine retrieval을 실제
  glasses·phone·cloud system에서 평가했다. 순수 device-only 근거는 아니다.
- [Memory-Centric EQA](https://arxiv.org/abs/2505.13948)와
  [ReMEmbR](https://arxiv.org/abs/2409.13682)은 memory retrieval이 답변뿐 아니라
  planning·stopping·navigation 목표에도 쓰일 수 있음을 보였다.
- [Vinci2](https://arxiv.org/abs/2607.11523)는 streaming memory를 proactive
  intervention에 연결한다. 2026-07 신규 preprint이므로 방향 근거이지 제품 검증은
  아니다.

따라서 논문을 그대로 재현하는 과제가 아니다. 여러 memory 연구를 bounded
lifecycle, provenance, permissions, multiple consumers, hardware budget으로 묶는
운영 architecture가 프로젝트 가설이다.

## 현재 자산과 gap

| 보유 | 미보유 |
|---|---|
| typed visual/episodic/semantic/spatial schema | encrypted hot ring과 tier migration |
| causal scope, byte cap, evidence lineage | 실제 vector/graph retrieval과 online consolidation |
| deterministic spatial proof와 abstention | Consumer ACL과 deletion propagation |
| QA adapter와 frame evidence 경로 | Routine Registry, trigger, action gate |
| local tiny-fixture 검증 | target-device latency·power·thermal 실측 |

현재 일부 routing은 keyword/lexical heuristic이다. `embedding_ref`가 존재해도 실제
vector retrieval이 구현됐다는 뜻은 아니다. Published WorldMM agent loop나
on-device 제품 성능을 재현했다고 주장하지 않는다.

## 성공 기준

| 영역 | 1단계에서 확인할 지표 |
|---|---|
| Memory | 시간당 byte, source 보존율, promotion/expiry rate |
| Retrieval | Recall@K, first-evidence latency, bytes/query, causal·ACL violation |
| QA | 경과시간별 정확도, unsupported-answer, risk-coverage |
| Visual Routine | false interruption, missed assist, time-to-assist, dismiss rate |
| Device 2단계 | p95 latency, power, thermal, RAM/flash peak, off-device bytes |

비교 baseline은 `recent-only ring`, `uniform sampling`, `summary-only`, `proposed
tiered memory`다. 같은 source와 총 byte budget을 사용한다. QA만 개선하고 Routine
evidence를 악화시키면 공통 runtime 가설은 통과하지 못한다.

## 투자 단계와 중단 조건

| 단계 | 수행 범위 | 다음 단계 조건 |
|---|---|---|
| 1. Offline value | 기존 승인 데이터로 고정-byte QA·Routine replay | 두 consumer의 retrieval/value 개선, 위험 증가 없음 |
| 2. Target device | 기기에서 저장·지연·전력·thermal·privacy 측정 | 합의된 device budget 충족 |
| 3. Limited pilot | opt-in routine과 recall 사용성 검증 | 사용자 가치와 false-interruption 기준 충족 |

학습형 retention/router는 rule baseline이 병목으로 확인된 경우에만 검토한다.

## 이번 승인 범위에서 제외

- 새 foundation model과 범용 assistant 개발
- 모든 원본 영상의 장기 보관
- 처음부터 end-to-end 학습하는 memory controller
- 범용 natural-language routine compiler
- 사용자 확인 없는 외부 action 자동 실행
- 목표 기기 측정 전 on-device 완료 주장
- 삭제된 근거에 대한 추측성 답변

## 요청 의사결정

1. 프로젝트를 **QA용 memory가 아닌 공통 온디바이스 메모리 런타임**으로 재정의한다.
2. 1단계 consumer를 `grounded QA + opt-in Visual Routine 알림`으로 정한다.
3. 첫 target device와 대표 관측 시나리오 하나를 지정한다.
4. 같은 byte budget의 네 baseline을 비교하는 1단계만 승인한다.

1단계 결과 전에는 별도 대형 모델 개발, multi-node training, 제품화 투자를
시작하지 않는다.
