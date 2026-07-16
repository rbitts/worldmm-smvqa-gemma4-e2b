# 온디바이스 멀티모달 메모리 런타임 아키텍처

| 항목 | 값 |
|---|---|
| Page ID | SM-MEMORY-RUNTIME-ARCH |
| 상태 | 목표 아키텍처 제안 |
| 기준일 | 2026-07-16 |
| 범위 | 공통 memory write/read runtime과 QA·Visual Routine consumer |
| 구현 상태 | 일부 primitive만 로컬 구현; lifecycle·routine은 미구현 |

## 핵심 결정

이 시스템의 제품 단위는 QA 모델이 아니라 **다중 consumer가 공유하는 로컬
멀티모달 메모리 런타임**이다.

- Write runtime은 관측을 보존·요약·통합·삭제할 대상을 고른다.
- Read runtime은 consumer의 목적, 시간, 지연, 위험도에 맞춰 기억을 검색한다.
- QA는 사용자가 질문할 때 동작하는 pull consumer다.
- Visual Routine은 현재 관측이 조건을 만족할 가능성이 있을 때 동작하는
  event-driven consumer다.
- Spatial Memory는 위치·거리·상태 전이를 제공하는 전문 memory view다.
- 최종 문장 생성과 행동 실행은 runtime 밖 consumer가 담당한다.

Routing만 구현해서는 부족하다. Write 시점에 미래에 필요한 시각 증거를 버리면
어떤 retriever도 복구할 수 없다. 따라서 `Retention/Promotion Controller`와
`Query Planner/Retrieval`을 동등한 핵심으로 둔다.

## 제품 경계

```text
인식/OCR/VLM/SLAM provider                  Consumer
          |                         +------------------------+
          v                         | QA                     |
  +--------------------+            | Visual Routine         |
  | Memory Runtime     |----------->| Search / Recall        |
  | write, retain,     | ContextPack| Navigation (future)    |
  | index, retrieve,   |            +------------------------+
  | verify, delete     |                       |
  +--------------------+                       v
                                           Answer / Notify /
                                           Confirm / Act
```

Runtime은 foundation model, 완성된 assistant, routine planner가 아니다. 모델은
관측을 record로 만드는 provider 또는 ContextPack을 소비하는 adapter다. 교체돼도
memory contract와 보존 정책은 유지된다.

Routine 정의, task progress, cooldown은 `Routine Registry`의 control state다.
관측 사실인 episodic/semantic/spatial memory와 섞지 않는다. 실행 권한도 memory
read 권한과 분리한다.

## Overall architecture

```text
                              CONTROL PLANE
 Policy/Consent -- Byte·Energy Budget -- Consumer ACL -- Routine Registry
          |                 |                    |              |
==========|=================|====================|==============|=======
                              DATA PLANE

 Camera / Mic / IMU / VIO / App events
          |
          v
 [1. Sensor Gateway]
 timestamp · source identity · calibration · privacy label
          |
          +-----------------------> [2. Encrypted Hot Evidence Ring]
          |                            bounded source pixels/audio/sensors
          v
 [3. Fast Trigger & Perception]
 scene change · motion · OCR/object candidate · gaze/user pin · routine cue
          |
          v
 [4. Memory Compiler]
          +--------> Visual Evidence Index ---- frame/crop ref + embedding/OCR
          +--------> Episodic Timeline -------- event/span + before/after
          +--------> Semantic Knowledge ------- supported fact/conflict
          +--------> Spatial State ------------ entity/place/pose/validity
          |
          v
 [5. Retention/Promotion Controller]
 admit · promote · merge · pin · expire · delete, under hard byte budget
          |
          v
 [6. Local Indexes]
 temporal interval · vector · entity/graph · spatial · provenance

 Pull request or current-event trigger
          |
          v
 [7. Request Gateway]
 consumer · cutoff · horizon · fidelity · latency/energy · risk · ACL
          |
          v
 [8. Query Planner / Router]
 prefilter -> parallel retrievers -> fuse/rerank -> budgeted refine/STOP
          |
          v
 [9. Evidence Verifier]
 causal · permission · validity · source availability · contradiction · completeness
          |
          v
 [10. ContextPack]
 evidence + provenance + sufficiency + uncertainty + missing evidence + trace
          |
          +--------> QA Adapter ------------ answer / abstain
          +--------> Routine Engine -------- silent / notify / confirm
          +--------> Future adapters ------- search / navigation / summary
```

항상 켜져 있는 부분은 1~3과 bounded ring이다. 4~6의 무거운 embedding,
captioning, consolidation은 기기 상태와 정책에 따라 batch 처리할 수 있다. Read
path는 매 프레임 대형 모델을 호출하지 않는다. Fast trigger가 후보를 만들 때만
retrieval과 reasoning을 깨운다.

## 네 가지 memory view

네 view는 인간 기억을 그대로 복제한 인지 이론이 아니라 서로 다른 질의를 위한
논리적 index다. 물리적으로 네 개 DB를 강제하지 않는다.

| View | 보존 내용 | 주 질의 | 원본과 관계 |
|---|---|---|---|
| Visual | 선택 frame/crop/clip reference, multimodal embedding, OCR/object candidate | 무엇이 보였나, 정확한 텍스트·외형은 무엇인가 | 정확한 재확인은 source pixel이 남아 있을 때만 가능 |
| Episodic | 시간 구간, 사건, 참여 entity, 순서·전이 | 언제, 전후에 무엇이 일어났나 | 여러 visual/sensor evidence를 사건 단위로 묶음 |
| Semantic | 반복 관측에서 통합한 사실·개념·선호와 충돌 이력 | 일반적으로 무엇이 사실인가 | 원 관측 support ID를 유지한 derived knowledge |
| Spatial | entity/place, 위치·거리·방향, validity, uncertainty | 어디, 마지막 위치, 이동 가능성 | episodic state change와 visual evidence에 연결 |

Visual embedding은 원본 frame의 가역 압축이 아니다. 원본이 삭제되면 embedding만
보고 글자를 다시 읽거나 작은 물체를 정확히 확인할 수 없다. 이때 runtime은
`source_available=false`, `fidelity=derived`를 반환해야 한다. OCR text를 별도
record로 남겼다면 그 text는 검색할 수 있지만 원본 재판독 결과로 표현하면 안 된다.

## Write pipeline

### 1. Ingest와 hot evidence

Sensor Gateway가 모든 입력에 event time, ingest time, source identity, device clock,
privacy scope를 붙인다. Raw source는 encrypted ring에만 쓰며 용량을 넘으면 오래된
항목부터 만료한다. 제품 profile에 따라 full frame, sampled frame, 짧은 clip을
보존한다. `약 1시간`은 초기 실험값이며 보장값이 아니다.

### 2. 저비용 trigger

항상 대형 VLM을 실행하지 않는다. 다음 신호로 compile 후보만 만든다.

- scene/shot change, 움직임과 위치 전이
- OCR·object detector confidence, user gaze 또는 명시적 pin
- 새 entity/place, 예상 밖 사건, 기존 record와 충돌
- 활성 routine의 context cue
- 시간·공간 coverage gap과 uncertainty 감소 가능성

Gaze와 surprise는 admission feature일 뿐 진실 label이 아니다. 무시된 영역도 미래
질문에 중요할 수 있어 최소 coverage reservoir를 별도로 둔다.

### 3. Compile과 provenance

Memory Compiler는 provider 출력에서 typed record를 만든다. 모든 derived record는
어떤 source와 선행 record에서 생성됐는지 `support_ids`로 연결한다. Caption이나
semantic fact가 source를 대체하지 않는다.

최소 공통 record envelope는 다음과 같다.

```text
MemoryRecord {
  memory_id, memory_type,
  event_time, valid_from, valid_to,
  entities, payload_ref, embedding_ref,
  support_ids, content_hash,
  confidence, uncertainty,
  source_available, fidelity,
  retention_class, expires_at,
  privacy_scope
}
```

`payload_ref`와 `embedding_ref`는 별도다. Pixel·audio payload를 지워도 index row가
남을 수 있으므로 availability를 매 read마다 검증한다.

### 4. Admission과 promotion

Hard byte budget 아래에서 rule-based baseline으로 시작한다.

```text
priority = user_pin + active_routine_relevance + event_surprise
         + novelty + temporal/spatial_coverage + uncertainty_reduction
         + prior_retrieval_value
         - redundancy - byte_cost - energy_cost - privacy_cost
```

학습형 controller는 이 정책이 실제 병목임이 확인된 뒤에만 검토한다. Routine과
현재 질문의 relevance만 높이면 unknown future question coverage가 붕괴하므로,
bounded random/coverage reservoir를 유지한다.

## Retention tier와 migration

`1시간/1일/1주`를 세 DB로 구현하지 않는다. 같은 record contract 위의 초기
retention policy profile로 둔다.

| Tier | 초기 horizon | 보존 대상 | 가능한 응답 |
|---|---:|---|---|
| H0 Hot source | 0~1시간 | 정책이 허용한 raw/sample frame, audio, sensors | 남은 원본 범위에서 재-OCR·재인식 가능 |
| H1 Evidence | 1시간~1일 | 선택 clip/crop, transcript/OCR span, atomic event·spatial state | 선택된 evidence 범위의 explicit QA와 routine 검증 |
| H2 Consolidated | 1일~1주 이상 | episodic summary, supported semantic fact, stable spatial state, 소수 핵심 evidence | 장기 recall; 삭제된 세부사항은 unknown |
| Pinned | 사용자 정책까지 | 명시적으로 고정한 source 또는 record | 권한·용량 정책 내 장기 재확인 |

Migration은 `derive -> support/hash 검증 -> 새 record atomic commit -> index 반영 ->
정책이 허용하면 source expire` 순서다. Source를 먼저 지우지 않는다. Consolidation이
실패하거나 crash가 나면 기존 source가 남도록 transaction boundary를 둔다.

Semantic consolidation은 새 사실로 과거 사실을 덮어쓰지 않는다. validity를 닫고
새 version을 추가하며 충돌을 함께 반환한다. 사용자 삭제는 provenance graph의
descendant까지 전파하고 vector/graph index에서도 제거해야 한다.

## 공통 Read pipeline

### Request contract

```text
MemoryRequest {
  request_id, consumer_id, mode,            // pull | event
  now, causal_cutoff, query_or_trigger,
  time_window, allowed_memory_types,
  min_fidelity, max_items, max_bytes,
  latency_budget_ms, energy_budget,
  risk_class, privacy_scope
}
```

Consumer는 “모든 기억을 달라”고 할 수 없다. 시간, byte, latency, permission
budget을 선언한다. `causal_cutoff` 이후 record는 planner 전에 차단한다.

### Query planning과 routing

Access control과 causal filter는 deterministic prefilter다. LLM이나 learned router가
이를 우회할 수 없다. MVP router도 규칙 기반으로 시작한다.

| 요청 단서 | 우선 route | 보조 route |
|---|---|---|
| 정확한 글자·외형·장면 | Visual timestamp/vector + available source | Episodic time narrowing |
| 언제·순서·과정 | Episodic interval/event graph | Visual evidence |
| 반복 사실·선호·개념 | Semantic entity/relation | Episodic support와 conflict |
| 어디·거리·마지막 위치 | Spatial operator/state | Episodic transition + visual evidence |
| Visual Routine 조건 | 현재 관측 + Routine Registry | 필요한 episodic/spatial/visual/semantic route만 fan-out |

Router는 memory type 하나를 정답처럼 고르지 않는다. 먼저 cheap temporal/entity
filter로 후보를 줄이고, 필요한 index를 병렬 검색한다. Fusion 단계가 중복을 제거하고
provenance·freshness·query relevance·fidelity로 rerank한다.

첫 retrieval이 불충분하면 남은 latency/energy budget 안에서만 query를 구체화한다.
예: semantic entity를 찾은 뒤 episodic interval을 좁히고, 그 구간의 visual source를
가져온다. 근거가 충분하거나 budget이 끝나면 `STOP`한다.

### Verification과 반환

Evidence Verifier는 다음을 검사한다.

- request cutoff와 evidence time의 causality
- consumer별 privacy scope와 source 권한
- record validity, freshness, coordinate frame, uncertainty
- payload가 실제 남아 있는지와 requested fidelity 충족 여부
- 상충 evidence와 complete-index 요구조건
- 답 또는 행동 조건을 지지하는 evidence coverage

Runtime 결과는 답변이 아니라 다음 `ContextPack`이다.

```text
ContextPack {
  request_id,
  evidence[], proofs[], contradictions[],
  sufficiency, completeness, uncertainty,
  missing_evidence[], source_availability,
  retrieval_trace, consumed_bytes, latency_ms
}
```

`sufficiency`는 consumer별로 해석한다. QA에는 answerability, Routine에는
actionability다. Evidence가 없으면 빈 결과가 정상이며 모델이 추측하도록 만들지
않는다.

## Consumer pipeline

| Policy | QA | Visual Routine |
|---|---|---|
| Trigger | 사용자 질문 | 현재 관측의 event/cue |
| Runtime 목표 | answerability | actionability |
| 주 context | 과거 evidence | 현재 관측 + 과거 evidence + routine state |
| 실패 비용 | unsupported answer | 불필요한 interruption 또는 missed assist |
| 불충분 evidence | abstain | silent 또는 ask confirmation |
| 추가 control | citation·proof consistency | stricter threshold, cooldown, rate limit, action permission |

### Pull: QA

```text
질문
 -> entity/time/operation parse
 -> MemoryRequest(pull, causal cutoff, fidelity 요구)
 -> route + retrieve + verify
 -> ContextPack
 -> QA model 또는 deterministic spatial executor
 -> cited answer / 근거 부족
```

질문이 “어제 화면에 표시된 정확한 번호”를 요구하지만 source와 OCR span이 모두
삭제됐다면 semantic summary가 관련돼도 답하지 않는다. 반대로 “열쇠를 마지막으로
어디서 봤나”는 spatial latest-valid state와 supporting episodic/visual evidence로
답할 수 있다.

### Event-driven: Visual Routine

```text
현재 관측
 -> cheap context trigger
 -> Routine Registry에서 후보 routine match
 -> 과거 상태가 필요한 조건만 MemoryRequest(event) 생성
 -> route + retrieve + verify
 -> Routine condition: satisfied | violated | unknown
 -> Action Gate: silent | notify | ask confirmation
 -> 사용자 feedback을 우선순위·routine state에 반영
```

예: 사용자 routine이 “현관을 나갈 때 열쇠를 두고 가면 알려줘”다.

1. 현재 위치 전이와 문 통과 신호가 routine을 깨운다.
2. 최근 episodic event에서 열쇠를 집은 사건을, spatial state에서 마지막 위치를,
   필요할 때만 visual evidence를 검색한다.
3. “현재 frame에서 열쇠가 안 보임”만으로 부재를 판정하지 않는다.
4. 열쇠가 책상에 남아 있고 이후 pickup evidence가 없으며 관측 coverage가 충분할
   때만 알린다.
5. 근거가 불완전하면 `unknown`으로 조용히 있거나 확인을 요청한다.

MVP action은 `silent`, `notify`, `ask confirmation`으로 제한한다. 메시지 전송,
구매, 문 제어, 이동 같은 외부 행동은 별도 승인·위험도 정책·사용자 확인 없이는
실행하지 않는다. 반복 알림에는 cooldown과 rate limit을 적용한다.

Visual Routine이 추가되면 평가 목표도 달라진다. “답을 잘했는가”뿐 아니라
“도움이 필요할 때만 개입했는가”를 측정해야 한다.

## Device execution profile

```text
Always-on fast path
  capture metadata, bounded ring, cheap trigger, ACL, temporal/entity filter

Opportunistic local path (idle/charging/thermal headroom)
  OCR/VLM caption, embeddings, consolidation, compaction, index maintenance

Optional paired/offload path (explicit policy)
  heavy perception/reasoning only; same request/record contract 사용
```

Device-only baseline을 먼저 정의하되, 안경 단독에서 모든 heavy model이 실행된다고
주장하지 않는다. Phone/cloud offload는 선택 profile이며 raw source의 기본 위치와
전송 허용 범위를 정책으로 고정한다. Offline 상태에서도 hot buffer, local index,
rule trigger, 최소 retrieval은 동작해야 한다.

## Failure와 trust control

| 위험 | Runtime control |
|---|---|
| 삭제된 원본을 본 것처럼 답함 | `source_available`, `fidelity`, support chain 검사; 부족 시 unknown |
| 미래 정보를 과거 질문에 사용 | ingest 전 causal cutoff prefilter와 trace |
| 요약이 원 관측을 왜곡 | derived/source 구분, support hash, conflict versioning |
| “안 보임”을 “없음”으로 판단 | completeness/coverage gate |
| 화면 속 명령문이 assistant를 조종 | visual/OCR payload를 data로 취급; system/action policy와 분리 |
| Consumer 간 개인정보 누출 | per-consumer ACL, privacy label, 최소 ContextPack |
| 원본 삭제 후 파생 index 잔존 | provenance descendant deletion과 index rebuild audit |
| Migration 중 crash/data loss | verify-before-expire, atomic manifest/transaction |
| Routine 과잉 개입 | actionability threshold, cooldown, rate limit, dismiss feedback |
| 고위험 외부 행동 | read와 execute 권한 분리, confirmation, audit log |

## 현재 repository와 목표 상태

이 문서는 목표 아키텍처다. 현재 구현을 완료 상태로 표현하지 않는다.

| Capability | 현재 상태 | 목표 gap |
|---|---|---|
| Episodic/semantic/visual/spatial record | schema와 fixture/일부 LLM build 경로 존재 | streaming compiler와 device provider 필요 |
| Causal retrieval | 구현 | 실제 vector/graph reranking과 iterative refine 필요 |
| Visual retrieval | frame/embedding reference 존재 | real embedding index, source availability 관리 필요 |
| Spatial proof | typed record와 deterministic proof 구현 | production sensor/provider와 target-device 검증 필요 |
| Byte control | canonical serialized byte cap 구현 | 전 memory tier 통합 budget과 flash write cost 필요 |
| Evidence lineage | 강한 artifact/QA lineage 구현 | live retention·delete provenance로 확장 필요 |
| Hot evidence ring | 미구현 | encrypted bounded ring과 crash recovery 필요 |
| Tier migration | 미구현 | promote/verify/expire state machine 필요 |
| Consumer ACL | 미구현 | request gateway와 privacy scope 필요 |
| Visual Routine | 미구현 | registry, trigger, condition state, action gate 필요 |
| Hardware profile | 미측정 | 저장·지연·전력·thermal 실측 필요 |

현재 retrieval의 일부는 lexical/keyword heuristic이고 visual `embedding_ref`는 실제
vector retrieval 근거가 아니다. Published WorldMM의 PPR, learned embedding retrieval,
LLM rerank, iterative STOP을 재현했다고 주장하지 않는다.

## 검증 지표와 gate

QA 하나의 accuracy로 런타임을 승인하지 않는다.

| 영역 | 최소 지표 |
|---|---|
| Memory write | byte/hour, flash write amplification, promotion/expiry rate, source retention, write energy |
| Retrieval | Recall@K, temporal IoU, first-evidence latency, bytes/query, causal/ACL violation |
| QA | age×task accuracy, evidence support rate, selective risk-coverage, unsupported-answer rate |
| Visual Routine | trigger precision/recall, false-interruption rate, missed-assist rate, time-to-assist, dismiss rate |
| Device | p50/p95 latency, average/peak power, thermal throttling, RAM/flash peak |
| Privacy | off-device bytes, unauthorized-read count, deletion propagation completion |

### Gate 1: fixed-byte offline value

같은 source와 총 byte budget에서 다음을 비교한다.

1. recent-only ring
2. uniform frame sampling
3. summary-only memory
4. proposed tiered typed memory

QA replay와 Visual Routine replay를 모두 사용한다. Proposed 방식이 한 consumer만
개선하고 다른 consumer의 evidence coverage를 악화하면 공통 runtime 가설은
통과하지 못한다.

### Gate 2: target-device feasibility

대표 기기와 시나리오 하나를 정한 뒤 저장량, 배터리, thermal, p95 retrieval
latency를 실측한다. 여기서 `1시간/1일/1주`를 byte·energy 결과에 맞게 조정한다.

### Gate 3: limited user value

알림은 opt-in routine에만 적용한다. QA unsupported-answer, routine false interruption,
삭제/권한 실패가 합의된 상한을 넘으면 확대하지 않는다.

## 설계 근거와 해석

| 1차 근거 | 관찰된 결과 | 이 설계에 사용한 부분 | 사용하지 않은 주장 |
|---|---|---|---|
| [WorldMM](https://arxiv.org/abs/2512.02425) | 저자 보고 기준 5개 long-video QA benchmark에서 이전 SOTA 대비 평균 8.4% 향상; episodic·semantic·visual memory와 adaptive multi-step retrieval 사용 | memory별 index, iterative retrieval, evidence sufficiency | on-device 구현·저장/전력 효율 근거로 사용하지 않음 |
| [SuperMemory-VQA](https://arxiv.org/abs/2606.00825) | 52.9시간의 RGB·음성·gaze·IMU·SLAM과 grounded QA 4,853개, explicit unanswerable option을 구성 | causal multimodal evidence와 abstention 평가 | runtime architecture가 검증됐다는 근거로 사용하지 않음 |
| [MemoLens, MobiSys 2026](https://sigmobile.org/mobisys/2026/accepted_papers/) | gaze-aware selective memory, hierarchical retention, coarse-to-fine retrieval을 glasses·phone·cloud prototype에서 평가 | selective write, hierarchy, physical latency/energy/storage 측정 | 순수 glasses 단독 실행 근거로 사용하지 않음 |
| [Worth Remembering](https://arxiv.org/abs/2606.03787) | surprise-gated episodic retention을 spatial memory와 결합해 저자 보고 기준 temporal·spatial·binary robot QA에서 기존 방법 대비 12% 이상 향상 | surprise를 promotion feature로 사용 | surprise만으로 미래 중요도를 안다고 가정하지 않음 |
| [Memory-Centric EQA](https://arxiv.org/abs/2505.13948) | module별 최소 sufficient memory를 planner·stopping·answer에 사용해 저자 보고 기준 MT-HM3D에서 baseline 대비 9.9% 향상 | shared runtime을 여러 control consumer가 사용 | 특정 map 구조를 그대로 채택하지 않음 |
| [ReMEmbR](https://arxiv.org/abs/2409.13682) | text·time·space retrieval로 질문 답과 navigation goal을 생성 | 동일 memory read API의 QA/행동 재사용 | 측정 latency를 target device 성능으로 전용하지 않음 |
| [Vinci2](https://arxiv.org/abs/2607.11523) | streaming egocentric memory로 개입 시점과 응답을 판단 | Visual Routine의 event-driven retrieval | 2026-07 신규 preprint 결과를 제품 근거로 확정하지 않음 |
| [Plan, Watch, Recover](https://arxiv.org/abs/2606.04970) | long-horizon plan과 per-frame watch/recovery를 분리 | cheap fast trigger와 conditional slow reasoning 분리 | 외부 행동의 안전성이 해결됐다고 보지 않음 |
| [ConceptGraphs](https://arxiv.org/abs/2309.16650), [Hydra](https://arxiv.org/abs/2201.13360) | compact scene graph와 incremental layered spatial state | spatial view와 fast tracking/persistent state 분리 | dense map 장기 보존을 기본값으로 두지 않음 |

WorldMM이 memory 분리와 adaptive retrieval의 알고리즘 근거라면, 이 프로젝트의
차별점은 이를 **bounded lifecycle, provenance, permissions, multiple consumers,
device budgets**를 가진 운영 계약으로 바꾸는 데 있다. 이는 논문 결과가 아니라
검증해야 할 프로젝트 가설이다.

## 남은 제품 결정

Architecture를 구현하기 전에 다음 두 결정만 필요하다.

1. 첫 target device와 관측 profile: camera sampling, sensor, 저장·전력 한도
2. 첫 routine class: 사용자 정의 단순 알림인지, 작업 절차 보조인지

외부 행동 자동화, 범용 routine language, 학습형 retention controller는 초기
범위에 넣지 않는다.

[임원 제안으로 돌아가기](on-device-memory-runtime-proposal.md) ·
[Spatial Memory 상세 명세](architecture.md) ·
[프로젝트 홈](README.md)
