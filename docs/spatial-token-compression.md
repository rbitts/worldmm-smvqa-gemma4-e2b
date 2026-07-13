# Spatial Token Compression

| 항목 | 값 |
|---|---|
| Page ID | SM-LEGACY-TOKEN-COMPRESSION |
| 문서 역할 | 레거시 안내 |
| Canonical 아키텍처 | [Spatial Memory 아키텍처](spatial-memory/architecture.md) |
| Canonical 상태 | [현재 상태](spatial-memory/status.md) |

## 핵심 결론

이 page에서 별도 spatial-memory specification을 유지하지 않는다. Canonical
decision은 dense geometry를 transient하게 사용하고 hard serialized-byte budget
아래에서 선택한 explicit typed record만 persist하는 것이다.

```text
sparse RGB + IMU/VIO/depth
    -> transient geometry
    -> typed candidates
    -> actual-byte writer
    -> causal retrieval
    -> deterministic proof
    -> QA or abstention
```

## 유지할 결론

- Compression은 write time에 redundant/low-value fact를 제외하는 방식이며,
  post-hoc feature quantization이 primary method가 아니다.
- Persistent fact에는 identity, frame, validity, uncertainty, provenance,
  evidence가 필요하다.
- Token count만으로는 부족하며 actual artifact byte와 QA quality를 비교한다.
- 현재 heuristic path는 local sanity만 검증했다. Learned production evidence는
  end-to-end로 실행하지 않았다.

## 의사결정 경로

- [문제와 성공 gate](spatial-memory/problem.md)
- [아키텍처와 control](spatial-memory/architecture.md)
- [채택한 ADR](spatial-memory/decisions/README.md)
- [실험과 결과](spatial-memory/experiments/README.md)
- [실행 로드맵](spatial-memory/roadmap.md)

새 내용은 위 canonical 문서에 기록한다.
