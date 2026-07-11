# Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

| Field | Value |
|---|---|
| Page ID | SM-PAPER-VSI-BENCH |
| Status | Reviewed; code and dataset available |
| Publication | CVPR 2025 Oral, pp. 10632–10643; arXiv:2412.14171 v2 |
| Primary source | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.html) |
| Official code | [vision-x-nyu/thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space) |
| Last checked | 2026-07-11 |
| Project links | [Parent: paper index](README.md) · [Architecture](../architecture.md) · [Traceability](../traceability.md) |
| Project claims | [Traceability](../traceability.md): C-003, C-008 |

## 30-second summary

VSI-Bench evaluates whether multimodal models can reconstruct and recall indoor space from sequential video. Prompt-only reasoning does not close the gap, while an explicit cognitive map improves relative-distance accuracy. This is direct evidence for a global spatial representation and geometry execution, but not for storage compression.

## Problem addressed

Strong video-language models may recognize content without building a stable allocentric model. VSI-Bench tests configuration, metric estimation, and spatiotemporal recall from egocentric indoor scan videos using eight spatial tasks.

## Relevant method

- Build more than 5,000 QA pairs over 288 videos from ScanNet, ScanNet++, and ARKitScenes.
- Test object count, relative distance, relative direction, route planning, absolute distance, object size, room size, and appearance order.
- Use accuracy for multiple-choice tasks and Mean Relative Accuracy for numerical tasks.
- Probe model reasoning with explanations and model-generated cognitive maps.

## Paper-reported evidence

| Dataset | Condition | Metric | Reported result | Location |
|---|---|---|---|---|
| VSI-Bench | Gemini-1.5 Pro | Average | 45.4 | Table 1 |
| VSI-Bench | Best reported open model, LLaVA-Video-72B | Average | 40.9 | Table 1 |
| VSI-Bench tiny | Human → Gemini-1.5 Pro | Average | 79.2 → 48.8 | Table 1 |
| Error analysis | 163 incorrect samples | Errors attributed to spatial reasoning | About 71% | Figure 7 and Section 5.1 |
| Relative-distance subset | No map → model-generated cognitive map | Accuracy | 46.0 → 56.0 | Table 3 |
| Relative-distance subset | Ground-truth 10×10 → 20×20 map | Accuracy | 66.0 → 78.0 | Table 3 |

The paper reports that zero-shot chain-of-thought and tree-of-thought reduced average performance by about four points, while model-generated cognitive maps improved the relative-distance task by ten points.

## What this supports here

- A unified allocentric representation is preferable to disconnected local impressions.
- Relative direction, distance, route, and metric questions should be evaluated separately.
- Explicit spatial structure can help where language-only reasoning prompts do not.
- The project infers that deterministic geometry operations should consume explicit records instead of asking a language model to estimate geometry.

## What it does not prove

- A particular typed-record schema or learned write policy.
- Storage reduction, actual-byte limits, lifelong memory, or 1 Hz sampling.
- Metric calibration from monocular RGB or pose uncertainty handling.
- SuperMemory-VQA transfer.

## Project reproduction status

Not reproduced. VSI-Bench is retained as an external spatial error taxonomy and evaluation reference.

## References

- Jihan Yang et al. [Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.html). CVPR 2025.
- [Official project page](https://vision-x-nyu.github.io/thinking-in-space.github.io/).
- [Official repository](https://github.com/vision-x-nyu/thinking-in-space).
- [Official dataset](https://huggingface.co/datasets/nyu-visionx/VSI-Bench).
- [Back to paper index](README.md).
