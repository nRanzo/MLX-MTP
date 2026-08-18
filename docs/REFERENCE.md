# Reference map

MTPLX is inspected only as a behavioral oracle. `mlx-mtp` contains no MTPLX
imports or copied source.

| Reference area | mlx-mtp equivalent |
| --- | --- |
| Qwen3.5 MTP injection | `adapters/qwythos.py` |
| MTP head/cache | `adapters/qwythos.py`, `cache.py` |
| Sampling/rejection | `sampling.py`, `speculative.py` |
| Runtime orchestration | deliberately not reproduced |

For Qwythos the observed contract is: expose the target pre-final-RMSNorm
residual stream; RMS-normalize embedding and residual independently; concatenate
`[embedding, hidden]`; project 8192 to 4096; run a full-attention Qwen3.5 layer;
RMS-normalize; apply the untied target LM head.

## Cache ownership

The target cache belongs only to accepted target history. MTP has its own
full-attention `KVCache`, one per physical MTP layer, and never reuses the
target's hybrid cache. During verification MLX-MTP clones the target cache.
Acceptance commits that clone; rejection starts from the prior committed cache
and replays the anchor plus residual replacement. This makes cache semantics
explicit at the cost of an intentionally unoptimized correction path.

## Deliberate exclusions

MTPLX includes a much broader runtime, compilation, server, session, vision,
and hardware-management surface. Those areas are not part of this project and
are not represented by this map.
