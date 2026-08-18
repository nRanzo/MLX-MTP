from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer
import mtplx.runtime as runtime


model_path = Path(
    "/Users/nicola/.mtplx/models/"
    "mlx-community--Qwythos-9B-v2-OptiQ-4bit"
)

# ============================================================
# LOAD
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

print("Loading MTPLX...")
rt = runtime.load(model_path, mtp=True)

print("MTP enabled:", rt.draft_mtp is not None)


# ============================================================
# PROMPT
# ============================================================

prompt = "Explain why the sky appears blue."

prompt_ids = tokenizer.encode(prompt)

input_ids = mx.array(
    [prompt_ids],
    dtype=mx.int64,
)

print("\nPROMPT:")
print(prompt)

print("prompt tokens:", len(prompt_ids))


# ============================================================
# STEP 1 — TARGET PREFILL
# ============================================================

print("\n=== TARGET PREFILL ===")

cache = rt.make_cache()

logits, hidden = rt.forward_ar(
    input_ids,
    cache=cache,
    return_hidden=True,
    emit_logits=True,
)

mx.eval(logits, hidden)

primary = mx.argmax(
    logits[:, -1, :],
    axis=-1,
).reshape(1, 1)

mx.eval(primary)

primary_id = int(primary.item())

print(
    "Target N+1:",
    primary_id,
    repr(tokenizer.decode([primary_id])),
)


# ============================================================
# STEP 2 — MTP DRAFT
# ============================================================

print("\n=== MTP DRAFT ===")

mtp_cache = rt.make_mtp_cache()

draft_logits, draft_hidden = rt.draft_mtp(
    hidden[:, -1:, :],
    primary,
    mtp_cache=mtp_cache,
    return_hidden=True,
    mtp_depth=1,
)

mx.eval(draft_logits, draft_hidden)

candidate = mx.argmax(
    draft_logits[:, -1, :],
    axis=-1,
).reshape(1, 1)

mx.eval(candidate)

candidate_id = int(candidate.item())

print(
    "MTP candidate N+2:",
    candidate_id,
    repr(tokenizer.decode([candidate_id])),
)


# ============================================================
# STEP 3 — TARGET VERIFICATION
#
# We explicitly feed:
#
# prompt + primary
#
# and ask target for the next token.
# ============================================================

print("\n=== TARGET VERIFICATION ===")

verify_input = mx.concatenate(
    [
        input_ids,
        primary.astype(mx.int64),
    ],
    axis=1,
)

verify_cache = rt.make_cache()

verify_logits, verify_hidden = rt.forward_ar(
    verify_input,
    cache=verify_cache,
    return_hidden=True,
    emit_logits=True,
)

mx.eval(verify_logits, verify_hidden)

target_n2 = mx.argmax(
    verify_logits[:, -1, :],
    axis=-1,
).reshape(1, 1)

mx.eval(target_n2)

target_n2_id = int(target_n2.item())

print(
    "Target true N+2:",
    target_n2_id,
    repr(tokenizer.decode([target_n2_id])),
)


# ============================================================
# STEP 4 — COMPARE
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(
    "N+1 target :",
    primary_id,
    repr(tokenizer.decode([primary_id])),
)

print(
    "N+2 MTP    :",
    candidate_id,
    repr(tokenizer.decode([candidate_id])),
)

print(
    "N+2 target :",
    target_n2_id,
    repr(tokenizer.decode([target_n2_id])),
)

print()
print("MTP MATCH:", candidate_id == target_n2_id)

if candidate_id == target_n2_id:
    print("✓ MTP prediction is correct for this position.")
else:
    print("✗ MTP prediction differs from target.")
