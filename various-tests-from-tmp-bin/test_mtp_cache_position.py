from pathlib import Path
import mlx.core as mx
import mtplx.runtime as runtime


MODEL = Path(".").resolve()
PROMPT = "Explain why the sky appears blue."

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def cache_offsets(cache):
    if cache is None:
        return []

    result = []

    for i, entry in enumerate(cache):
        result.append(
            (
                i,
                type(entry).__name__,
                getattr(entry, "offset", None),
                getattr(entry, "rollback_state", None),
            )
        )

    return result


def print_offsets(label, cache):
    print(f"\n[{label}]")

    if cache is None:
        print("  cache=None")
        return

    for i, typ, offset, rollback in cache_offsets(cache):
        print(
            f"  layer={i:2d} "
            f"type={typ:30s} "
            f"offset={offset} "
            f"rollback={rollback}"
        )


def top10(logits, tokenizer):
    row = logits[:, -1, :]

    indices = mx.argsort(row, axis=-1)[..., -10:][..., ::-1]

    values = mx.take_along_axis(row, indices, axis=-1)
    
    values = values[0]
    indices = indices[0]

    result = []

    for i in range(10):
        token_id = int(indices[i].item())
        value = float(values[i].item())

        result.append(
            (
                token_id,
                value,
                tokenizer.decode([token_id]),
            )
        )

    return result


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

print("Loading...")

rt = runtime.load(
    MODEL,
    mtp=True,
)

tokenizer = rt.tokenizer

# ------------------------------------------------------------
# Tokenize
# ------------------------------------------------------------

prompt_tokens = tokenizer.encode(PROMPT)

input_ids = mx.array([prompt_tokens])

print("\nPROMPT:")
print(PROMPT)

print("input_ids:", input_ids.shape)
print("prompt length:", len(prompt_tokens))

# ------------------------------------------------------------
# TARGET PREFILL
# ------------------------------------------------------------

print("\n=== TARGET PREFILL ===")

logits, hidden = rt.forward_ar(
    input_ids,
    return_hidden=True,
)

mx.eval(logits, hidden)

target_n1 = int(
    mx.argmax(
        logits[:, -1, :],
        axis=-1,
    ).item()
)

print(
    "Target N+1:",
    target_n1,
    repr(tokenizer.decode([target_n1])),
)

print("hidden:", hidden.shape)

# ------------------------------------------------------------
# TARGET N+2
#
# We explicitly perform another target forward using N+1.
# This is the ground truth we compare against.
# ------------------------------------------------------------

print("\n=== TARGET N+2 ===")

target_n2_logits = rt.forward_ar(
    mx.array([[target_n1]]),
)

mx.eval(target_n2_logits)

target_n2 = int(
    mx.argmax(
        target_n2_logits[:, -1, :],
        axis=-1,
    ).item()
)

print(
    "Target N+2:",
    target_n2,
    repr(tokenizer.decode([target_n2])),
)

print("\nTarget TOP 10:")

for rank, (token_id, value, text) in enumerate(
    top10(target_n2_logits, tokenizer),
    1,
):
    print(
        f"{rank:2d}. "
        f"{token_id:6d} "
        f"{value:12.6f} "
        f"{repr(text)}"
    )

# ------------------------------------------------------------
# MTP TEST MATRIX
# ------------------------------------------------------------

positions = [
    None,
    0,
    1,
    len(prompt_tokens) - 1,
    len(prompt_tokens),
    len(prompt_tokens) + 1,
]

cache_modes = [
    "none",
    "fresh",
]

print("\n")
print("=" * 100)
print("MTP CACHE / POSITION TEST")
print("=" * 100)

results = []

for cache_mode in cache_modes:

    for position_offset in positions:

        print("\n" + "-" * 100)

        print(
            f"cache={cache_mode:6s} "
            f"position_offset={str(position_offset):>4s}"
        )

        print("-" * 100)

        if cache_mode == "none":
            mtp_cache = None
        else:
            mtp_cache = rt.make_mtp_cache()

        print_offsets(
            "BEFORE",
            mtp_cache,
        )

        try:

            draft_logits, draft_hidden = rt.draft_mtp(
                hidden,
                mx.array([[target_n1]]),
                mtp_cache=mtp_cache,
                return_hidden=True,
                mtp_depth=1,
                position_offset=position_offset,
            )

            mx.eval(
                draft_logits,
                draft_hidden,
            )

            candidate = int(
                mx.argmax(
                    draft_logits[:, -1, :],
                    axis=-1,
                ).item()
            )

            print(
                "\nMTP N+2:",
                candidate,
                repr(tokenizer.decode([candidate])),
            )

            print(
                "MATCH:",
                candidate == target_n2,
            )

            print("\nMTP TOP 10:")

            for rank, (token_id, value, text) in enumerate(
                top10(draft_logits, tokenizer),
                1,
            ):
                print(
                    f"{rank:2d}. "
                    f"{token_id:6d} "
                    f"{value:12.6f} "
                    f"{repr(text)}"
                )

            print_offsets(
                "AFTER",
                mtp_cache,
            )

            results.append(
                {
                    "cache": cache_mode,
                    "position": position_offset,
                    "token": candidate,
                    "match": candidate == target_n2,
                }
            )

        except Exception as e:

            print(
                "\nERROR:",
                type(e).__name__,
                str(e),
            )

            results.append(
                {
                    "cache": cache_mode,
                    "position": position_offset,
                    "token": None,
                    "match": False,
                }
            )


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    "Target N+1:",
    target_n1,
    repr(tokenizer.decode([target_n1])),
)

print(
    "Target N+2:",
    target_n2,
    repr(tokenizer.decode([target_n2])),
)

print()

for result in results:

    print(
        f"cache={result['cache']:6s} "
        f"position={str(result['position']):>4s} "
        f"token={str(result['token']):>6s} "
        f"match={result['match']}"
    )

print("\nDONE")
