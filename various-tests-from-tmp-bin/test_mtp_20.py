from pathlib import Path
import mlx.core as mx
import mtplx.runtime as runtime

MODEL = Path(".").resolve()

PROMPTS = [
    "Explain why the sky appears blue.",
    "What is 2 + 2?",
    "The capital of France is",
    "The color of grass is",
    "Water freezes at",
    "The opposite of hot is",
    "Python is a",
    "Machine learning is",
    "The sun is a",
    "Earth is the third planet from the",
    "Once upon a time",
    "The quick brown fox",
    "In physics, gravity is",
    "A triangle has",
    "The largest ocean is",
    "Rome is the capital of",
    "1 + 1 =",
    "Hello, my name is",
    "Artificial intelligence is",
    "The meaning of life is",
]

print("Loading...")
rt = runtime.load(MODEL, mtp=True)
tokenizer = rt.tokenizer

print()

def decode(token_id):
    return repr(tokenizer.decode([int(token_id)]))

def top10(logits):
    row = logits[0, -1, :]
    indices = mx.argsort(row)[-10:][::-1]
    return [(int(i), float(row[i])) for i in indices]

results = []

for n, prompt in enumerate(PROMPTS, 1):

    print("=" * 80)
    print(f"{n:02d}. {prompt}")

    input_ids = mx.array(
        [tokenizer.encode(prompt)],
        dtype=mx.int32,
    )

    # ------------------------------------------------------------
    # TARGET PREFILL
    # ------------------------------------------------------------

    target_logits, hidden = rt.forward_ar(
        input_ids,
        return_hidden=True,
    )

    target_n1 = int(
        mx.argmax(target_logits[:, -1, :], axis=-1).item()
    )

    # ------------------------------------------------------------
    # TARGET N+2
    # ------------------------------------------------------------

    target_n1_ids = mx.array([[target_n1]], dtype=mx.int32)

    target_n2_logits, _ = rt.forward_ar(
        target_n1_ids,
        return_hidden=True,
    )

    target_n2 = int(
        mx.argmax(target_n2_logits[:, -1, :], axis=-1).item()
    )

    # ------------------------------------------------------------
    # MTP N+2
    #
    # IMPORTANT:
    # MTP vuole solo l'ultimo hidden state:
    # (1, 1, 4096)
    # ------------------------------------------------------------

    hidden_last = hidden[:, -1:, :]

    mtp_logits, _ = rt.draft_mtp(
        hidden_last,
        target_n1_ids,
        return_hidden=True,
        mtp_depth=1,
    )

    mtp_n2 = int(
        mx.argmax(mtp_logits[:, -1, :], axis=-1).item()
    )

    # ------------------------------------------------------------
    # RANKS
    # ------------------------------------------------------------

    target_row = target_n2_logits[0, -1, :]
    mtp_row = mtp_logits[0, -1, :]

    target_order = mx.argsort(target_row)[::-1]
    mtp_order = mx.argsort(mtp_row)[::-1]

    target_rank_of_mtp = (
        int(mx.argmax((target_order == mtp_n2).astype(mx.int32)).item())
        + 1
    )

    mtp_rank_of_target = (
        int(mx.argmax((mtp_order == target_n2).astype(mx.int32)).item())
        + 1
    )

    match = target_n2 == mtp_n2

    results.append({
        "match": match,
        "target_rank_of_mtp": target_rank_of_mtp,
        "mtp_rank_of_target": mtp_rank_of_target,
    })

    print()
    print(f"Target N+1 : {target_n1:6d} {decode(target_n1)}")
    print(f"Target N+2 : {target_n2:6d} {decode(target_n2)}")
    print(f"MTP N+2    : {mtp_n2:6d} {decode(mtp_n2)}")
    print(f"Match      : {match}")
    print()
    print(f"MTP token rank in TARGET : {target_rank_of_mtp}")
    print(f"TARGET token rank in MTP : {mtp_rank_of_target}")

    print()
    print("TARGET TOP 10:")
    for rank, (tid, logit) in enumerate(top10(target_n2_logits), 1):
        print(
            f"  {rank:2d}. {tid:6d} "
            f"{logit:10.4f} {decode(tid)}"
        )

    print()
    print("MTP TOP 10:")
    for rank, (tid, logit) in enumerate(top10(mtp_logits), 1):
        print(
            f"  {rank:2d}. {tid:6d} "
            f"{logit:10.4f} {decode(tid)}"
        )

# ============================================================
# SUMMARY
# ============================================================

matches = sum(r["match"] for r in results)
total = len(results)

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Matches: {matches}/{total}")
print(f"Accuracy: {100.0 * matches / total:.1f}%")

print()
print("Prompt | Match | MTP rank in Target | Target rank in MTP")

for i, r in enumerate(results, 1):
    print(
        f"{i:02d}     | "
        f"{str(r['match']):5s} | "
        f"{r['target_rank_of_mtp']:17d} | "
        f"{r['mtp_rank_of_target']:17d}"
    )
