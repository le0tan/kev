"""Per-qid comparison of two or more imajev answers files (kev tiers, or kev vs native).

Each answers file is one JSON object per request row with an "answers" dict {qid: answer}. This tool
expands every file to per-question records and compares them pairwise under the official v1.1 argmax
(evaluate_text_decisions.canonical_distribution):

  noul  -> true = min(1-u, max(0, noul-0.5u)), false = 1-u-true, unknown = u   (noul is true+0.5u)
  choice-> known probabilities as stored (same to_response on every side being compared) + unknown = u

label = argmax over known classes + __unknown__, exactly the official "max(served, key=served.get)".

Usage: python tools/imajev_answers_compare.py a.jsonl b.jsonl [--label-a C1] [--label-b C16] [--out diff.json]
"""
import argparse, json, math, statistics


def load(path):
    """-> {key: (type, label, top_known, top_known_name, u, raw)} with raw kept for value diffs.
    Accepts both on-disk formats: kev per-request rows ({"ridx","chunk","answers":{qid:ans}}) and native
    per-question rows ({"ridx","chunk","qid","answer"}). The comparison unit is one question of one
    request row, keyed (ridx, chunk, qid) — qid strings repeat across requests."""
    out = {}

    def put(ridx, chunk, qid, a):
        key = f"{ridx}:{chunk}:{qid}"
        u = a["unknown_probability"]
        if a["type"] == "noul":
            # kev boolean (jev_api.to_response): noul = true_raw + 0.5*u; official canonical reconstructs
            # true = min(1-u, max(0, noul-0.5u)), false = 1-u-true
            true_p = min(1.0 - u, max(0.0, a["noul"] - 0.5 * u))
            false_p = 1.0 - u - true_p
            served = {"true": true_p, "false": false_p, "__unknown__": u}
            label = max(served, key=served.get)
            top = max(("true", "false"), key=lambda k: served[k])
            out[key] = ("boolean", label, served[top], top, u,
                        {"served": served, "noul": a["noul"], "u": u})
        elif a["type"] == "choice":
            probs = a["probabilities"]
            # stored probabilities are known-renormalized (sum to 1); the official canonical_distribution
            # serves p*(1-u) per option plus __unknown__=u, so argmax must include that scaling.
            # native's to_answer writes booleans in this shape too (probabilities over true/false) —
            # reduce those to the boolean form; the math reduces to the same served distribution
            if set(probs) == {"true", "false"}:
                served = {"true": probs["true"] * (1 - u), "false": probs["false"] * (1 - u), "__unknown__": u}
                label = max(served, key=served.get)
                top = max(("true", "false"), key=lambda k: served[k])
                out[key] = ("boolean", label, served[top], top, u, {"served": served, "u": u})
                return
            served = {k: v * (1.0 - u) for k, v in probs.items()}; served["__unknown__"] = u
            label = max(served, key=served.get)
            top_name = max(probs, key=probs.get)
            out[key] = ("choice", label, probs[top_name] * (1.0 - u), top_name, u,
                        {"choice": a.get("choice"), "probs": probs, "u": u})
        else:
            raise SystemExit(f"unknown answer type {a['type']!r} for {key}")

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "answers" in row:   # kev per-request row
                for qid, a in row["answers"].items():
                    put(row["ridx"], row["chunk"], qid, a)
            else:                  # native per-question row
                put(row["ridx"], row["chunk"], row["qid"], row["answer"])
    return out


def pdiff(vals):
    """p50/p90/max of a list of absolute differences."""
    if not vals: return {"n": 0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    s = sorted(vals)
    return {"n": len(vals), "p50": round(s[len(s)//2], 6), "p90": round(s[int(len(s)*0.9)], 6),
            "max": round(s[-1], 6)}


def compare(A, B, la, lb):
    qids = sorted(set(A) & set(B))
    if len(A) != len(B) or len(qids) != len(A):
        only_a = set(A) - set(B); only_b = set(B) - set(A)
        raise SystemExit(f"qid sets differ: only-{la}={len(only_a)} only-{lb}={len(only_b)} (first: {sorted(only_a)[:3]} / {sorted(only_b)[:3]})")
    known_flip = abstain_flip = both_unknown = exact_float = 0
    du, dknown, dtrue = [], [], []
    flip_rows = []
    for qid in qids:
        ta, la_, ka, na, ua, rawa = A[qid]
        tb, lb_, kb, nb, ub, rawb = B[qid]
        assert ta == tb, f"type mismatch on {qid}"
        if la_ == "__unknown__" and lb_ == "__unknown__": both_unknown += 1
        elif la_ == "__unknown__" or lb_ == "__unknown__": abstain_flip += 1
        elif la_ != lb_: known_flip += 1
        if rawa == rawb: exact_float += 1
        du.append(abs(ua - ub))
        dknown.append(abs(ka - kb))
        if ta == "boolean" and "served" in rawa and "served" in rawb:
            dtrue.append(abs(rawa["served"]["true"] - rawb["served"]["true"]))
        if la_ != lb_:
            flip_rows.append({"qid": qid, "type": ta, f"label_{la}": la_, f"label_{lb}": lb_,
                              f"u_{la}": ua, f"u_{lb}": ub,
                              **{f"served_{la}": rawa.get("served"), f"served_{lb}": rawb.get("served")}})
    return {"questions": len(qids), "known_class_flips": known_flip, "abstain_flips": abstain_flip,
            "both_unknown": both_unknown, "identical_answer_dicts": exact_float,
            "unknown_prob_diff": pdiff(du), "top_known_prob_diff": pdiff(dknown),
            "served_true_diff": pdiff(dtrue) if dtrue else None, "flips": flip_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="two or more answers .jsonl files")
    ap.add_argument("--labels", help="comma-separated names matching the files, e.g. C1,C16")
    ap.add_argument("--out", help="write the full JSON (including flip rows) here")
    a = ap.parse_args()
    labels = a.labels.split(",") if a.labels else [f"f{i}" for i in range(len(a.files))]
    assert len(labels) == len(a.files), "one label per file"
    data = {l: load(f) for l, f in zip(labels, a.files)}
    names = list(data)
    report = {"files": dict(zip(labels, a.files)), "pairs": {}}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = names[i], names[j]
            r = compare(data[x], data[y], x, y)
            report["pairs"][f"{x}_vs_{y}"] = r
            print(f"[{x} vs {y}] n={r['questions']} known_flips={r['known_class_flips']} "
                  f"abstain_flips={r['abstain_flips']} both_unknown={r['both_unknown']} "
                  f"identical_dicts={r['identical_answer_dicts']} "
                  f"|du| max={r['unknown_prob_diff']['max']} p50={r['unknown_prob_diff']['p50']} "
                  f"|dtop| max={r['top_known_prob_diff']['max']}")
    if a.out:
        with open(a.out, "w") as f: json.dump(report, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
