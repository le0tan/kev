"""Independent CPU re-audit: native vs kev answers on the exp10k pool (10003 questions).

Deliberately self-contained — it does NOT import tools/imajev_answers_compare.py. Labels come from the
OFFICIAL evaluator's own canonical_distribution (evaluate_text_decisions.py, v1.1) called on each answer,
so tie-breaking and construction order are exactly the official semantics. The comparison unit is the
composite question key (ridx, chunk, qid): raw qid strings repeat across requests (they are question
names), so composite keys — not qid strings — define uniqueness.

Usage: IMAJEV_SCRIPTS=<imajev>/scripts python tools/imajev_cross_audit.py --pool <pool.jsonl> \
    --native <answers.jsonl> --kev <answers.jsonl> --out-dir <dir>
"""
import argparse, collections, hashlib, json, math, os, sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(vals, q):
    if not vals: return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(q * len(s)))], 6)


def dist(vals):
    if not vals: return {"n": 0}
    return {"n": len(vals), "p50": pct(vals, .5), "p90": pct(vals, .9), "max": round(max(vals), 6)}


def official_served(answer, gold_probs):
    """The official served distribution via the official function; gold extras are 0-probability and
    cannot win the argmax, so gold={} is exact for label purposes."""
    return canonical_distribution(answer, {"probabilities": gold_probs})


def label_of(served):
    return max(served, key=served.get)   # official: first max in insertion order


def load_pool(path):
    """-> expected[(ridx,chunk,qid)] = qtype; also counts for the duplicate audit."""
    expected, dup, qid_names = {}, 0, set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            for qid, q in row["questions"].items():
                key = (row["ridx"], row["chunk"], qid)
                if key in expected: dup += 1
                expected[key] = q["type"]
                qid_names.add(qid)
    return expected, dup, qid_names


def load_answers(path, fmt):
    """fmt 'kev': per-request rows; 'native': per-question rows. -> dict[key] = answer dict."""
    out, written = {}, 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if fmt == "kev":
                for qid, a in row["answers"].items():
                    out[(row["ridx"], row["chunk"], qid)] = a
                    written += 1
            else:
                out[(row["ridx"], row["chunk"], row["qid"])] = row["answer"]
                written += 1
    return out, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--kev", required=True)
    ap.add_argument("--kev-summary", help="kev bench summary JSON (config provenance)")
    ap.add_argument("--native-summary", help="native bench summary JSON")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-sha", default=None, help="expected pool sha256 prefix to assert")
    a = ap.parse_args()

    report = {"pool_sha256": sha256_file(a.pool), "expect_sha_prefix": a.expect_sha}
    if a.expect_sha and not report["pool_sha256"].startswith(a.expect_sha):
        raise SystemExit(f"POOL SHA MISMATCH: {report['pool_sha256']}")
    for name, path in (("kev", a.kev_summary), ("native", a.native_summary)):
        if path:
            d = json.load(open(path))
            report[f"{name}_config"] = {k: d.get(k) for k in
                                        ("side", "concurrency", "warmup_pass_s", "questions", "answers_written",
                                         "answered", "abstained", "coverage_ok",
                                         "prefix_cache_cumulative", "end_to_end_s",
                                         "questions_per_s_end_to_end")}

    expected, pool_dups, qid_names = load_pool(a.pool)
    kev, kev_written = load_answers(a.kev, "kev")
    native, native_written = load_answers(a.native, "native")
    report["key_sets"] = {
        "pool_questions": len(expected), "pool_duplicate_keys": pool_dups,
        "pool_unique_qid_names": len(qid_names),
        "kev_rows_written": kev_written, "kev_unique_keys": len(kev),
        "native_rows_written": native_written, "native_unique_keys": len(native),
        "kev_missing_vs_pool": len(set(expected) - set(kev)), "kev_extra_vs_pool": len(set(kev) - set(expected)),
        "native_missing_vs_pool": len(set(expected) - set(native)), "native_extra_vs_pool": len(set(native) - set(expected)),
        "kev_native_intersection": len(set(kev) & set(native)),
    }
    if set(expected) != set(kev) or set(expected) != set(native):
        raise SystemExit("KEY SET MISMATCH — see report key_sets")

    # per side: official served distribution, labels, field-level abstain
    def side(data, name):
        s = {}
        for key, ans in data.items():
            served = official_served(ans, list(ans.get("probabilities", {})))
            known = {k: v for k, v in served.items() if k != "__unknown__"}
            top_known = max(known, key=known.get)
            s[key] = {"type": "boolean" if ans["type"] == "noul" or set(ans.get("probabilities", {})) == {"true", "false"} else "choice",
                      "label_full": label_of(served), "label_known": top_known,
                      "top_known_name": top_known, "top_known_p": known[top_known],
                      "u": float(ans["unknown_probability"]), "abstained_field": bool(ans.get("abstained")),
                      "n_ties": int(max(collections.Counter(round(v, 15) for v in served.values()).values()) > 1),
                      "served": served, "raw": ans}
        return s
    S_kev, S_nat = side(kev, "kev"), side(native, "native")

    # field-level abstain flag vs canonical argmax, per side
    for name, S in (("kev", S_kev), ("native", S_nat)):
        mism = [k for k, v in S.items() if v["abstained_field"] != (v["label_full"] == "__unknown__")]
        report[f"{name}_abstained_field_vs_canonical_mismatches"] = len(mism)

    # flips and distributions
    full_flip = known_flip = abstain_flip = 0
    full_flip_keys, known_keys, abstain_keys = [], [], []
    du, dtop, dtrue, draw_true = [], [], [], []
    rows = []
    for k in sorted(expected):
        N, K = S_nat[k], S_kev[k]
        flipped = N["label_full"] != K["label_full"]
        if flipped: full_flip += 1; full_flip_keys.append(k)
        if N["label_full"] != "__unknown__" and K["label_full"] != "__unknown__" and N["label_full"] != K["label_full"]:
            known_flip += 1; known_keys.append(k)
        if (N["label_full"] == "__unknown__") != (K["label_full"] == "__unknown__"):
            abstain_flip += 1; abstain_keys.append(k)
        du.append(abs(N["u"] - K["u"]))
        dtop.append(abs(N["top_known_p"] - K["top_known_p"]))
        if N["type"] == "boolean":
            dtrue.append(abs(N["served"]["true"] - K["served"]["true"]))
            # raw-level transform check: native keeps raw scores; kev stores noul = raw_true + 0.5*raw_u
            rt = N["raw"].get("scores", {}).get("true")
            if rt is not None:
                draw_true.append(abs((K["raw"]["noul"] - 0.5 * K["u"]) - rt))
        rows.append({"key": "|".join(map(str, k)), "type": N["type"],
                     "native_label_full": N["label_full"], "kev_label_full": K["label_full"],
                     "native_label_known": N["label_known"], "kev_label_known": K["label_known"],
                     "native_u": N["u"], "kev_u": K["u"],
                     "native_top_known_p": N["top_known_p"], "kev_top_known_p": K["top_known_p"],
                     "native_abstained_field": N["abstained_field"], "kev_abstained_field": K["abstained_field"],
                     "full_flip": flipped})

    ov = set(known_keys) & set(abstain_keys)
    by_type_known = collections.Counter(S_nat[k]["type"] for k in known_keys)
    by_type_abst = collections.Counter((S_nat[k]["label_full"], S_kev[k]["label_full"]) for k in abstain_keys)
    report["flips"] = {
        "full_prediction_flips_incl_unknown": full_flip,
        "known_class_prediction_flips": known_flip, "known_flip_by_type": dict(by_type_known),
        "abstain_flips_exactly_one_unknown": abstain_flip, "abstain_flip_breakdown": {f"{a}->{b}": c for (a, b), c in by_type_abst.items()},
        "known_abstain_overlap": len(ov), "unique_flipped_composite_keys": len(set(full_flip_keys)),
        "unique_flipped_qid_names": len({k[2] for k in full_flip_keys}),
    }
    report["distributions"] = {
        "unknown_prob_diff": dist(du), "top_known_prob_diff": dist(dtop),
        "boolean_served_true_diff": dist(dtrue), "boolean_raw_true_diff_kev_noul_vs_native_scores": dist(draw_true),
    }
    n_ties = sum(1 for S in (S_nat, S_kev) for v in S.values() if v["n_ties"])
    report["exact_tie_rows_across_both_sides"] = n_ties

    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, "reaudit_exp10k")
    with open(base + "_per_question.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(base + "_summary.json", "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("\nper-question:", base + "_per_question.jsonl")
    print("summary:", base + "_summary.json")


if __name__ == "__main__":
    # the official evaluator import must come after argparse so --help doesn't need it
    sys.path.insert(0, os.environ["IMAEV_SCRIPTS"] if "IMAEV_SCRIPTS" in os.environ else
                    "/mnt/bn/product-lib-rd-merlin/tan.yuanhong/imajev/imajev/scripts")
    from evaluate_text_decisions import canonical_distribution   # noqa: E402
    main()
