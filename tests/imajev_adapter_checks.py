"""Checks for the imajev-on-kev text adapter (kev/imajev_adapter.py), CPU first, GPU where it says so.

Run with the kev venv python; IMAJEV_SRC must point at the imajev repository's src/ (vision_decision lives
there) and IMAJEV_SCRIPTS at its scripts/ (torch_decision.py, the native eval engine).

  tokens    native render/tokenize vs kev.imajev_adapter.imajev_encode, per (field, rotation) row: reassembled
            prefix + branch must equal the native full token ids bit-for-bit, the processor (native prepare) and
            tokenizer (native prepare_fast) paths must agree, every rendered prompt must end with the decision
            suffix, the codebook token ids must equal verified_label_ids at the real decision position, row
            positions must continue the state, decide_idx must sit on each row's last token, and crafted
            over-limit inputs must raise ContextOverflow. No backbone weights (devbox-safe).
  codebook  ImajevAssets validation alone: codebook derivation, decision_readout.json binding, labels.
  mapping   (1) every adapter tensor copied bit-identical under the key remap; (2) the original adapter's files
            still match SHA256SUMS (read-only proof); (3) verify_adapter_mapping against the actually-loaded
            backbone's named_modules. Loads the 4B (worker CPU or GPU).
  forward   end-to-end parity: native TorchDecision (original adapter, unmerged LoRA + fp32 readout, the
            evaluate_decision_model_torch path) vs the kev backend (remapped copy, unmerged, kev readout) on the
            same records -> per (field, rotation) max|dlogit|, max|dprob|, argmax flips, abstain flips, and the
            assembled Jev answers vs the native to_response. --device cuda for GPU; --graphs exercises the CUDA
            graph passes (eager vs replayed comparison included); --fused the merged+fused rewrite; --long adds an
            over-graph-limit record so the eager prefix path runs; --small keeps two tiny records for CPU.
"""
import argparse, hashlib, importlib.util, json, os, sys, time
import torch

assert os.environ.get("IMAJEV_SRC"), "set IMAJEV_SRC to the imajev repository's src directory"
assert os.environ.get("IMAJEV_SCRIPTS"), "set IMAJEV_SCRIPTS to the imajev repository's scripts directory"
sys.path.insert(0, os.environ["IMAJEV_SRC"])
sys.path.insert(0, os.environ["IMAJEV_SCRIPTS"])

from kev.imajev_adapter import (DECISION_TAIL, ImajevAssets, imajev_encode, imajev_modules, load_imajev,
                                score_request, verify_adapter_mapping)
from kev.model import ContextOverflow, DecisionModel, load_tokenizer


def read_records(path, limit=0):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return rows[:limit] if limit else rows


def imajev_payloads(record, native):
    """One dev row -> Jev payloads, exactly as batch_eval.py/eval_imajev.py feed the server: field ids
    sanitized to the serving rule (else q{i}), non-protocol keys stripped from each question, then the
    questions chunked to <=8 per request (the native MAX_QUESTIONS)."""
    import re
    id_re = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    id_map, qs, used = {}, {}, set()
    for i, (qid, q) in enumerate(record["questions"].items()):
        sid = qid if id_re.match(qid) else f"q{i}"
        if sid in used:
            sid = f"q{i}"
        id_map[sid] = qid
        used.add(sid)
        qs[sid] = {k: v for k, v in q.items() if k not in ("label", "target", "src", "_meta")}
    chunks = []
    qids = list(qs)
    for i in range(0, len(qids), 8):
        sub = {qid: qs[qid] for qid in qids[i:i + 8]}
        chunks.append({"state": record["state"], "questions": sub})
    return chunks


def sample_records(rows, small=False):
    """A fixed mixed sample. Full: smallest, noul-only, most-choice, largest, longest-state. Small (CPU): one
    choice-bearing and one noul-only record, both with few questions."""
    def n_q(r): return len(r["questions"])
    def n_choice(r): return sum(1 for q in r["questions"].values() if q.get("type") == "choice")
    def state_chars(r): return len(json.dumps(r["state"], ensure_ascii=False))
    if small:
        with_choice = sorted((r for r in rows if n_choice(r)), key=lambda r: (n_q(r), state_chars(r)))
        noul_only = sorted((r for r in rows if not n_choice(r)), key=lambda r: (n_q(r), state_chars(r)))
        picked = ([("small_choice", with_choice[0])] if with_choice else []) + ([("small_noul", noul_only[0])] if noul_only else [])
        if len(picked) < 2:
            picked = [("small_0", sorted(rows, key=lambda r: (n_q(r), state_chars(r)))[0])]
        return [{"label": k, "record": r} for k, r in picked]
    picked = {"smallest": min(rows, key=lambda r: (n_q(r), state_chars(r))),
              "largest": max(rows, key=lambda r: (n_q(r), state_chars(r))),
              "most_choice": max(rows, key=lambda r: (n_choice(r), n_q(r))),
              "longest_state": max(rows, key=state_chars)}
    noul = [r for r in rows if not n_choice(r)]
    if noul:
        picked["noul_only"] = min(noul, key=lambda r: (n_q(r), state_chars(r)))
    order = ["smallest", "noul_only", "most_choice", "largest", "longest_state"]
    seen, out = set(), []
    for k in order:
        if k in picked and id(picked[k]) not in seen:
            seen.add(id(picked[k]))
            out.append({"label": k, "record": picked[k]})
    return out


def enc_rows(encs, meta):
    """Reassemble each row's full token ids from the encodings, in owner order: state prefix + its branch run."""
    built = []
    for enc, enc_meta in zip(encs, meta["encs"]):
        Ls = enc["seg"].count(0)
        assert enc["seg"][:Ls] == [0] * Ls, "the state segment must be seg 0 and first"
        prefix = enc["ids"][:Ls]
        assert enc["pos"][:Ls] == list(range(Ls)), "state positions must be 0..Ls-1"
        start = Ls
        for t, (j, offset) in enumerate(enc_meta["rows"], start=1):
            end = start
            while end < len(enc["seg"]) and enc["seg"][end] == t:
                end += 1
            assert end > start, f"row {t} has an empty branch"
            assert enc["pos"][start:end] == list(range(Ls, Ls + end - start)), "branch positions must continue the state"
            assert enc["decide_idx"][t - 1] == end - 1, "decide_idx must be the row's last token"
            built.append((j, offset, prefix + enc["ids"][start:end]))
            start = end
        assert start == len(enc["seg"]), "every token must belong to a row"
        assert enc["opt_idx"] == [[] for _ in enc_meta["rows"]], "imajev rows carry no pointer-head options"
    return built


def cmd_tokens(a):
    assets = ImajevAssets(a.base, a.adapter)
    native = imajev_modules()
    suffix = assets.tokenizer.encode(DECISION_TAIL, add_special_tokens=False)
    records = sample_records(read_records(a.data, a.records))
    n_rows = 0
    for item in records:
        for ci, payload in enumerate(imajev_payloads(item["record"], native)):
            for rotations in (1, 4):
                encs, request, plan, meta = imajev_encode(assets, payload, rotations=rotations, max_state=8192, max_branch=16384)
                built = enc_rows(encs, meta)
                expected = []
                for j, f in enumerate(request.fields):
                    header, choices, texts = native.compile_question(f, request.state, assets.prompt_layout)
                    labels = assets.labels(len(choices))
                    assert native.verified_label_ids(assets.tokenizer, assets.render(header), labels) == meta["token_ids"][j], \
                        f"{item['label']} chunk {ci} field {f.id}: codebook token ids differ from verified_label_ids"
                    for offset in native.cyclic_offsets(len(choices), rotations):
                        prompt = header + "\n".join(f"{l}: {t}" for l, t in zip(labels, native.rotate(texts, offset)))
                        rendered = assets.render(prompt)
                        ids = list(assets.tokenizer(rendered, add_special_tokens=False).input_ids)
                        assert ids[-len(suffix):] == suffix, "the decision suffix must close every rendered prompt"
                        expected.append((j, offset, ids))
                assert built == expected, f"{item['label']} chunk {ci} rotations={rotations}: reassembled rows differ from the native render"
                n_rows += len(expected)
                # the native prepare path (processor) must give the same ids the tokenizer path gives
                for j, f in enumerate(request.fields):
                    header, choices, texts = native.compile_question(f, request.state, assets.prompt_layout)
                    labels = assets.labels(len(choices))
                    prompt = header + "\n".join(f"{l}: {t}" for l, t in zip(labels, texts))
                    rendered = assets.render(prompt)
                    proc_ids = assets.processor(text=[rendered], images=None, return_tensors="pt")["input_ids"][0].tolist()
                    assert proc_ids == list(assets.tokenizer(rendered, add_special_tokens=False).input_ids), \
                        "native prepare (processor) and prepare_fast (tokenizer) disagree on text-only ids"
    print(f"tokens: {n_rows} rows bit-identical to the native render, codebook ids == verified_label_ids, "
          f"processor == tokenizer, suffix closes every prompt")
    try:
        imajev_encode(assets, {"state": {"blob": "x" * 600}, "questions": {"q": {"type": "noul", "instructions": "present?"}}},
                      max_state=16, max_branch=16384)
        raise SystemExit("an over-limit state did not raise ContextOverflow")
    except ContextOverflow:
        pass
    try:
        imajev_encode(assets, {"state": {}, "questions": {"q": {"type": "choice", "instructions": "pick",
                                                                "criteria": {str(i): "d" * 40 for i in range(64)}}}},
                      rotations=1, max_state=8192, max_branch=64)
        raise SystemExit("an over-limit row did not raise ContextOverflow")
    except ContextOverflow:
        pass
    print("tokens: over-limit state and row raise ContextOverflow")
    labels255 = assets.labels(255)
    assert labels255[0] == "A" and len(set(labels255)) == 255 and len(assets.codebook) == 256
    print(f"codebook: {len(assets.codebook)} codes, labels A.. up to 255 options + unknown, readout {tuple(assets.readout.shape)} fp32")


def cmd_codebook(a):
    assets = ImajevAssets(a.base, a.adapter)
    labels = assets.labels(255)
    print(json.dumps({"codes": len(assets.codebook), "readout_shape": list(assets.readout.shape),
                      "dtype": str(assets.readout.dtype), "layout": assets.prompt_layout,
                      "first": assets.codebook[0], "last": assets.codebook[-1],
                      "labels_255": [labels[0], labels[25], labels[-1]], "unknown": assets.unknown}))


def cmd_mapping(a):
    spec = importlib.util.spec_from_file_location("mkc", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "make_kev_compatible_adapter.py"))
    mkc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mkc)
    from safetensors.torch import load_file
    orig = load_file(f"{a.adapter_orig}/adapter_model.safetensors")
    copy = load_file(f"{a.adapter}/adapter_model.safetensors")
    assert len(orig) == len(copy), f"{len(orig)} original tensors vs {len(copy)} in the copy"
    for k, v in orig.items():
        nk = mkc.remap(k)
        assert nk in copy and copy[nk].dtype == v.dtype and torch.equal(v, copy[nk]), f"tensor {k} did not copy bit-identical"
    print(f"mapping: {len(orig)} tensors copied bit-identical under the key remap")
    sums = dict(line.split("  ", 1) for line in open(f"{a.adapter}/SHA256SUMS").read().splitlines())
    for name, want in sums.items():
        if name.startswith("original/"):
            fn = f"{a.adapter_orig}/{name.split('/', 1)[1]}"
            h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
            assert h == want, f"original adapter file changed: {fn}"
    print("mapping: every original adapter file still matches SHA256SUMS (read-only proof)")
    print(f"loading the backbone on {a.device} for the named_modules check ...", flush=True)
    tok = load_tokenizer(a.base)
    m = DecisionModel(a.base, tok, a.device, dtype=torch.bfloat16)
    from peft import PeftModel
    m.lm = PeftModel.from_pretrained(m.lm, a.adapter, torch_device=str(a.device))
    report = verify_adapter_mapping(a.adapter, m.lm)
    print(json.dumps(report, indent=1, default=str))


def native_prompt_rows(engine, payload, rotations):
    """(field index, offset, rendered, token_ids, labels) per row, the native eval's per-prompt path."""
    native = imajev_modules()
    request, _ = native.jev_api.to_request_with_plan(payload, request_id="check", max_options=engine.codes - 1)
    rows = []
    for j, f in enumerate(request.fields):
        header, choices, texts = native.compile_question(f, request.state, engine.prompt_layout)
        labels = engine.labels(len(choices), 0)
        for offset in native.cyclic_offsets(len(choices), rotations):
            prompt = header + "\n".join(f"{l}: {t}" for l, t in zip(labels, native.rotate(texts, offset)))
            rendered, images, token_ids = engine.render_example(None, prompt, labels)
            rows.append((j, offset, rendered, token_ids, labels))
    return request, rows


def native_logits(engine, rows, single=False):
    """candidate_logits_batch over token-budgeted batches (the evaluate_decision_model_torch batching);
    single=True runs every row alone, which quantifies the native path's own batch-composition noise."""
    keys, values = [], []
    budget = 1 if single else 16000

    def flush():
        if not batch:
            return
        inputs, token_ids, _ = engine.collate([(r[2], None, r[3], None) for r in batch])
        with torch.no_grad():
            out = engine.candidate_logits_batch(inputs, token_ids)
        for r, lg in zip(batch, out):
            keys.append((r[0], r[1]))
            values.append([float(x) for x in lg])
        batch.clear()

    batch, longest = [], 0
    for r in rows:
        n = len(engine.processor.tokenizer(r[2], add_special_tokens=False).input_ids)
        if batch and (max(longest, n) * (len(batch) + 1) > budget or len(batch) >= 48):
            flush()
            longest = 0
        batch.append(r)
        longest = max(longest, n)
    flush()
    return keys, values


def adapted_picks(model, payload, rotations, max_length):
    """imajev_encode -> hidden_picks_batch -> per (field, offset) logits over that field's labels."""
    native = imajev_modules()
    encs, request, plan, meta = imajev_encode(model.imajev, payload, rotations=rotations, max_state=max_length, max_branch=max_length)
    picks, _ = model.hidden_picks_batch(encs, [None] * len(encs), [False] * len(encs))
    per = {}
    for enc_picks, enc_meta in zip(picks, meta["encs"]):
        for (j, offset), h in zip(enc_meta["rows"], enc_picks):
            per[(j, offset)] = [float(x) for x in model.imajev.logits(h, meta["labels"][j])]
    return encs, request, plan, meta, picks, per


def field_result(native, choices, token_ids, passes):
    """The native result for one field from its (offset, logits) passes, exactly as score_request picks it."""
    if len(passes) == 1 and passes[0][0] == 0:
        return native.result_from_logits(choices, passes[0][1], token_ids=token_ids)
    return native.combine_rotations(choices, passes)


def diff_answers(native_answer, adapted_answer, path="$"):
    """Max numeric difference between two answer trees, plus the first structural mismatch."""
    bad = None
    worst = 0.0
    if isinstance(native_answer, dict) and isinstance(adapted_answer, dict):
        if set(native_answer) != set(adapted_answer):
            return 0.0, f"{path}: keys {sorted(set(native_answer) ^ set(adapted_answer))}"
        for k in native_answer:
            d, b = diff_answers(native_answer[k], adapted_answer[k], f"{path}.{k}")
            worst = max(worst, d)
            bad = bad or b
    elif isinstance(native_answer, (int, float)) and isinstance(adapted_answer, (int, float)):
        worst = abs(float(native_answer) - float(adapted_answer))
    elif native_answer != adapted_answer:
        bad = f"{path}: {native_answer!r} != {adapted_answer!r}"
    return worst, bad


def cmd_forward(a):
    from peft import PeftModel
    from torch_decision import TorchDecision
    if str(a.device) == "cpu":
        # CPU smoke: the DeltaNet conv/delta-rule kernels are CUDA-only; hide fla and causal_conv1d from
        # transformers so both engines fall back to the same torch-only path and the comparison stays fair
        import importlib.util
        real_find_spec = importlib.util.find_spec

        def _no_cuda_kernels(name, *args, **kwargs):
            if name.split(".")[0] in ("fla", "causal_conv1d"):
                return None
            return real_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = _no_cuda_kernels
        sys.modules.update({name: None for name in ("fla", "causal_conv1d")})
        print("cpu run: fla/causal_conv1d hidden, transformers torch-only fallback on both sides", flush=True)
    native = imajev_modules()
    print(f"[{time.strftime('%H:%M:%S')}] loading the native engine (original adapter, unmerged) on {a.device} ...", flush=True)
    t0 = time.time()
    engine = TorchDecision(a.base, a.device)
    engine.model = PeftModel.from_pretrained(engine.model, a.adapter_orig).eval()
    engine.enable_readout(a.adapter_orig, trainable=False)
    print(f"[{time.strftime('%H:%M:%S')}] native engine ready in {time.time() - t0:.0f}s "
          f"(codes={engine.codes}, layout={engine.prompt_layout})", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] loading the kev backend (remapped copy, unmerged) on {a.device} ...", flush=True)
    t0 = time.time()
    tok, model, report = load_imajev(a.base, a.adapter, a.device, dtype=torch.bfloat16,
                                     cuda_graphs=a.graphs, fused=a.fused, merge=a.fused or None)
    print(f"[{time.strftime('%H:%M:%S')}] kev backend ready in {time.time() - t0:.0f}s; "
          f"mapping {report['tensors']} tensors / {report['modules']} modules; graphs={model.graphs is not None}", flush=True)

    records = sample_records(read_records(a.data, a.records), small=a.small)
    expanded = []
    for item in records:
        for ci, payload in enumerate(imajev_payloads(item["record"], native)):
            expanded.append({"label": f"{item['label']}#{ci}", "record_payload": payload})
    if a.max_chunks:   # CPU smoke: at most max_chunks per record, first come first served
        kept, counts = [], {}
        for e in expanded:
            base = e["label"].split("#")[0]
            counts[base] = counts.get(base, 0) + 1
            if counts[base] <= a.max_chunks:
                kept.append(e)
        expanded = kept
    if a.long:
        first = imajev_payloads(records[-1]["record"], native)[0]
        padded = dict(first["state"])
        padded["padding"] = "long evidence. " * 600    # ~2.7k state tokens: past the graphed state pass, under max_length
        expanded.append({"label": "long_state", "record_payload": {**first, "state": padded}})

    report_rows, answer_diffs = [], []
    if a.graphs and model.graphs is not None:
        # warm the buckets eagerly, then capture the way the serving loop does, so the record loop below
        # replays captured graphs; the per-record comparison is then graphs-on vs the detached eager path
        adapted_picks(model, expanded[0]["record_payload"], a.rotations, a.max_length)
        model.graphs.capture_pending()
        print(f"[{time.strftime('%H:%M:%S')}] cuda graphs after capture: {model.graphs.stats()}", flush=True)
    graphs_obj = model.graphs
    for item in expanded:
        payload = item["record_payload"]
        n_request, prow = native_prompt_rows(engine, payload, a.rotations)
        n_keys, n_vals = native_logits(engine, prow, single=a.native_batch_1)
        encs, a_request, plan, meta, picks, per = adapted_picks(model, payload, a.rotations, a.max_length)
        built_keys = []
        for enc, enc_meta in zip(encs, meta["encs"]):
            built_keys += [(j, offset) for j, offset in enc_meta["rows"]]
        assert sorted(built_keys) == sorted((j, offset) for j, offset in n_keys), \
            f"{item['label']}: row keys differ between paths (groups may interleave row order; keyed join below)"
        for (j, offset), nlog in zip(n_keys, n_vals):
            alog = per[(j, offset)]
            n64, a64 = torch.tensor(nlog, dtype=torch.float64), torch.tensor(alog, dtype=torch.float64)
            pn, pa = torch.softmax(n64, -1), torch.softmax(a64, -1)
            rn = field_result(native, meta["choices"][j], meta["token_ids"][j], [(offset, nlog)])
            ra = field_result(native, meta["choices"][j], meta["token_ids"][j], [(offset, alog)])
            report_rows.append({"record": item["label"], "field_id": meta["fields"][j].id, "offset": offset,
                                "n_candidates": len(nlog),
                                "max_abs_dlogit": float((n64 - a64).abs().max()),
                                "max_abs_dprob": float((pn - pa).abs().max()),
                                "argmax_flip": int(n64.argmax()) != int(a64.argmax()),
                                "abstain_flip": rn.status != ra.status,
                                "native_status": rn.status, "adapted_status": ra.status})
        # graphs on/off: replayed picks vs the detached eager path's picks, same records
        if a.graphs and graphs_obj is not None:
            g_picks = picks
            model.graphs = None
            _, _, _, _, e_picks, _ = adapted_picks(model, payload, a.rotations, a.max_length)
            model.graphs = graphs_obj
            d = max((float((gh - eh).abs().max()) for gg, eg in zip(g_picks, e_picks) for gh, eh in zip(gg, eg)), default=0.0)
            answer_diffs.append({"record": item["label"], "kind": "graphs_vs_eager_picks", "max_abs_dlogit": d})
            print(f"[{time.strftime('%H:%M:%S')}] {item['label']}: graphs vs eager picks max|d|={d:.3e}, {graphs_obj.stats()}", flush=True)
        # end-to-end answers: native to_response over native logits vs score_request over kev picks
        n_by_key = dict(zip(n_keys, n_vals))
        n_results = []
        for j in range(len(meta["fields"])):
            offsets = sorted(o for (jj, o) in n_keys if jj == j)
            n_results.append(field_result(native, meta["choices"][j], meta["token_ids"][j],
                                          [(o, n_by_key[(j, o)]) for o in offsets]))
        n_resp = native.jev_api.to_response(a_request, n_results, plan=plan)
        a_resp = score_request(model.imajev, a_request, plan, meta, picks)
        worst, bad = diff_answers(n_resp["answers"], a_resp["answers"])
        answer_diffs.append({"record": item["label"], "kind": "answers_vs_native", "max_abs_diff": worst, "mismatch": bad})
        print(f"[{time.strftime('%H:%M:%S')}] {item['label']}: {len(n_keys)} rows in {len(encs)} prefix group(s) "
              f"(state tokens {[enc['seg'].count(0) for enc in encs]}), "
              f"max|dlogit|={max(r['max_abs_dlogit'] for r in report_rows if r['record'] == item['label']):.3e}, "
              f"answers max|diff|={worst:.3e}{' MISMATCH ' + bad if bad else ''}", flush=True)

    flips = sum(r["argmax_flip"] for r in report_rows)
    abstains = sum(r["abstain_flip"] for r in report_rows)
    worst_row = max(report_rows, key=lambda r: r["max_abs_dlogit"])
    worst_ans = max(d["max_abs_diff"] for d in answer_diffs if d["kind"] == "answers_vs_native")
    summary = {"rows": len(report_rows), "argmax_flips": flips, "abstain_flips": abstains,
               "max_abs_dlogit": worst_row["max_abs_dlogit"], "worst_row": {k: worst_row[k] for k in ("record", "field_id", "offset")},
               "max_abs_answer_diff": worst_ans, "mismatches": [d["mismatch"] for d in answer_diffs if d.get("mismatch")],
               "graphs": a.graphs, "fused": a.fused, "long": a.long, "rotations": a.rotations, "device": a.device,
               "native_batch_1": a.native_batch_1}
    print(json.dumps(summary, indent=1))
    if a.json_out:
        json.dump({"summary": summary, "rows": report_rows, "answers": answer_diffs}, open(a.json_out, "w"), indent=1)
        print(f"full report -> {a.json_out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["tokens", "codebook", "mapping", "forward"])
    ap.add_argument("--base", required=True, help="local Qwen3.5-4B snapshot directory")
    ap.add_argument("--adapter", help="kev-compatible adapter copy")
    ap.add_argument("--adapter-orig", help="original imajev adapter directory (the native side's)")
    ap.add_argument("--data", help="dev jsonl with Jev-shaped rows {state, questions}")
    ap.add_argument("--records", type=int, default=0, help="read only the first N rows before sampling")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rotations", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=4096, help="the row (state+branch) token limit, native max_length")
    ap.add_argument("--graphs", action="store_true", help="attach CUDA graphs and compare replayed vs eager picks")
    ap.add_argument("--fused", action="store_true", help="load merged + fused (serving rewrite) for this run")
    ap.add_argument("--long", action="store_true", help="append an over-graph-limit state record (eager prefix path)")
    ap.add_argument("--small", action="store_true", help="two tiny records only (CPU runs)")
    ap.add_argument("--max-chunks", type=int, default=0, help="keep at most this many 8-question chunks per record (0 = all)")
    ap.add_argument("--native-batch-1", action="store_true", help="run the native side one row per batch (controls for native batch noise)")
    ap.add_argument("--json-out", help="write the full row report here")
    a = ap.parse_args()
    {"tokens": cmd_tokens, "codebook": cmd_codebook, "mapping": cmd_mapping, "forward": cmd_forward}[a.command](a)


if __name__ == "__main__":
    main()
