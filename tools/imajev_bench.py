#!/usr/bin/env python3
"""Bounded throughput bench on a fixed real request pool: the native imajev eval path (TorchDecision +
official CUDA kernels, unmerged fp32-delta LoRA) vs the kev adapter serving path (merged + fused + CUDA
graphs). Same pool, rot1, no calibration, exclusive GPU. Cold load and JIT are recorded separately from
the warm wall clock; forward and end-to-end are both reported.

  python tools/imajev_bench.py build-pool --data DEV.jsonl --records 50 --out pool.jsonl
  python tools/imajev_bench.py run --side native --base BASE --adapter-orig ADAPTER_ORIG --pool pool.jsonl --out native.json
  python tools/imajev_bench.py run --side kev    --base BASE --adapter ADAPTER       --pool pool.jsonl --out kev.json

Pool rows are exactly batch_eval's contract: sanitize field ids, strip non-protocol keys, chunk to <=8
questions (the native MAX_QUESTIONS). "First N records" in file order, deterministic."""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def env_assert():
    assert os.environ.get("IMAJEV_SRC"), "set IMAJEV_SRC to the imajev repo src dir"
    assert os.environ.get("IMAJEV_SCRIPTS"), "set IMAJEV_SCRIPTS to the imajev repo scripts dir"
    sys.path.insert(0, os.environ["IMAJEV_SRC"])
    sys.path.insert(0, os.environ["IMAJEV_SCRIPTS"])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_build_pool(a):
    records = [json.loads(l) for l in open(a.data)][:a.records]
    pool, line = [], 0
    for ridx, rec in enumerate(records):
        id_map, qs, used = {}, {}, set()
        for i, (qid, q) in enumerate(rec["questions"].items()):
            sid = qid if ID_RE.match(qid) else f"q{i}"
            if sid in used:
                sid = f"q{i}"
            id_map[sid] = qid
            used.add(sid)
            qs[sid] = {k: v for k, v in q.items() if k not in ("label", "target", "src", "_meta")}
        qids = list(qs)
        for ci in range(0, len(qids), 8):
            sub = {qid: qs[qid] for qid in qids[ci:ci + 8]}
            pool.append({"ridx": ridx, "chunk": ci // 8, "state": rec["state"], "questions": sub})
    with open(a.out, "w") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"pool": a.out, "records": len(records), "requests": len(pool),
                      "questions": sum(len(p["questions"]) for p in pool),
                      "sha256": sha256_file(a.out)}, indent=1))


def _sample_records(recs, target, rng):
    """Whole-record seeded sample for one dev set: sort records by state json char length, split into
    terciles (short/mid/long), shuffle each with the seeded RNG, then draw round-robin (one record per
    tercile per round) until the question target is met. Records are never reused or repeated; the last
    record may overshoot the target. -> (used lines, questions, per-tercile questions, tercile bounds)."""
    lengths = [len(json.dumps(r["state"])) for r in recs]
    order = sorted(range(len(recs)), key=lambda i: lengths[i])
    ter = [list(order[:len(order) // 3]), list(order[len(order) // 3:2 * len(order) // 3]),
           list(order[2 * len(order) // 3:])]
    for t in ter:
        rng.shuffle(t)
    used, nq, per_t = [], 0, [0, 0, 0]
    ti = 0
    while nq < target and any(ter):
        while not ter[ti]:
            ti = (ti + 1) % 3
        ri = ter[ti].pop(0)
        used.append(ri)
        nq += len(recs[ri]["questions"])
        per_t[ti] += len(recs[ri]["questions"])
        ti = (ti + 1) % 3
    return used, nq, per_t, [lengths[order[len(order) // 3]], lengths[order[2 * len(order) // 3]]]


def cmd_build_pool2(a):
    """Seeded, reproducible expanded pool from both dev sets: whole records only (no record reused, no
    repeat-looping), stratified by state length terciles for short/mid/long coverage, chunked to <=8
    questions exactly like build-pool (sanitized ids, non-protocol keys stripped). A sidecar meta JSON
    records the seed, per-set provenance (rows carry qid_map to original qids), terciles and the pool sha256."""
    rng = random.Random(a.seed)
    meta_sets, pool, gidx = {}, [], 0
    for src, path, target in (("textile", a.textile, a.target // 2), ("beauty", a.beauty, a.target - a.target // 2)):
        recs = [json.loads(l) for l in open(path) if l.strip()]
        used, nq, per_t, ter_bounds = _sample_records(recs, target, rng)
        nchoice = sum(1 for ri in used for q in recs[ri]["questions"].values() if q.get("type") == "choice")
        meta_sets[src] = {"file": path, "records_available": len(recs), "records_used": len(used),
                          "used_record_lines": sorted(used), "questions": nq, "choice_questions": nchoice,
                          "state_char_len_tercile_bounds": ter_bounds, "questions_per_tercile": per_t}
        for ri in used:
            rec = recs[ri]
            id_map, qs, used_ids = {}, {}, set()
            for i, (qid, q) in enumerate(rec["questions"].items()):
                sid = qid if ID_RE.match(qid) else f"q{i}"
                if sid in used_ids:
                    sid = f"q{i}"
                id_map[sid] = qid
                used_ids.add(sid)
                qs[sid] = {k: v for k, v in q.items() if k not in ("label", "target", "src", "_meta")}
            qids = list(qs)
            for ci in range(0, len(qids), 8):
                sub = {qid: qs[qid] for qid in qids[ci:ci + 8]}
                pool.append({"ridx": gidx, "chunk": ci // 8, "src": src, "orec": ri,
                             "state": rec["state"], "questions": sub, "qid_map": {k: id_map[k] for k in sub}})
                gidx += 1
    with open(a.out, "w") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {"seed": a.seed, "target_questions": a.target, "sampling_rule":
            "whole records only; per set: sort by state json char length, terciles, seeded shuffle, round-robin draw to the per-set question target; chunks of <=8 questions keep the record state",
            "sets": meta_sets, "requests": len(pool),
            "questions": sum(len(p["questions"]) for p in pool), "sha256": sha256_file(a.out)}
    with open(a.meta, "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items() if k != "sets"}, indent=1), flush=True)
    for s, m in meta_sets.items():
        print(f"[{s}] records_used={len(m['used_record_lines'])} questions={m['questions']} choice={m['choice_questions']} q_per_tercile={m['questions_per_tercile']}", flush=True)



class UtilSampler:
    """Background sampler of GPU utilization and used memory (pynvml); best effort."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml = pynvml
        except Exception:
            self.h = None

    def _loop(self):
        while not self.stop.is_set():
            if self.h is not None:
                try:
                    u = self.nvml.nvmlDeviceGetUtilizationRates(self.h)
                    m = self.nvml.nvmlDeviceGetMemoryInfo(self.h)
                    self.samples.append((round(time.time(), 3), u.gpu, round(m.used / 2**20)))
                except Exception:
                    pass
            self.stop.wait(0.5)

    def __enter__(self):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.samples:
            utils = [s[1] for s in self.samples]
            mems = [s[2] for s in self.samples]
            print(f"[gpu] util avg={sum(utils) / len(utils):.0f}% max={max(utils)}% "
                  f"mem_used max={max(mems)}MiB over {len(self.samples)} samples", flush=True)


def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))] if vals else 0


def to_answer(result, qtype):
    """Mirrors batch_eval.to_answer: internal Result -> response-shaped dict (scores kept for cross-checks)."""
    scores = result.scores
    unknown = scores.get("__unknown__", 0.0)
    known = {k: v for k, v in scores.items() if k != "__unknown__"}
    total = sum(known.values())
    if total > 0:
        known = {k: v / total for k, v in known.items()}
    base = {"unknown_probability": unknown, "abstained": result.status == "abstained",
            "scores": dict(scores), "raw_logits": dict(result.raw_logits)}
    if qtype == "noul":
        base.update({"type": "noul", "noul": scores.get("true", 0.0) + 0.5 * unknown})
    else:
        base.update({"type": "choice", "probabilities": known})
    return base


def cmd_run(a):
    env_assert()
    import torch
    pool = [json.loads(l) for l in open(a.pool)]
    n_q = sum(len(p["questions"]) for p in pool)
    print(f"pool: {len(pool)} requests, {n_q} questions, sha {sha256_file(a.pool)}", flush=True)
    if a.side == "native":
        run_native(a, pool, n_q)
    else:
        run_kev(a, pool, n_q)


def run_native(a, pool, n_q):
    """The official native eval path (batch_eval.py): TorchDecision + unmerged adapter + fp32 readout,
    hub kernels enabled, length-sorted batches of 32 through candidate_logits_batch."""
    import torch
    from peft import PeftModel
    from torch_decision import TorchDecision
    from vision_decision.scoring import compile_question, cyclic_offsets, result_from_logits
    from vision_decision.jev_api import to_request_with_plan

    # kernel-enablement evidence: the official eval config runs with the hub-kernels package on the path
    # (transformers uses it for the hybrid backbone when importable and silently falls back otherwise)
    try:
        import kernels
        print(f"[cfg] hub kernels ENABLED (kernels={getattr(kernels, '__version__', '?')}, path={kernels.__file__})", flush=True)
    except Exception as e:
        print(f"[cfg] hub kernels NOT importable ({e}) — slow fallback, NOT the official-kernels config", flush=True)

    t0 = time.time()
    engine = TorchDecision(a.base, "cuda", dtype=torch.bfloat16)
    engine.model = PeftModel.from_pretrained(engine.model, a.adapter_orig).eval()
    engine.enable_readout(a.adapter_orig, trainable=False)
    t_cold = time.time() - t0
    print(f"[cold] native engine loaded in {t_cold:.1f}s (codes={engine.codes}, layout={engine.prompt_layout})", flush=True)

    # build tasks exactly like batch_eval.build_tasks (rot1)
    tasks = []
    for p in pool:
        request, _ = to_request_with_plan({"state": p["state"], "questions": p["questions"]}, request_id="bench")
        for field in request.fields:
            header, choices, texts = compile_question(field, p["state"], engine.prompt_layout)
            labels = engine.labels(len(choices), 0)
            offset = cyclic_offsets(len(choices), 1)[0]
            prompt = header + "\n".join(f"{lab}: {txt}" for lab, txt in zip(labels, (texts[offset:] + texts[:offset])))
            tasks.append({"prompt": prompt, "labels": labels, "ridx": p["ridx"], "chunk": p["chunk"],
                          "qid": field.id, "qtype": field.type, "choices": choices})
    t_render0 = time.time()
    rendered = []
    for t in tasks:
        r = engine.render(t["prompt"], 0)
        rendered.append((r, [], engine.label_ids(r, t["labels"]), None))
    t_render = time.time() - t_render0
    # token length distribution (batch encode, cheap)
    try:
        _lens = sorted(len(x) for x in engine.processor.tokenizer([r[0] for r in rendered], add_special_tokens=False)["input_ids"])
        print(f"[lens] n={len(_lens)} p50={pct(_lens, .5)} p90={pct(_lens, .9)} p99={pct(_lens, .99)} max={_lens[-1]} tokens", flush=True)
    except Exception as exc:
        print(f"[lens] skipped: {exc}", flush=True)

    order = sorted(range(len(tasks)), key=lambda i: len(rendered[i][0]))
    batches = [order[bi:bi + 32] for bi in range(0, len(order), 32)]

    def native_pass(score):
        """One full-pool pass (length-sorted batches of 32 through candidate_logits_batch). Returns
        (t_fwd, t_jit, t_score, n_answered, n_abstained, batch_shapes). Scoring runs whenever score=True
        so every pass takes the same path; answers are only written by the timed pass."""
        from queue import Queue
        q = Queue(maxsize=3)

        def collate_worker():
            for idxs in batches:
                batch = [rendered[i] for i in idxs]
                inputs, token_ids_list, _targets = engine.collate(batch)
                q.put((idxs, inputs, token_ids_list))
            q.put(None)

        collate_thread = threading.Thread(target=collate_worker, daemon=True)
        collate_thread.start()
        t_fwd = t_jit = t_score = 0.0
        first = True
        n_answered = n_abstained = 0
        shapes = []
        while True:
            item = q.get()
            if item is None:
                break
            idxs, inputs, token_ids_list = item
            _t = time.time()
            with torch.inference_mode():
                outs = engine.candidate_logits_batch(inputs, token_ids_list)
            outs = [x.cpu() for x in outs]
            torch.cuda.synchronize()
            _dt = time.time() - _t
            if first:
                t_jit = _dt
                first = False
                print(f"[jit] first native batch ({len(idxs)} tasks): {_dt:.2f}s", flush=True)
            t_fwd += _dt
            if score:
                # native scoring/answer assembly inside the measured window (D2H already synced above)
                _ts = time.time()
                for k, i in enumerate(idxs):
                    t = tasks[i]
                    result = result_from_logits(t["choices"], outs[k], token_ids=token_ids_list[k])
                    answers_f.write(json.dumps({"ridx": t["ridx"], "chunk": t["chunk"], "qid": t["qid"],
                                                "type": t["qtype"], "answer": to_answer(result, t["qtype"])},
                                               ensure_ascii=False) + "\n")
                    n_answered += result.status == "answered"
                    n_abstained += result.status == "abstained"
                t_score += time.time() - _ts
            shapes.append((len(idxs), int(max(len(t) for t in inputs["input_ids"]))))
            print(f"[batch{'/timed' if score else '/warm'}] {len(shapes)}/{len(batches)} n={len(idxs)} "
                  f"maxtok={shapes[-1][1]} {_dt:.2f}s", flush=True)
        return t_fwd, t_jit, t_score, n_answered, n_abstained, shapes

    t_warm = None
    if a.warmup_pass:
        _tw = time.time()
        native_pass(score=False)
        t_warm = time.time() - _tw
        print(f"[warmup] native full pool pass {t_warm:.1f}s (untimed; JIT/autotune shapes absorbed here)", flush=True)

    answers_f = open(a.answers_out, "w") if a.answers_out else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    with UtilSampler() as sampler:
        t_fwd, t_jit, t_score, n_answered, n_abstained, batch_shapes = native_pass(score=answers_f is not None)
    t_e2e = time.time() - t_all
    if answers_f is not None:
        answers_f.close()
        print(f"[answers] native: {n_answered + n_abstained} rows written ({n_answered} answered, {n_abstained} abstained)", flush=True)
    summary = {"side": "native", "requests": len(pool), "questions": n_q, "tasks": len(tasks),
               "cold_load_s": round(t_cold, 2), "jit_first_batch_s": round(t_jit, 2),
               "render_s": round(t_render, 2), "warmup_pass_s": round(t_warm, 2) if t_warm is not None else None,
               "forward_s": round(t_fwd, 2), "score_s": round(t_score, 2),
               "scored_in_window": answers_f is not None,
               "end_to_end_s": round(t_e2e, 2),
               "questions_per_s_end_to_end": round(n_q / t_e2e, 1), "questions_per_s_forward": round(n_q / t_fwd, 1),
               "answers_written": n_answered + n_abstained, "answered": n_answered, "abstained": n_abstained,
               "batches": len(batch_shapes), "batch_sizes": sorted({b[0] for b in batch_shapes}),
               "batch_max_tokens_p50": pct([b[1] for b in batch_shapes], .5),
               "batch_max_tokens_max": max((b[1] for b in batch_shapes), default=0),
               "peak_mem_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
               "gpu_util_avg_pct": round(sum(s[1] for s in sampler.samples) / len(sampler.samples), 1) if sampler.samples else None,
               "rotations": 1}
    print(json.dumps(summary, indent=1), flush=True)
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)


def run_kev(a, pool, n_q):
    import torch
    from kev.imajev_adapter import imajev_encode, score_request, load_imajev
    from kev.imajev_serve import ImajevServer

    if a.max_batch:
        os.environ["KEV_MAX_BATCH"] = str(a.max_batch)   # serve-layer switch, read at server construction
    if a.prefix_size:
        os.environ["KEV_PREFIX_CACHE"] = str(a.prefix_size)   # prefix-cache LRU capacity in states

    t0 = time.time()
    tok, model, rep = load_imajev(a.base, a.adapter, "cuda", dtype=torch.bfloat16,
                                  cuda_graphs=True, fused=True, merge=True)
    server = ImajevServer(base=a.base, adapter=a.adapter, tok=tok, model=model, device="cuda", max_length=4096)
    t_cold = time.time() - t0

    # JIT: encode one request, run it through the server, let the model thread capture at its own idle point.
    # NEVER call model.graphs.capture_pending() off the model thread: it races the model thread's idle
    # auto-capture (imajev_serve._work), and concurrent captures abort the process. wait_idle blocks until
    # the queue is drained and no capture is pending or mid-flight.
    t1 = time.time()
    encs0, req0, plan0, meta0 = imajev_encode(model.imajev, {"state": pool[0]["state"], "questions": pool[0]["questions"]},
                                              rotations=1, max_state=4096, max_branch=4096)
    picks0 = [server.submit_enc(e).result() for e in encs0]
    server.wait_idle()
    t_jit = time.time() - t1
    print(f"[cold] load {t_cold:.1f}s; [jit] warm+capture {t_jit:.1f}s; graphs: "
          f"{model.graphs.stats() if model.graphs is not None else None}", flush=True)

    encoded = []
    t_enc = 0.0
    for p in pool:
        _t = time.time()
        encs, request, plan, meta = imajev_encode(model.imajev, {"state": p["state"], "questions": p["questions"]},
                                                  rotations=1, max_state=4096, max_branch=4096)
        t_enc += time.time() - _t
        encoded.append((p, encs, request, plan, meta))

    def score_one(i, picked, t_submit, collect):
        """Readout + answer assembly for one completed request (in whatever window the caller opened);
        returns (score_s, latency_s, answer row or None, missing or None, entries, answered, abstained)."""
        p, encs, request, plan, meta = encoded[i]
        t2 = time.time()
        resp = score_request(model.imajev, request, plan, meta, [pk for pk, _ in picked])
        t_s = time.time() - t2
        direct = set(resp["answers"])
        owned = {fid for entry in (plan or {}).values() for _, fid in entry["labels"]}
        missing = set(p["questions"]) - direct - owned
        entries = len(resp["answers"])
        n_ans = sum(1 for v in resp["answers"].values() if not v.get("abstained"))
        n_abs = sum(1 for v in resp["answers"].values() if v.get("abstained"))
        row = {"ridx": p["ridx"], "chunk": p["chunk"], "src": p.get("src"),
               "n_questions": len(p["questions"]), "answers": resp["answers"]} if collect else None
        return t_s, time.time() - t_submit, row, (missing or None), entries, n_ans, n_abs

    def full_pass(collect):
        """One pass over the whole pool at a.concurrency; scoring always runs (same shapes/paths), answers
        and coverage are only collected when collect=True. Latencies are submit -> last answer, always."""
        lat, enc_stats, answers, missing_all = [], [], [], []
        t_fwd = t_score = 0.0
        tot = [0, 0, 0]   # entries, answered, abstained
        if a.concurrency > 1:
            # bounded C-N window: up to N requests in flight, the model thread batches whatever is queued;
            # a request is scored (readout + assembly, in-window) when its last encoding completes
            from concurrent.futures import FIRST_COMPLETED, wait
            nxt, active, fut_req, remaining, t_sub, picks_acc = 0, 0, {}, {}, {}, {}

            def submit_next():
                nonlocal nxt
                if nxt >= len(encoded):
                    return False
                t_sub[nxt] = time.time()
                remaining[nxt] = len(encoded[nxt][1])
                for e in encoded[nxt][1]:
                    fut_req[server.submit_enc(e)] = nxt
                nxt += 1
                return True

            while active < a.concurrency and submit_next():
                active += 1
            while fut_req:
                done, _ = wait(set(fut_req), return_when=FIRST_COMPLETED)
                for f in done:
                    r = fut_req.pop(f)
                    pk, stats = f.result()   # one encoding's (picks, stats)
                    enc_stats.append((stats["tokens"], stats["state_tokens"]))
                    picks_acc.setdefault(r, []).append((pk, stats))
                    remaining[r] -= 1
                    if remaining[r] == 0:
                        active -= 1
                        s, t_lat, row, miss, e_, a_, b_ = score_one(r, picks_acc.pop(r), t_sub[r], collect)
                        t_score += s
                        lat.append(t_lat)
                        if collect and row is not None:
                            answers.append(row)
                        if miss:
                            missing_all.append({"ridx": encoded[r][0]["ridx"], "chunk": encoded[r][0]["chunk"], "missing": sorted(miss)})
                        tot[0] += e_; tot[1] += a_; tot[2] += b_
                        if submit_next():
                            active += 1
            server.wait_idle()
        else:
            for i, (p, encs, request, plan, meta) in enumerate(encoded):
                t_sub = time.time()
                futures = [server.submit_enc(e) for e in encs]
                t2 = time.time()
                picked = [f.result() for f in futures]
                # no main-thread sync: the model thread's copy-out already syncs its stream, and a sync here
                # can land inside a mid-load capture window (legacy stream vs capturing blocking stream)
                t_fwd += time.time() - t2
                for _, stats in picked:
                    enc_stats.append((stats["tokens"], stats["state_tokens"]))
                s, t_lat, row, miss, e_, a_, b_ = score_one(i, picked, t_sub, collect)
                t_score += s
                lat.append(t_lat)
                if collect and row is not None:
                    answers.append(row)
                if miss:
                    missing_all.append({"ridx": p["ridx"], "chunk": p["chunk"], "missing": sorted(miss)})
                tot[0] += e_; tot[1] += a_; tot[2] += b_
                server.wait_idle()
        return {"lat": lat, "enc_stats": enc_stats, "t_fwd": t_fwd, "t_score": t_score,
                "answers": answers, "missing": missing_all, "tot": tot}

    # optional full-pool warmup pass (untimed, identical submit/score path) — the cold/warm definition is:
    # cold = fresh process, empty prefix cache, graphs uncaptured; warm = after this pass (prefix cache holds
    # its LRU working set, graphs buckets that appeared during the pass are captured). First-time Triton
    # autotune shapes land in the warmup pass, not the timed one.
    t_warm = None
    if a.warmup_pass:
        t1 = time.time()
        full_pass(collect=False)
        server.wait_idle()
        t_warm = time.time() - t1
        print(f"[warmup] full pool pass {t_warm:.1f}s; graphs: "
              f"{model.graphs.stats() if model.graphs is not None else None}; prefix hits/misses so far: "
              f"{server.prefix_cache.hits}/{server.prefix_cache.misses}", flush=True)

    h0, m0 = server.prefix_cache.hits, server.prefix_cache.misses
    b0 = len(server.batch_log)
    torch.cuda.reset_peak_memory_stats()   # query only; no sync — all CUDA stays on the model thread
    t_all = time.time()
    with UtilSampler() as sampler:
        res = full_pass(collect=True)
    t_e2e = time.time() - t_all
    t_write = 0.0
    if a.answers_out:
        t2 = time.time()
        with open(a.answers_out, "w") as f:
            for row in res["answers"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        t_write = time.time() - t2
    bl = server.batch_log[b0:]
    encs_b = [b["encs"] for b in bl]
    rows_b = [b["rows"] for b in bl]
    tok_b = [b["max_tokens"] for b in bl]
    lat_ms = sorted(round(x * 1000, 1) for x in res["lat"])
    summary = {"side": "kev", "requests": len(pool), "questions": n_q,
               "concurrency": a.concurrency, "max_batch": server.max_batch,
               "cold_load_s": round(t_cold, 2), "jit_warm_capture_s": round(t_jit, 2),
               "encode_s": round(t_enc, 2),
               "warmup_pass_s": round(t_warm, 2) if t_warm is not None else None,
               "forward_s": (round(t_e2e - res["t_score"], 2) if a.concurrency > 1 else round(res["t_fwd"], 2)),
               "forward_includes_queue_wait": a.concurrency > 1,
               "score_s": round(res["t_score"], 2),
               "end_to_end_s": round(t_e2e, 2),
               "questions_per_s_end_to_end": round(n_q / t_e2e, 1),
               "write_answers_s": round(t_write, 2),
               "request_latency_ms_p50": pct(lat_ms, .5), "request_latency_ms_p95": pct(lat_ms, .95),
               "request_latency_ms_max": lat_ms[-1] if lat_ms else None,
               "batches": len(bl),
               "encodings_per_batch_p50": pct(encs_b, .5), "encodings_per_batch_p90": pct(encs_b, .9),
               "encodings_per_batch_max": max(encs_b, default=0),
               "rows_per_batch_p50": pct(rows_b, .5), "rows_per_batch_p90": pct(rows_b, .9),
               "rows_per_batch_max": max(rows_b, default=0),
               "batch_maxtok_buckets": {"le1024": sum(1 for t in tok_b if t <= 1024),
                                        "1025_2048": sum(1 for t in tok_b if 1024 < t <= 2048),
                                        "gt2048": sum(1 for t in tok_b if t > 2048)},
               "encodings": len(res["enc_stats"]),
               "enc_tokens_p50": pct([b[0] for b in res["enc_stats"]], .5),
               "enc_tokens_p90": pct([b[0] for b in res["enc_stats"]], .9),
               "enc_tokens_max": max((b[0] for b in res["enc_stats"]), default=0),
               "state_prefix_tokens_p50": pct([b[1] for b in res["enc_stats"]], .5),
               "state_prefix_tokens_p90": pct([b[1] for b in res["enc_stats"]], .9),
               "state_prefix_tokens_max": max((b[1] for b in res["enc_stats"]), default=0),
               "prefix_cache_timed": {"hits": server.prefix_cache.hits - h0, "misses": server.prefix_cache.misses - m0},
               "prefix_cache_cumulative": {"hits": server.prefix_cache.hits, "misses": server.prefix_cache.misses,
                                           "oom_retries": server.prefix_cache.oom_retries,
                                           "size_states": server.prefix_cache.size},
               "graphs_final": model.graphs.stats() if model.graphs is not None else None,
               "peak_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
               "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20),
               "gpu_util_avg_pct": round(sum(s[1] for s in sampler.samples) / len(sampler.samples), 1) if sampler.samples else None,
               "answers_entries": res["tot"][0], "answered": res["tot"][1], "abstained": res["tot"][2],
               "coverage_missing_requests": len(res["missing"]), "coverage_ok": not res["missing"],
               "scored_in_window": True, "rotations": 1}
    print(json.dumps(summary, indent=1), flush=True)
    server.close()
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)



def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-pool")
    b.add_argument("--data", required=True)
    b.add_argument("--records", type=int, default=50)
    b.add_argument("--out", required=True)
    b2 = sub.add_parser("build-pool2", help="expanded seeded pool from both dev sets (whole records, length terciles)")
    b2.add_argument("--textile", required=True)
    b2.add_argument("--beauty", required=True)
    b2.add_argument("--target", type=int, default=10000, help="total question target (half per set)")
    b2.add_argument("--seed", type=int, default=20261001)
    b2.add_argument("--out", required=True)
    b2.add_argument("--meta", required=True, help="sidecar metadata JSON (seed, provenance, terciles, sha256)")
    r = sub.add_parser("run")
    r.add_argument("--side", choices=["native", "kev"], required=True)
    r.add_argument("--pool", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--adapter", help="kev-compatible adapter copy (kev side)")
    r.add_argument("--adapter-orig", help="original imajev adapter (native side)")
    r.add_argument("--out", help="summary JSON out")
    r.add_argument("--answers-out", help="write full per-question answers (JSONL) and include readout+answer "
                                         "assembly in the measured window (native: batch_eval-identical "
                                         "result_from_logits+to_answer; kev C1: score_request per request)")
    r.add_argument("--concurrency", type=int, default=1,
                   help="kev side: in-flight request window (1 = serial per request, C1 baseline)")
    r.add_argument("--warmup-pass", action="store_true",
                   help="one untimed full-pool pass (identical submit+score path) before the timed window "
                        "(both sides: native absorbs JIT/autotune, kev absorbs graph capture + cache)")
    r.add_argument("--max-batch", type=int, default=None,
                   help="kev side: override the model thread's max encodings per batch (KEV_MAX_BATCH; default 64)")
    r.add_argument("--prefix-size", type=int, default=None,
                   help="kev side: prefix-cache LRU capacity in distinct states (KEV_PREFIX_CACHE; default 4)")
    r.set_defaults(func=lambda x: None)
    a = ap.parse_args()
    if a.cmd == "build-pool":
        cmd_build_pool(a)
    elif a.cmd == "build-pool2":
        cmd_build_pool2(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
