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
                  f"mem_used max={max(mem)}MiB over {len(self.samples)} samples", flush=True)


def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))] if vals else 0


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
    from vision_decision.scoring import compile_question, cyclic_offsets
    from vision_decision.jev_api import to_request_with_plan

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
            tasks.append({"prompt": prompt, "labels": labels})
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
    batch_shapes = []
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
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    t_fwd = t_jit = 0.0
    first = True
    done = 0
    with UtilSampler() as sampler:
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
            batch_shapes.append((len(idxs), int(max(len(t) for t in inputs["input_ids"]))))
            done += len(idxs)
    t_e2e = time.time() - t_all
    summary = {"side": "native", "requests": len(pool), "questions": n_q, "tasks": len(tasks),
               "cold_load_s": round(t_cold, 2), "jit_first_batch_s": round(t_jit, 2),
               "render_s": round(t_render, 2), "forward_s": round(t_fwd, 2), "end_to_end_s": round(t_e2e, 2),
               "questions_per_s_end_to_end": round(n_q / t_e2e, 1), "questions_per_s_forward": round(n_q / t_fwd, 1),
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

    t0 = time.time()
    tok, model, rep = load_imajev(a.base, a.adapter, "cuda", dtype=torch.bfloat16,
                                  cuda_graphs=True, fused=True, merge=True)
    server = ImajevServer(base=a.base, adapter=a.adapter, tok=tok, model=model, device="cuda", max_length=4096)
    t_cold = time.time() - t0

    # JIT: encode one request, run it through the server, capture the pending graphs
    t1 = time.time()
    encs0, req0, plan0, meta0 = imajev_encode(model.imajev, pool[0], rotations=1, max_state=4096, max_branch=4096)
    picks0 = [server.submit_enc(e).result() for e in encs0]
    model.graphs.capture_pending()
    server.wait_idle()
    t_jit = time.time() - t1
    stats_after_capture = model.graphs.stats() if model.graphs is not None else None
    print(f"[cold] load {t_cold:.1f}s; [jit] warm+capture {t_jit:.1f}s; graphs: {stats_after_capture}", flush=True)

    encoded = []
    t_enc = 0.0
    for p in pool:
        _t = time.time()
        encs, request, plan, meta = imajev_encode(model.imajev, p, rotations=1, max_state=4096, max_branch=4096)
        t_enc += time.time() - _t
        encoded.append((p, encs, request, plan, meta))

    batch_shapes = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    t_fwd = 0.0
    n_batches = 0
    with UtilSampler() as sampler:
        for p, encs, request, plan, meta in encoded:
            futures = [server.submit_enc(e) for e in encs]
            t2 = time.time()
            picked = [f.result() for f in futures]
            # no torch.cuda.synchronize() here: the model thread's own copy-out already syncs its stream, and a
            # main-thread sync can land inside a mid-load capture window (legacy stream vs capturing blocking stream)
            t_fwd += time.time() - t2
            for _, stats in picked:
                batch_shapes.append((stats["tokens"], stats["state_tokens"]))
            server.wait_idle()
            n_batches += 1
        t_e2e = time.time() - t_all
    # answers assembly is part of end-to-end scoring; time one full pass including score_request
    t3 = time.time()
    x0 = encoded[0]
    picks = [server.submit_enc(e).result() for e in x0[1]]
    server.wait_idle()
    score_request(model.imajev, x0[2], x0[3], x0[4], [pk for pk, _ in picks])
    t_score = time.time() - t3
    summary = {"side": "kev", "requests": len(pool), "questions": n_q,
               "cold_load_s": round(t_cold, 2), "jit_warm_capture_s": round(t_jit, 2),
               "encode_s": round(t_enc, 2), "forward_s": round(t_fwd, 2),
               "score_sample_s": round(t_score, 2), "end_to_end_s": round(t_e2e, 2),
               "questions_per_s_end_to_end": round(n_q / t_e2e, 1),
               "rows": len(batch_shapes),
               "state_tokens_p50": pct([b[1] for b in batch_shapes], .5),
               "state_tokens_p90": pct([b[1] for b in batch_shapes], .9),
               "state_tokens_max": max((b[1] for b in batch_shapes), default=0),
               "row_tokens_p50": pct([b[0] for b in batch_shapes], .5),
               "row_tokens_p90": pct([b[0] for b in batch_shapes], .9),
               "row_tokens_max": max((b[0] for b in batch_shapes), default=0),
               "batches": server.batches, "batched_requests": server.batched_requests,
               "prefix_cache": {"hits": server.prefix_cache.hits, "misses": server.prefix_cache.misses,
                                "oom_retries": server.prefix_cache.oom_retries},
               "graphs_final": model.graphs.stats() if model.graphs is not None else None,
               "peak_mem_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
               "gpu_util_avg_pct": round(sum(s[1] for s in sampler.samples) / len(sampler.samples), 1) if sampler.samples else None,
               "rotations": 1}
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
    r = sub.add_parser("run")
    r.add_argument("--side", choices=["native", "kev"], required=True)
    r.add_argument("--pool", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--adapter", help="kev-compatible adapter copy (kev side)")
    r.add_argument("--adapter-orig", help="original imajev adapter (native side)")
    r.add_argument("--out", help="summary JSON out")
    r.set_defaults(func=lambda x: None)
    a = ap.parse_args()
    if a.cmd == "build-pool":
        cmd_build_pool(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
