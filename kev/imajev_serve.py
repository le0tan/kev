"""Standalone imajev serving entry: the Jev request/response protocol on kev's Qwen3.5 backend.

Run: IMAJEV_SRC=<imajev repo>/src python -m kev.imajev_serve --base <qwen3.5-4b snapshot> --adapter <kev-compatible copy> --port 8018

Deliberately isolated from kev.serve (default KEV behavior untouched): it loads no checkpoint and no head.pt,
never touches model.probs*/PointerHead, and shares only the state-prefix cache and the one-model-thread
batching pattern. Prompts, the candidate mapping, rotations and the unknown/abstain semantics come from the
native imajev modules through kev.imajev_adapter; scoring is hidden states -> the adapter's 256-code fp32
readout -> native result_from_logits / combine_rotations -> jev_api.to_response. Text-only: /v1/score
whitelists {state, questions, rotations}, so any visual input is rejected 422 before it reaches the model.
"""
import argparse, asyncio, atexit, os, queue, sys, threading, time, traceback
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .checkpoint import LoadOptions, fused_available
from .device import default_device, empty_cache, out_of_memory, sync
from .imajev_adapter import imajev_encode, load_imajev, score_request
from .serve import MAX_BATCH, PrefixCache

ALLOWED_PAYLOAD_KEYS = {"state", "questions", "rotations"}
ALLOWED_QUESTION_KEYS = {"type", "instructions", "criteria", "threshold"}


def check_text_only(payload):
    """The explicit visual rejection: anything beyond the text-only Jev payload shape — an images field, an
    attachment, extra keys — is refused 422 before the model sees it."""
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), dict) or set(payload) - ALLOWED_PAYLOAD_KEYS:
        raise HTTPException(422, "expected {state, questions[, rotations]}; visual input is not supported on this text-only backend")
    for name, q in payload["questions"].items():
        if not isinstance(q, dict) or set(q) - ALLOWED_QUESTION_KEYS:
            raise HTTPException(422, f"question {name!r} carries fields outside the text-only protocol: visual input is not supported on this backend")


@dataclass
class ImajevServer:
    """The loaded imajev model, the state-prefix cache, and the one model thread running hidden_picks_batch.
    Mirrors kev.serve.Server; the futures carry (picks, stats) per encoding instead of probabilities."""
    base: str
    adapter: str
    tok: object
    model: object
    device: str
    max_length: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    batches: int = 0
    batched_requests: int = 0

    def __post_init__(self):
        self.model.imajev  # fails fast if the adapter assets were not attached
        self.prefix_cache = PrefixCache(int(os.environ.get("KEV_PREFIX_CACHE", "4")),
                                        int(os.environ["KEV_PREFIX_MIN_TOKENS"]) if os.environ.get("KEV_PREFIX_MIN_TOKENS") else self.model.prefix_min_tokens)
        self.queue, self.stopping = queue.Queue(), threading.Event()
        self.switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0005)
        self.thread = threading.Thread(target=self._work, name="kev-imajev-model", daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def close(self):
        if self.stopping.is_set(): return
        self.stopping.set(); self.thread.join()
        while not self.queue.empty():
            self.queue.get_nowait()[1].set_exception(RuntimeError("the server stopped")); self.queue.task_done()
        sys.setswitchinterval(self.switch_interval)

    def submit_enc(self, enc):
        """Queue one encoding for the model thread; -> a Future of (picks, stats)."""
        if self.stopping.is_set(): raise HTTPException(503, "the server is stopping")
        done = Future()
        self.queue.put((enc, done))
        return done

    def _work(self):
        graphs = getattr(self.model, "graphs", None)
        while not self.stopping.is_set():
            try: batch = [self.queue.get(timeout=0.05)]
            except queue.Empty:
                if graphs is not None and graphs.capture_due(idle=True):
                    with self.lock: graphs.capture_pending(limit=1)
                continue
            while len(batch) < MAX_BATCH:
                try: batch.append(self.queue.get_nowait())
                except queue.Empty: break
            try:
                with self.lock: results = self._run([enc for enc, _ in batch])
            except Exception as e:   # every encoding of the batch gets the error; the thread lives on
                traceback.clear_frames(e.__traceback__)
                results = [e] * len(batch)
            for (_, done), result in zip(batch, results):
                (done.set_exception if isinstance(result, Exception) else done.set_result)(result)
                self.queue.task_done()
            if graphs is not None and graphs.capture_due(idle=False):
                with self.lock: graphs.capture_pending(limit=1)

    def _run(self, encs):
        """One batch through model.hidden_picks_batch, with the prefix cache and kev.serve's OOM retry."""
        sync(self.device); t = time.time()
        for retry in (False, True):
            keys, cached, keep = self.prefix_cache.plan(encs)
            try: picks, prefixes = self.model.hidden_picks_batch(encs, cached, keep); break
            except Exception as e:
                if retry or not self.prefix_cache.entries or not out_of_memory(e): raise
            cached = None; self.prefix_cache.clear(); self.prefix_cache.oom_retries += 1
            empty_cache(self.device)
        sync(self.device); dt = round((time.time() - t) * 1000, 1)
        self.prefix_cache.store(keys, cached, prefixes)
        self.batches += 1; self.batched_requests += len(encs)
        return [(p, {"tokens": len(enc["ids"]), "state_tokens": enc["seg"].count(0), "latency_ms": dt,
                     "prefix_cache_hit": c is not None})
                for enc, p, c in zip(encs, picks, cached)]

    def wait_idle(self):
        """Block until every submitted encoding is answered and no CUDA graph waits to be captured."""
        self.queue.join()
        graphs = getattr(self.model, "graphs", None)
        while graphs is not None and graphs.capture_due(idle=True): time.sleep(0.01)
        with self.lock: pass


app = FastAPI(title="kev-imajev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["server-timing"])


@app.middleware("http")
async def timing(request, call_next):
    started = time.perf_counter()
    resp = await call_next(request)
    resp.headers["server-timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"
    return resp


def server() -> ImajevServer:
    return app.state.server


@app.post("/v1/score")
async def score(payload: dict):
    """One Jev request {state, questions[, rotations]} -> {"model", "answers", "usage", "latency_ms"}.

    A request's questions may split into several prefix-group encodings; each is queued separately (they run
    in the same or adjacent batches), and the answers are assembled over all of them. latency_ms is the model
    time of the slowest encoding's batch."""
    check_text_only(payload)
    s = server()
    rotations = payload.get("rotations", 1)
    if isinstance(rotations, bool) or not isinstance(rotations, int) or not 1 <= rotations <= 255:
        raise HTTPException(422, f"rotations must be an integer in [1, 255]; got {rotations!r}")
    body = {k: payload[k] for k in ("state", "questions")}
    try:
        encs, request, plan, meta = imajev_encode(s.model.imajev, body, rotations=rotations,
                                                  max_state=s.max_length, max_branch=s.max_length)
    except ValueError as e:
        raise HTTPException(422, str(e))
    futures = [s.submit_enc(enc) for enc in encs]
    picks_list = await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
    answers = score_request(s.model.imajev, request, plan, meta, [p for p, _ in picks_list])
    return {"model": "imajev-kev", "answers": answers["answers"],
            "usage": {"input_tokens": sum(len(enc["ids"]) for enc in encs), "output_tokens": 0},
            "latency_ms": round(max(m["latency_ms"] for _, m in picks_list), 1)}


@app.get("/v1/models")
def models():
    """The imajev-on-kev serving card: base, adapter, device, backend and precision, graphs and prefix-cache
    stats, batch accounting."""
    s = server()
    graphs = getattr(s.model, "graphs", None)
    return {"models": [{"name": "imajev-kev", "base": s.base, "adapter": s.adapter, "device": s.device,
                        "backend": "torch", "dtype": s.model.dtype, "hybrid": s.model.hybrid,
                        "max_length_tokens": s.max_length,
                        "cuda_graphs": graphs.stats() if graphs is not None else None,
                        "prefix_cache": {"size": s.prefix_cache.size, "hits": s.prefix_cache.hits,
                                         "misses": s.prefix_cache.misses, "cached_states": len(s.prefix_cache.entries),
                                         "oom_retries": s.prefix_cache.oom_retries},
                        "batches": {"count": s.batches, "requests": s.batched_requests, "queued": s.queue.qsize()}}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="local Qwen3.5-4B snapshot directory")
    ap.add_argument("--adapter", required=True, help="kev-compatible adapter copy (tools/make_kev_compatible_adapter.py)")
    ap.add_argument("--max-length", type=int, default=4096, help="row (state+branch) token limit, the native max_length guard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8018)
    a = ap.parse_args()
    if not os.path.isdir(a.base) or not os.path.isdir(a.adapter):
        raise SystemExit("base snapshot and adapter copy must be local directories")
    if not os.environ.get("IMAJEV_SRC"):
        raise SystemExit(f"set IMAJEV_SRC to the imajev repository's src directory (the native scoring imports)")
    dev = default_device()
    opts = LoadOptions.from_env()
    if dev != "cpu" and opts.dtype is None: opts = replace(opts, dtype=torch.bfloat16)   # serving default, as kev.serve
    if dev == "cuda" and opts.cuda_graphs is None: opts = replace(opts, cuda_graphs=True)
    fused_default = dev == "cuda" and opts.fused is None
    if fused_default: opts = replace(opts, fused=fused_available())
    # the fused rewrite needs merged weights; otherwise keep the adapter unmerged, exactly as the native eval loads it
    tok, model, report = load_imajev(a.base, a.adapter, dev, opts=opts, cuda_graphs=bool(opts.cuda_graphs) and dev == "cuda",
                                     fused=bool(opts.fused), merge=bool(opts.fused) or None)
    print(f"adapter mapping: {report['tensors']} tensors on {report['modules']} modules, r={report['r']} "
          f"alpha={report['alpha']} peft={report['peft_version']} merged={bool(report) and bool(opts.fused)}")
    app.state.server = ImajevServer(a.base, a.adapter, tok, model, dev, a.max_length)
    print(f"serving imajev ({a.adapter} on {a.base}) on {dev} via torch ({model.dtype}) {a.host}:{a.port}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
