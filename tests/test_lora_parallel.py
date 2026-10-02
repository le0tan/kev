"""CPU-only checks for optimizer-step LoRA collectives and global record accounting."""
import datetime
import json
import multiprocessing
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from kev import lora_parallel
from kev.train import microbatch_plan, parse_args
from test_unit import tiny_base   # the existing no-download hybrid backbone/tokenizer fixture


def _records(n):
    return [{"_meta": {"id": i}, "state": "state " * (1 + i % 5),
             "questions": {"q": {"type": "noul", "instructions": "question " * (1 + i % 3)}}}
            for i in range(n)]


def _args(length_sort, accum):
    return SimpleNamespace(batch=1, accum=accum, length_sort=length_sort, pass_tokens_max=0, shared_prefix=1)


def _steps(plan):
    steps, current = [], []
    for records, count, ends in plan:
        current.extend(r["_meta"]["id"] for r in records)
        if ends:
            steps.append((current, count)); current = []
    assert not current
    return steps


@pytest.mark.parametrize("length_sort", [0, 1])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 8, 9, 15, 16, 17])
@pytest.mark.parametrize("world", [2, 4])
def test_global_optimizer_steps_have_same_records_without_tail_repetition(length_sort, n, world):
    """Keep global batch eight when changing world size, including tails with idle ranks."""
    records = _records(n)
    single = _steps(microbatch_plan(records, _args(length_sort, 8), 1, 0, pad_to_world=False))
    plans = [microbatch_plan(records, _args(length_sort, 8 // world), world, rank, pad_to_world=False)
             for rank in range(world)]
    assert len({len(p) for p in plans}) == 1
    assert all([ends for _, _, ends in p] == [ends for _, _, ends in plans[0]] for p in plans)
    ranked = list(map(_steps, plans))
    assert all(len(steps) == len(single) for steps in ranked)
    seen = []
    for step, (reference, count) in enumerate(single):
        combined = [record for rank_steps in ranked for record in rank_steps[step][0]]
        assert sorted(combined) == sorted(reference) == list(range(step * 8, min((step + 1) * 8, n)))
        assert all(rank_steps[step][1] == count == len(combined) for rank_steps in ranked)
        seen.extend(combined)
    assert sorted(seen) == list(range(n))


def test_legacy_padded_tail_is_still_available():
    records = _records(3)
    args = _args(0, 4)
    legacy = [r["_meta"]["id"] for rank in range(2)
              for batch, _, _ in microbatch_plan(records, args, 2, rank) for r in batch]
    assert len(legacy) == 4 and sorted(set(legacy)) == [0, 1, 2]


@pytest.mark.parametrize("world,flags,error", [
    (2, [], "requires --lora_distributed"),
    (2, ["--lora_distributed", "1", "--full_ft", "1"], "separate training modes"),
    (2, ["--full_ft", "1", "--weights_dtype", "bf16", "--row_budget", "128"], "full-weight torchrun"),
    (2, ["--lora_distributed", "1", "--row_budget", "128", "--perm_kl", "0.1"], "--perm_kl"),
    (2, ["--lora_distributed", "1", "--row_budget", "128", "--anchor", "anchors.json", "--anchor_w", "0.1"], "--anchor_w"),
    (2, ["--lora_distributed", "1", "--device", "mps"], "not MPS"),
    (2, ["--lora_distributed", "1", "--p_none_pair", "0.5"], "sibling normalizer"),
])
def test_distributed_cli_rejects_unsafe_modes(monkeypatch, tmp_path, capsys, world, flags, error):
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setattr(sys, "argv", ["kev.train", "--out", str(tmp_path / "new"), *flags])
    with pytest.raises(SystemExit) as refused:
        parse_args()
    assert refused.value.code == 2
    assert error in capsys.readouterr().err


@pytest.mark.parametrize("world,flags", [(1, []), (1, ["--lora_distributed", "1"]), (2, ["--lora_distributed", "1"])])
def test_row_budget_allows_explicit_replicated_lora(monkeypatch, tmp_path, world, flags):
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setattr(sys, "argv", ["kev.train", "--device", "cpu", "--row_budget", "128", "--out", str(tmp_path / "new"), *flags])
    args = parse_args()
    assert args.row_budget == 128 and not args.full_ft


def _parameters(rank=0):
    return [torch.nn.Parameter(torch.tensor([0.3, -0.7], dtype=torch.float64) + rank),
            torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64) + rank),
            torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float64) + rank)]


def _backward_record(params, record, denominator, split):
    """A weighted product mean split into unequal question passes, normalized by the global record count."""
    i = record["_meta"]["id"]
    questions = list(range(1 + i % 4))
    weights = [0.4 + j for j in questions]
    mass = 0.25 + (i % 3)
    passes = [questions[j:j + 2] for j in range(0, len(questions), 2)] if split else [questions]
    for part in passes:
        loss = 0
        for j in part:
            x = torch.tensor([1 + i / 7, -0.3 + j / 5], dtype=torch.float64)
            prediction = params[0] @ x
            if i == 0: prediction = prediction + params[1]   # rank-local use; tail ranks may have no gradients
            loss = loss + weights[j] * (prediction - (i - j) / 5).square()
        (loss * mass / (sum(weights) * denominator)).backward()
    return len(passes)


def _gloo_updates(rank, rendezvous, out, length_sort):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    try:
        params = _parameters(rank)
        lora_parallel.broadcast_parameters(params)
        initial = [p.detach().clone() for p in params]
        opt = torch.optim.AdamW(params, lr=0.03, weight_decay=0.1)
        # Small buckets exercise multiple collectives without large allocations.
        lora_parallel.BUCKET_BYTES = 16
        trace, backwards = [], 0
        for batch, count, ends in microbatch_plan(_records(9), _args(length_sort, 4), 2, rank, pad_to_world=False):
            for record in batch: backwards += _backward_record(params, record, count, split=True)
            if not ends: continue
            lora_parallel.sum_gradients(params)
            gradients = [None if p.grad is None else p.grad.clone() for p in params]
            norm = torch.nn.utils.clip_grad_norm_(params, 0.12, error_if_nonfinite=True)
            opt.step(); opt.zero_grad(set_to_none=True)
            trace.append({"grads": gradients, "norm": norm, "params": [p.detach().clone() for p in params]})
        torch.save({"initial": initial, "trace": trace, "backwards": backwards}, out / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("length_sort", [0, 1])
def test_real_two_rank_sum_matches_single_update_with_unequal_passes_and_empty_tail(tmp_path, length_sort):
    """SUM precedes clipping/AdamW; globally unused parameters avoid weight decay; all replicas agree."""
    context = multiprocessing.get_context("spawn")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    workers = [context.Process(target=_gloo_updates, args=(rank, rendezvous, tmp_path, length_sort)) for rank in range(2)]
    for worker in workers: worker.start()
    try:
        for worker in workers: worker.join(timeout=90)
        assert all(not worker.is_alive() for worker in workers), "gloo gradient collective hung"
        assert all(worker.exitcode == 0 for worker in workers), [worker.exitcode for worker in workers]
    finally:
        for worker in workers:
            if worker.is_alive(): worker.terminate(); worker.join(timeout=10)
    outputs = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
    reference = _parameters()
    opt = torch.optim.AdamW(reference, lr=0.03, weight_decay=0.1)
    expected = []
    for batch, count, ends in microbatch_plan(_records(9), _args(length_sort, 8), 1, 0, pad_to_world=False):
        for record in batch: _backward_record(reference, record, count, split=False)
        if not ends: continue
        gradients = [None if p.grad is None else p.grad.clone() for p in reference]
        norm = torch.nn.utils.clip_grad_norm_(reference, 0.12, error_if_nonfinite=True)
        opt.step(); opt.zero_grad(set_to_none=True)
        expected.append({"grads": gradients, "norm": norm, "params": [p.detach().clone() for p in reference]})
    assert outputs[0]["backwards"] != outputs[1]["backwards"]
    for output in outputs:
        for initial, baseline in zip(output["initial"], _parameters()):
            torch.testing.assert_close(initial, baseline.detach(), rtol=0, atol=0)
        assert len(output["trace"]) == len(expected) == 2
        for observed, baseline in zip(output["trace"], expected):
            torch.testing.assert_close(observed["norm"], baseline["norm"], rtol=1e-12, atol=1e-12)
            for key in ("grads", "params"):
                for actual, correct in zip(observed[key], baseline[key]):
                    if correct is None: assert actual is None
                    else: torch.testing.assert_close(actual, correct, rtol=1e-12, atol=1e-12)
            assert observed["grads"][-1] is None
            torch.testing.assert_close(observed["params"][-1], torch.tensor(0.8, dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("length_sort,shared_prefix", [(0, 0), (1, 1)])
def test_real_trainer_saves_reloadable_lora_and_global_tail_metrics(tiny_base, tmp_path, length_sort, shared_prefix):
    """Two CPU ranks train weighted, split hybrid records; nine unique records include an idle final rank."""
    from safetensors.torch import load_file
    from kev.checkpoint import Checkpoint, read_meta
    from kev.data import load_records, materialize

    rows = [json.loads(line) for line in (tiny_base / "data.jsonl").read_text(encoding="utf-8").splitlines()[:9]]
    for i, row in enumerate(rows):
        row["_meta"] = {"id": str(i), "loss_weight": 0.5 + i / 4}
        for j in range(i % 3):
            row["questions"][f"extra{j}"] = dict(row["questions"]["angry"])
        for j, question in enumerate(row["questions"].values()): question["loss_weight"] = 1 + j
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "checkpoint"
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--rdzv-backend=c10d",
               "--rdzv-endpoint=127.0.0.1:0", "--local-addr=127.0.0.1", "--nproc_per_node=2", "-m", "kev.train",
               "--base", str(tiny_base / "base"), "--data", str(data), "--device", "cpu", "--out", str(out),
               "--lora_distributed", "1", "--lora", "4", "--head_dim", "16", "--batch", "1", "--accum", "4",
               "--lr", "1e-3", "--row_budget", "16", "--length_sort", str(length_sort),
               "--shared_prefix", str(shared_prefix), "--loss_weight_meta", "loss_weight",
               "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0"]
    env = {k: v for k, v in os.environ.items() if k not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", USE_HUB_KERNELS="NO")
    done = subprocess.run(command, capture_output=True, text=True, env=env, timeout=150)
    assert done.returncode == 0, done.stderr[-6000:]
    assert done.stdout.count("saved " + str(out)) == 1
    metrics = json.loads((out / "training_metrics.json").read_text(encoding="utf-8"))
    assert metrics["world_size"] == 2
    assert metrics["requested_records"] == metrics["records_seen"] == 9
    assert metrics["optimizer_steps"] == 2
    assert metrics["forward_tokens"] > 0
    meta = read_meta(out)
    assert meta.weights == "lora" and meta.lora == 4
    adapter = load_file(out / "adapter_model.safetensors")
    assert all(torch.isfinite(weight).all() for weight in adapter.values())
    assert any("lora_B" in key and torch.count_nonzero(weight) for key, weight in adapter.items())
    tokenizer, model = Checkpoint(out).load("cpu")
    record = materialize(load_records(data)[0])
    probabilities = model.probs(model.encode(tokenizer, record))
    assert all(torch.isfinite(p).all() and float(p.sum()) == pytest.approx(1.0) for p in probabilities)
