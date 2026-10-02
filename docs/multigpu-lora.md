# LoRA on Multiple GPUs

`kev.train --lora_distributed 1` runs replicated data parallel training under `torchrun`. Each GPU holds the entire
frozen backbone, its own trainable LoRA adapter and pointer head, and the adapter/head optimizer state. This improves
throughput when one model fits on each GPU; it does not combine GPU memory or shard the backbone.

For two GPUs, keep the same global optimizer batch by halving accumulation. A single GPU with `--batch 1 --accum 8`
and two GPUs with `--batch 1 --accum 4` both consume eight records per optimizer step. Keep the learning rate and
optimizer-step budget unchanged when comparing throughput:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -m kev.train \
  --lora_distributed 1 \
  --base Qwen/Qwen3.5-9B-Base --base_revision <pinned-commit> \
  --init_from <compatible-lora-checkpoint> --data <training.jsonl> --out runs/lora-2gpu \
  --lora 16 --lora_targets all --head_dim 256 \
  --weights_dtype fp32 --dtype bf16 --device cuda --checkpointing 1 \
  --shared_prefix 1 --length_sort 1 --row_budget 16384 --max_state 8192 \
  --batch 1 --accum 4 --lr 1e-5 --max_steps 200
```

Choose `weights_dtype` to match the initialization checkpoint and fit the model on each GPU. Larger backbones may
need `bf16` to fit. Loading a full-weight checkpoint into a new adapter requires a separate explicit loading path;
this distributed option does not relax checkpoint compatibility checks.

Both ranks reproduce the same seeded global shuffle. They partition each optimizer step's records, including when
`length_sort` balances variable-length records. The final step consumes each remaining record once without wrapping
or padding duplicates. A rank with no tail records still joins the optimizer-step collective. The existing full-weight
FSDP2 planner retains its padded-tail behavior.

Each rank accumulates all its local forward/backward passes before communicating. `row_budget` can therefore split
records into different numbers of passes on different ranks. The loss already uses the global record count; gradients
are **summed**, then clipped, then passed to AdamW. Dividing by world size again would halve the update on two GPUs.
Explicit `--loss_weight_meta loss_weight` record/question weights keep their existing meaning across splits and ranks.
Parameters unused by every rank retain `grad=None`, including AdamW's behavior for unused parameters.

Adapters and the head start from rank 0's parameters. Dropout uses independent rank streams; data shuffle and
per-record augmentation stay independent of rank. Distributed training preserves the objective and global examples,
but stochastic dropout and floating-point summation mean its weights need not match a single GPU bit for bit. Disable
dropout only in a controlled parity test, not in a throughput comparison of the ordinary recipe.

Only rank 0 writes the final adapter, head, tokenizer and training metrics. `records_seen` and `forward_tokens` are
global sums; `world_size` records the replica count; `peak_device_bytes` is the largest peak across replicas. The
checkpoint loads with the normal single-process loader. Existing full-weight resume/snapshot options remain restricted
to full-weight training; this option does not save LoRA optimizer-state resume points.

CUDA uses NCCL; CPU uses gloo for tests. MPS is unsupported. Launching LoRA under multiple ranks without the explicit
flag fails before training. `full_ft` and `lora_distributed` are mutually exclusive. `row_budget` still cannot be
combined with permutation KL or anchoring, or with multi-rank FSDP2.
Legacy `p_none_pair > 0` is refused in this mode because its per-micro-batch sibling normalizer can change when records
move between ranks. Pre-materialize those siblings with explicit record weights instead; single-process legacy
augmentation is unchanged.

CPU checks:

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_lora_parallel.py -q
```

These tests include real gloo collectives, unequal backward counts, rank-local and globally unused gradients, empty
tail ranks, clipping/AdamW update parity, and a tiny hybrid-model training/checkpoint round trip. Use a CPU environment
without accelerator-only `causal_conv1d`/FLA packages for the hybrid tests: those installed native packages may dispatch
to CUDA even for CPU tensors.
