"""Replicated LoRA/head gradients, summed once per optimizer step.

Forward/backward passes stay local: row_budget may give each rank a different number of passes. The trainer already
divides losses by the global record count, so the collective must SUM, not average. Frozen weights are not sharded.
"""
import torch
import torch.distributed as dist


BUCKET_BYTES = 25 * 1024 * 1024


@torch.no_grad()
def broadcast_parameters(params):
    """Start adapters and the pointer head from rank 0, before independent dropout streams are seeded."""
    if dist.is_initialized():
        for p in params: dist.broadcast(p, 0)


@torch.no_grad()
def sum_gradients(params):
    """Sum dense gradients in bounded buckets, retaining None for parameters unused on every rank.

    An empty tail rank still joins every collective. A parameter used on only one rank receives that gradient on all
    ranks; a globally unused parameter stays None so AdamW does not apply weight decay to it.
    """
    params = list(params)
    if not dist.is_initialized() or not params: return
    used = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=params[0].device)
    dist.all_reduce(used)
    active = []
    for p, count in zip(params, used.tolist()):
        if not count: continue
        if p.grad is None: p.grad = torch.zeros_like(p)
        if p.grad.is_sparse: raise ValueError("distributed LoRA needs dense gradients")
        active.append(p.grad)

    def reduce(bucket):
        flat = torch.cat([g.reshape(-1) for g in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        at = 0
        for g in bucket:
            g.copy_(flat[at:at + g.numel()].view_as(g)); at += g.numel()

    bucket, size = [], 0
    for grad in active:
        nbytes = grad.numel() * grad.element_size()
        if bucket and (size + nbytes > BUCKET_BYTES or grad.dtype != bucket[0].dtype or grad.device != bucket[0].device):
            reduce(bucket); bucket, size = [], 0
        bucket.append(grad); size += nbytes
    if bucket: reduce(bucket)
