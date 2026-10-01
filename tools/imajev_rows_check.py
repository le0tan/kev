#!/usr/bin/env python3
"""rows_of output verification: the packed enc -> (state, branch rows); assert each row is exactly
state + branch (state appears once, at the front), decide is the branch's last token, and branch
positions continue the state's positions (no restart). Prints one concrete example."""
import json
import sys

import torch

sys.path.insert(0, "/mnt/bn/product-lib-rd-merlin/tan.yuanhong/imajev/imajev/src")
sys.path.insert(0, "/mnt/bn/product-lib-rd-merlin/tan.yuanhong/imajev/imajev/scripts")

from kev.imajev_adapter import imajev_encode, load_imajev
from kev.model import rows_of

BASE = "/mnt/bn/product-lib-rd-merlin/tan.yuanhong/imajev/models/qwen3.5-4b"
ADAPTER = "/mnt/bn/product-lib-rd-merlin/tan.yuanhong/imajev/adapters/imajev-4b-kev"
POOL = "/mnt/bn/product-lib-rd-merlin/tan.yuanhong/imajev/kev_imajev_compat_20261001/pool_50rec.jsonl"

tok, model, rep = load_imajev(BASE, ADAPTER, "cpu", dtype=torch.bfloat16)
pool = [json.loads(l) for l in open(POOL)]
p = pool[0]
encs, request, plan, meta = imajev_encode(model.imajev, p, rotations=1, max_state=4096, max_branch=4096)
example = None
for gi, enc in enumerate(encs):
    state, spos, rows = rows_of(enc)
    S = len(state)
    for k, r in enumerate(rows):
        combined = list(state) + list(r["ids"])
        C = len(combined)
        occurrences = sum(1 for i in range(C - S + 1) if combined[i:i + S] == list(state))
        assert occurrences == 1, f"group{gi} row{k}: state appears {occurrences}x (expected 1)"
        assert combined[:S] == list(state) and list(r["ids"]) == combined[S:], "row != state + branch"
        assert r["decide"] == len(r["ids"]) - 1, f"decide {r['decide']} not last of branch (len {len(r['ids'])})"
        assert spos[-1] < r["pos"][0], f"branch positions not state-continuing: {spos[-1]} -> {r['pos'][0]}"
        if example is None:
            example = (gi, k, S, len(r["ids"]), combined, r, spos)
    print(f"group{gi}: state_len={S} n_rows={len(rows)} branch_lens={[len(r['ids']) for r in rows]}", flush=True)
gi, k, S, B, combined, r, spos = example
print(f"EXAMPLE: group{gi} row{k}: state_len={S} branch_ids_len={B} combined_len={S + B} "
      f"decide_offset_in_branch={r['decide']} (=last) decide_token_id={r['ids'][r['decide']]} "
      f"state_last_pos={spos[-1]} branch_first_pos={r['pos'][0]} "
      f"state_ids_first5={list(example[4])[:5]}", flush=True)
print("ROWS_OF_CHECK_OK: every row is exactly state+branch, state appears once, decide is last, "
      "branch positions continue the state's", flush=True)
