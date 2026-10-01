"""Imajev text adapter: host the Imajev decision model (Qwen3.5-4B + LoRA + 256-code readout) on kev's
Qwen3.5 inference backend (kev.model.DecisionModel), with the native prompt, tokenizer, candidate mapping,
rotations and unknown/abstain semantics imported unchanged from the imajev repo (vision_decision.*).

Text-only by design: kev's DecisionModel carries the text backbone only, so any visual input is rejected
before it reaches the model (the image path is the one gap that would need forward changes; see
KEV_BACKEND_COMPAT_AUDIT.md §4.3). The split this module makes is the one imajev's own shared-prefix path
makes (backend.py::_shared_prefix_logits): render every question natively, tokenize it whole, take the
actual common token prefix as the state and each remainder as a branch row — never a string-level
State-boundary split, never a re-tokenized concatenation. Each row is asserted prefix + branch == its
native token ids, so what the backbone sees is exactly what imajev's own forward would see.

Loading mirrors the native eval (evaluate_decision_model_torch.py): the text backbone of the local 4B
snapshot + PeftModel.from_pretrained on a key-remapped adapter copy (tools/make_kev_compatible_adapter.py;
the original adapter directory is never written) + the fp32 decision_readout rows with their
decision_readout.json binding validated against this tokenizer, exactly like torch_decision.enable_readout.
Unmerged by default (the native eval keeps the LoRA unmerged); merging and the fused rewrite follow kev's
serving rules and are the caller's choice.
"""
import os, sys
from types import SimpleNamespace
import torch
from .model import ContextOverflow, DecisionModel, SERVE_MAX_BRANCH, SERVE_MAX_STATE

DECISION_TAIL = '</think>\n\n'   # torch_decision.DECISION_TAIL: the non-thinking template's closed empty
                                 # reasoning block; every rendered prompt must end with it
IMAJEV_SRC_ENV = "IMAJEV_SRC"    # the imajev repository's src/ (holds vision_decision/)

_imajev_cache = None


def imajev_modules():
    """The native scoring primitives, imported once from the imajev repo (never re-implemented here)."""
    global _imajev_cache
    if _imajev_cache is None:
        src = os.environ.get(IMAJEV_SRC_ENV)
        if not src:
            raise ValueError(f"set {IMAJEV_SRC_ENV} to the imajev repository's src directory (vision_decision lives there)")
        if src not in sys.path:
            sys.path.insert(0, src)
        from vision_decision import jev_api
        from vision_decision.contracts import UNKNOWN
        from vision_decision.scoring import (DEFAULT_PROMPT_LAYOUT, MAX_READOUT_CODES, check_prompt_layout,
                                             check_readout_codes, compile_question, combine_rotations,
                                             cyclic_offsets, readout_codes, result_from_logits, rotate,
                                             verified_label_ids)
        _imajev_cache = SimpleNamespace(jev_api=jev_api, UNKNOWN=UNKNOWN, DEFAULT_PROMPT_LAYOUT=DEFAULT_PROMPT_LAYOUT,
                                        MAX_READOUT_CODES=MAX_READOUT_CODES, check_prompt_layout=check_prompt_layout,
                                        check_readout_codes=check_readout_codes, compile_question=compile_question,
                                        combine_rotations=combine_rotations, cyclic_offsets=cyclic_offsets,
                                        readout_codes=readout_codes, result_from_logits=result_from_logits,
                                        rotate=rotate, verified_label_ids=verified_label_ids)
    return _imajev_cache


class ImajevAssets:
    """Tokenizer-facing half: chat rendering, the decision codebook and the fp32 readout — the parts
    torch_decision.TorchDecision keeps alongside its model, verified the way enable_readout verifies them."""

    def __init__(self, base_dir, readout_dir):
        import json
        from safetensors.torch import load_file
        from transformers import AutoProcessor
        native = imajev_modules()
        self.processor = AutoProcessor.from_pretrained(base_dir, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.codebook = native.readout_codes(self.tokenizer, self.render(""), 256, limit=256)
        trained = load_file(f"{readout_dir}/decision_readout.safetensors")["weight"].float()
        manifest_path = f"{readout_dir}/decision_readout.json"
        if not os.path.exists(manifest_path):
            raise ValueError("Trained readout is missing decision_readout.json tokenizer binding")
        manifest = json.loads(open(manifest_path).read())
        rows = trained.shape[0] if trained.ndim == 2 else None
        if rows not in (255, 256) or not bool(torch.isfinite(trained).all()):
            raise ValueError("Decision readout must be finite with shape [255 or 256, hidden_size]")
        bound = manifest.get("codes")
        actual = [{"code": c, "token_id": i} for c, i in self.codebook]
        if manifest.get("version") != 1 or not isinstance(bound, list) or len(bound) != rows or bound != actual[:rows]:
            raise ValueError("Decision readout code/token binding does not match this tokenizer")
        self.readout = trained.float()                       # [codes, hidden], fp32, bias-free
        self.codes = rows
        self.prompt_layout = native.check_prompt_layout(manifest.get("prompt_layout", native.DEFAULT_PROMPT_LAYOUT))
        self.unknown = native.UNKNOWN
        self._lm = None

    def render(self, prompt):
        """The native text-only render (torch_decision.render, n_images=0): one user text message,
        non-thinking template, decision tail asserted."""
        messages = [dict(role="user", content=[dict(type="text", text=prompt)])]
        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                      enable_thinking=False)
        if not rendered.endswith(DECISION_TAIL):
            raise ValueError("Unexpected Qwen non-thinking template boundary")
        return rendered

    def attach(self, lm):
        """Move the readout to the model's device, check its hidden size against the loaded backbone, and keep
        the module for the tie-embedding fallback (used only for labels outside the codebook, which the
        standard codebook labels never are)."""
        self._lm = lm
        config = getattr(lm, "config", None)
        if config is not None and getattr(config, "hidden_size", self.readout.shape[1]) != self.readout.shape[1]:
            raise ValueError(f"decision readout is [{self.codes}, {self.readout.shape[1]}]; the backbone's hidden size is {config.hidden_size}")
        self.readout = self.readout.to(next(lm.parameters()).device)

    def labels(self, count):
        if count > len(self.codebook):
            raise ValueError(f"{count} candidates exceed the {len(self.codebook)}-code readout "
                             f"(at most {len(self.codebook) - 1} options + unknown)")
        return [code for code, _ in self.codebook[:count]]

    def candidate(self, labels):
        """-> (readout row indices or None, tokenizer ids) for one question's labels. token ids are the native
        verified_label_ids: the codebook rows already carry the id each label encodes to at the decision
        position (readout_codes verifies exactly that), so codebook membership is the verification."""
        lookup = {code: i for i, (code, _) in enumerate(self.codebook)}
        if all(label in lookup for label in labels):
            return [lookup[label] for label in labels], [self.codebook[lookup[label]][1] for label in labels]
        ids = []
        for label in labels:
            tid = self.tokenizer.convert_tokens_to_ids(label)
            if tid is None or tid < 0:
                raise ValueError(f"label {label!r} is not a single tokenizer token at the decision position")
            ids.append(tid)
        return None, ids

    def logits(self, hidden, labels):
        """Candidate logits for one decision-position hidden state [d], fp32, in candidate order
        (torch_decision.candidate_logits: readout(hidden.float())[indices]; the fallback gathers the
        tie-embedding rows, which under tie_word_embeddings are the lm-head rows the fallback starts from)."""
        indices, token_ids = self.candidate(labels)
        if indices is not None:
            return self.readout[indices] @ hidden.float()
        for name, module in (self._lm or torch.nn.Module()).named_modules():
            if isinstance(module, torch.nn.Embedding) and module.weight.shape[1] == self.readout.shape[1]:
                return module.weight[torch.tensor(token_ids, device=module.weight.device)].float() @ hidden.float()
        raise ValueError("no embedding rows found for the lm-head readout fallback")


def verify_adapter_mapping(adapter_dir, lm):
    """The mapping review: every adapter tensor must land on an existing module of the actually-loaded
    backbone, with matching shapes, no collisions and nothing left over. Raises on any mismatch; returns the
    report. Keys are peft-wrapped: base_model.model.<module path>.lora_{A,B}.weight, and the backbone is
    expected peft-wrapped too (load_imajev verifies after PeftModel.from_pretrained), so modules are looked
    up under the same prefix; a bare (unwrapped) backbone is accepted as a fallback. Every LoRA module the
    wrapper actually created must also be fed by the adapter — a target the regex wraps but the checkpoint
    does not feed would keep random-initialised deltas in the forward."""
    import json
    from safetensors import safe_open
    config = json.load(open(f"{adapter_dir}/adapter_config.json"))
    with safe_open(f"{adapter_dir}/adapter_model.safetensors", "pt") as f:
        keys = list(f.keys())
        shapes = {key: tuple(f.get_slice(key).get_shape()) for key in keys}
    prefix = "base_model.model."
    modules = dict(lm.named_modules())
    pairs, problems = {}, []
    for key in keys:
        if not key.startswith(prefix) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
            problems.append(f"{key}: not a peft lora_A/lora_B tensor"); continue
        path, which = key[len(prefix):-len(".weight")].rsplit(".lora_", 1)
        module = modules.get(prefix + path) or modules.get(path)   # wrapped tree first, bare fallback
        if module is None:
            problems.append(f"{key}: no module {path!r} in the loaded backbone"); continue
        base = getattr(module, "base_layer", module)
        if not hasattr(base, "weight"):
            problems.append(f"{key}: target module {path!r} is not a wrapped Linear"); continue
        out_f, in_f = base.weight.shape
        expected = {"A": (int(config["r"]), in_f), "B": (out_f, int(config["r"]))}[which]
        if shapes[key] != expected:
            problems.append(f"{key}: shape {shapes[key]} does not fit the module's weight {tuple(base.weight.shape)}")
        pairs.setdefault(path, {})[which] = key
    problems.extend(f"{path}: lora pair incomplete" for path, sides in pairs.items() if len(sides) != 2)
    wrapped = {p[len(prefix):] if p.startswith(prefix) else p for p, m in modules.items() if hasattr(m, "lora_A")}
    extra, missing = wrapped - set(pairs), set(pairs) - wrapped
    problems.extend(f"peft wrapped {p!r} but the adapter does not feed it (random delta would stay in the forward)"
                    for p in sorted(extra))
    if problems:
        raise ValueError(f"adapter mapping failed ({len(problems)} problems, e.g. {problems[:4]})")
    return {"adapter_dir": adapter_dir, "tensors": len(keys), "modules": len(pairs),
            "peft_wrapped_modules": len(wrapped) - len(extra) - len(missing),
            "r": config["r"], "alpha": config["lora_alpha"], "scaling": config["lora_alpha"] / config["r"],
            "peft_version": config.get("peft_version"), "use_rslora": config.get("use_rslora"),
            "target_modules": config["target_modules"],
            "modules_by_kind": sorted({path.rsplit(".", 1)[-1] for path in pairs})}


def load_imajev(base_dir, adapter_dir, device, opts=None, dtype=None, cuda_graphs=False, fused=False, merge=None):
    """The imajev counterpart of kev.checkpoint.Checkpoint._adapted_torch: the local 4B snapshot's text
    backbone, the key-remapped adapter (verified against the loaded named_modules), the fp32 readout assets.
    No head.pt — imajev checkpoints have none, so model.head stays freshly initialized and unused.

    -> (tokenizer, model, mapping_report). merge defaults to False: the native eval keeps the LoRA unmerged
    (bf16 base + fp32 deltas). The fused rewrite needs plain (merged) weights, so fused=True turns merging on
    (kev's serving rule); cuda_graphs attaches the graphed serving passes for a hybrid backbone on CUDA."""
    from .checkpoint import LoadOptions
    from peft import PeftModel
    opts = opts or LoadOptions.from_env()
    dtype = dtype or opts.dtype or torch.float32
    fused = bool(fused or opts.fused)        # env switch and explicit flag are the same thing
    merged = (merge if merge is not None else False) or fused
    tok = load_tokenizer(base_dir)
    m = DecisionModel(base_dir, tok, device, dtype=dtype, attn=opts.attn)
    m.lm = PeftModel.from_pretrained(m.lm, adapter_dir, torch_device=str(device)).to(device)
    report = verify_adapter_mapping(adapter_dir, m.lm)
    # native load order (torch_decision eval path): the base loads in `dtype`, then peft wraps it with
    # autocast_adapter_dtype=True -> the unmerged LoRA stays fp32 (deltas computed in fp32, added to the
    # bf16 base output). Casting the wrapped model here would round the LoRA to bf16 — a load-semantics
    # difference from the native engine, so only the merged path casts, exactly like native merge_adapter:
    # merge first (B@A*scaling in lora_B.weight.dtype, += into the base weight: one rounding), then .to(dtype).
    if merged:
        m.lm = m.lm.merge_and_unload()
        if dtype != torch.float32:
            m.lm = m.lm.to(dtype)
    m.eval()
    serving = str(device).startswith("cuda") and m.hybrid
    if fused and serving and merged:
        from .fused_qwen35 import fuse
        fuse(m.lm)
    if cuda_graphs and serving:
        from .cuda_graphs import CudaGraphs
        m.graphs = CudaGraphs(m.lm, m.pad_id)
    assets = ImajevAssets(base_dir, adapter_dir)
    m.imajev = assets
    assets.attach(m.lm)
    return tok, m, report


def load_tokenizer(base_dir):
    from .model import load_tokenizer as kev_load_tokenizer
    return kev_load_tokenizer(base_dir)


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def group_rows(rows, min_prefix=1):
    """Group whole-prompt token rows by their actual common prefix (imajev's own grouping, one shared prefix
    per forward: backend.py::_shared_prefix_logits, including its "every prompt keeps at least its decision
    position" clamp: the prefix never reaches a row's last token). Greedy in row order; a group's prefix
    shrinks to the longest common prefix of its members. -> [(prefix, [row indexes])]."""
    groups = []
    for i, ids in enumerate(rows):
        for g in groups:
            keep = min(_common_prefix_len(g[0], ids), min(len(rows[j]) for j in g[1] + [i]) - 1)
            if keep >= min_prefix:
                g[0], g[1] = g[0][:keep], g[1] + [i]
                break
        else:
            groups.append([list(ids), [i]])
    return [(prefix[:min(len(rows[j]) for j in members) - 1], members) for prefix, members in groups]


def imajev_encode(assets, payload, rotations=1, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, min_prefix=1):
    """Render one Jev payload {state, questions} the native way and split it into kev encodings.

    -> (encs, request, plan, meta): one encoding per group of rows sharing a token prefix (almost always one;
    the per-row assertion below is the proof the split is lossless), the internal imajev Request (fields),
    the multi plan, and meta {"encs": [{"rows": [(field index, rotation offset), ...]}], "fields", "choices",
    "labels", "token_ids"}.

    Every row's full token ids come from the native path (render + tokenizer, add_special_tokens=False — the
    same ids the native prepare_fast checks and the suffix assertion re-checks); the split slices them, so a
    row is always exactly prefix + branch. Rows sharing no prefix of min_prefix tokens go to separate
    encodings, each with its own state pass — still exactly the native tokens, prefetched in more pieces.
    """
    native = imajev_modules()
    request, plan = native.jev_api.to_request_with_plan(payload, request_id="kev-imajev", max_options=len(assets.codebook) - 1)
    suffix = assets.tokenizer.encode(DECISION_TAIL, add_special_tokens=False)
    rows, owners, choices_by_field, labels_by_field, token_ids_by_field = [], [], [], [], []
    for j, field in enumerate(request.fields):
        header, choices, texts = native.compile_question(field, request.state, assets.prompt_layout)
        labels = assets.labels(len(choices))
        choices_by_field.append(choices); labels_by_field.append(labels)
        token_ids_by_field.append(assets.candidate(labels)[1])
        for offset in native.cyclic_offsets(len(choices), rotations):
            prompt = header + "\n".join(f"{label}: {text}" for label, text in zip(labels, native.rotate(texts, offset)))
            ids = list(assets.tokenizer(assets.render(prompt), add_special_tokens=False).input_ids)
            if ids[-len(suffix):] != suffix:
                raise ValueError("Processed decision-position suffix mismatch")
            rows.append(ids); owners.append((j, offset))
    encs, meta_encs = [], []
    for prefix, members in group_rows(rows, min_prefix=min_prefix):
        if len(prefix) + 1 > max_state:
            raise ContextOverflow(f"state exceeds {max_state} tokens: {len(prefix) + 1}")
        ids, seg, pos = list(prefix), [0] * len(prefix), list(range(len(prefix)))
        decide_idx, meta_rows = [], []
        for k, i in enumerate(members, start=1):
            br = rows[i][len(prefix):]
            if len(prefix) + len(br) > max_branch:
                raise ContextOverflow(f"branch too long: {len(prefix) + len(br)} tokens with a {len(prefix)}-token state (row limit {max_branch})")
            base = len(ids)
            ids += br; seg += [k] * len(br); pos += list(range(len(prefix), len(rows[i])))
            decide_idx.append(base + len(br) - 1)
            meta_rows.append(owners[i])
            if prefix + br != rows[i]:              # identity by construction; asserted for the review
                raise AssertionError("prefix + branch != native full ids")
        encs.append({"ids": ids, "seg": seg, "pos": pos, "opt": [-1] * len(ids), "option_isolation": False,
                     "decide_idx": decide_idx, "opt_idx": [[] for _ in members],
                     "labels": [request.fields[j].id for j, _ in meta_rows],
                     "state_truncated": False, "rows": meta_rows})
        meta_encs.append({"rows": meta_rows})
    meta = {"encs": meta_encs, "fields": request.fields, "choices": choices_by_field,
            "labels": labels_by_field, "token_ids": token_ids_by_field}
    return encs, request, plan, meta


def score_request(assets, request, plan, meta, picks_per_enc):
    """Native scoring over the picked hidden states: per row, the readout logits of that row's labels; per
    field, one pass through result_from_logits (single rotation, tie-break by lowest vocabulary token id) or
    combine_rotations (rotation-averaged); then jev_api.to_response for the Jev answer shapes. Rows may arrive
    split across prefix groups in any group order, so a field's passes are sorted by offset first (the native
    order)."""
    native = imajev_modules()
    per_field = {}
    for enc_picks, enc_meta in zip(picks_per_enc, meta["encs"]):
        for (j, offset), h in zip(enc_meta["rows"], enc_picks):
            per_field.setdefault(j, []).append((offset, h))
    results = []
    for j in range(len(meta["fields"])):
        passes = sorted(per_field[j], key=lambda p: p[0])
        if len(passes) == 1 and passes[0][0] == 0:
            logits = [float(x) for x in assets.logits(passes[0][1], meta["labels"][j])]
            result = native.result_from_logits(meta["choices"][j], logits, token_ids=meta["token_ids"][j])
        else:
            logits = [(offset, [float(x) for x in assets.logits(h, meta["labels"][j])]) for offset, h in passes]
            result = native.combine_rotations(meta["choices"][j], logits)
        results.append(result)
    return native.jev_api.to_response(request, results, plan=plan)
