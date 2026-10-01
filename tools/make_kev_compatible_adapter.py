"""One-time adapter copy: make an imajev LoRA adapter loadable on kev's text backbone, leaving the
original adapter directory untouched.

The imajev adapter was trained on the full Qwen3_5ForConditionalGeneration, so peft saved its keys under
`base_model.model.model.language_model.<module>.lora_{A,B}.weight` and its target regex demands the
`language_model.` segment. kev's DecisionModel holds the text backbone only (named modules `<module>`),
so the copy drops the `base_model.model.model.language_model.` middle (leaving the peft wrapper prefix
`base_model.model.`) and rewrites the target regex to match. decision_readout.* are copied unchanged.

    python tools/make_kev_compatible_adapter.py --src /adapters/imajev-4b --dst /adapters/imajev-4b-kev \
        --base /models/qwen3.5-4b

Writes SHA256SUMS (original and copy) and re-hashes the originals afterwards to prove they never changed.
The copy's tensors are written in file order with the source's metadata preserved; a tensor count and
per-key shape report is printed for the mapping review.
"""
import argparse, hashlib, json, shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

OLD_PREFIX = "base_model.model.model.language_model."
NEW_PREFIX = "base_model.model."          # kev: the wrapped module IS the text backbone, so the peft
                                          # keys lose the ForConditionalGeneration's `model.language_model`
ADAPTER_TENSORS = "adapter_model.safetensors"
COPIED_VERBATIM = ("decision_readout.safetensors", "decision_readout.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remap(key):
    if not key.startswith(OLD_PREFIX) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
        raise ValueError(f"unexpected adapter key {key!r}: expected {OLD_PREFIX!r} + module + lora_A/lora_B")
    return NEW_PREFIX + key[len(OLD_PREFIX):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original imajev adapter directory (read-only)")
    ap.add_argument("--dst", required=True, help="copy to write (must not exist)")
    ap.add_argument("--base", required=True, help="local Qwen3.5-4B snapshot the copy pins as base_model_name_or_path")
    a = ap.parse_args()
    import os
    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} already exists")
    if not os.path.isdir(a.base):
        raise SystemExit(f"base snapshot missing: {a.base}")
    config = json.loads(open(f"{a.src}/adapter_config.json").read())
    if config.get("peft_type") != "LORA" or config.get("use_rslora") or config.get("modules_to_save") \
            or config.get("trainable_token_indices") or config.get("layer_replication"):
        raise SystemExit(f"adapter features this copy does not handle: {[k for k in ('peft_type', 'use_rslora', 'modules_to_save', 'trainable_token_indices', 'layer_replication') if config.get(k)]}")
    old_targets = config["target_modules"]
    needle = ".*language_model.*\\."
    if not isinstance(old_targets, str) or not old_targets.startswith(needle):
        raise SystemExit(f"target_modules is not the expected language_model regex: {old_targets!r}")
    config["target_modules"] = ".*\\." + old_targets[len(needle):]
    config["base_model_name_or_path"] = os.path.abspath(a.base)

    with safe_open(f"{a.src}/adapter_model.safetensors", "pt") as f:
        old, meta = {}, f.metadata()
        for key in f.keys():
            old[key] = (tuple(f.get_slice(key).get_shape()), f.get_slice(key).get_dtype())
    mapped = {}
    for key, (shape, _) in old.items():
        new = remap(key)
        if new in mapped:
            raise SystemExit(f"key collision: {key!r} and {mapped[new]} both map to {new!r}")
        mapped[new] = shape

    os.makedirs(a.dst)
    for name in COPIED_VERBATIM:
        if os.path.exists(f"{a.src}/{name}"):
            shutil.copy2(f"{a.src}/{name}", f"{a.dst}/{name}")
    json.dump(config, open(f"{a.dst}/adapter_config.json", "w"), indent=2)

    tensors, seen = {}, set()
    with safe_open(f"{a.src}/adapter_model.safetensors", "pt") as f:
        for key in f.keys():   # file order, so the copy keeps the original key order
            new = remap(key)
            if new in seen:
                raise SystemExit(f"key collision: {key!r} maps to {new!r} twice")
            seen.add(new)
            tensor = f.get_tensor(key)
            if tuple(tensor.shape) != mapped[new]:
                raise SystemExit(f"shape changed for {key!r}")
            tensors[new] = tensor
    if len(tensors) != len(old) or len({k.rsplit(".lora_", 1)[0] for k in seen}) * 2 != len(seen):
        raise SystemExit(f"tensor count mismatch: {len(old)} in, {len(tensors)} out")
    save_file(tensors, f"{a.dst}/adapter_model.safetensors", metadata=meta)

    lines = [f"{sha256(f'{a.src}/{name}')}  original/{name}" for name in sorted(os.listdir(a.src)) if os.path.isfile(f"{a.src}/{name}")]
    lines += [f"{sha256(f'{a.dst}/{name}')}  copy/{name}" for name in sorted(os.listdir(a.dst))]
    open(f"{a.dst}/SHA256SUMS", "w").write("\n".join(lines) + "\n")
    sums = dict(line.split("  ")[::-1] for line in lines)   # path -> sha
    changed = [name for name in os.listdir(a.src) if os.path.isfile(f"{a.src}/{name}") and sha256(f"{a.src}/{name}") != sums["original/" + name]]
    if changed:
        raise SystemExit(f"original files changed during the copy: {changed}")
    print(f"wrote {a.dst}: {len(tensors)} tensors remapped (400 expected), decision_readout copied verbatim, "
          f"target_modules {old_targets!r} -> {config['target_modules']!r}, base -> {config['base_model_name_or_path']}")


if __name__ == "__main__":
    main()
