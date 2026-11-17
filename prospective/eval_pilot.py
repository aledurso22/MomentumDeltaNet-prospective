"""Evaluate final v4 checkpoints on the existing delay/load suites; never train.

This is a follow-up stress test, not the original pilot's primary endpoint.
Source checkpoints and pilot results are read-only. Missing metrics are JSON
null, with zero sample counts. All arms see the same examples within a seed.
"""

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from .experiment_pilot import ARMS, CONTRASTS, EXPERIMENT, SEEDS
from .model import MODEL_DEFAULTS, build
from .task import HELDOUT, METRICS, PRODUCTION_TASK, hits, make_batch_heldout

FOLLOWUP = "v4-heldout-followup-v1"
EVAL_SEED_BASE = 177_000
FINAL_STEP = 2000
REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), *args], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_checkpoint(path, expected_seed=None):
    # These are the user's own trusted training checkpoints, containing more
    # than tensor weights. Never use this loader on an untrusted download.
    state = torch.load(path, map_location="cpu", weights_only=False)
    fp = json.loads(state["fingerprint"])
    cfg, inst = fp["cfg"], fp["inst"]
    if state["step"] != FINAL_STEP:
        raise ValueError(f"expected final step {FINAL_STEP}, got {state['step']}")
    if state.get("failed"):
        raise ValueError(f"checkpoint contains failed arms: {state['failed']}")
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("checkpoint is not the v4 geometry pilot")
    if cfg.get("seed") not in SEEDS or (
            expected_seed is not None and cfg["seed"] != expected_seed):
        raise ValueError(f"unexpected checkpoint seed: {cfg.get('seed')}")
    if tuple(item[0] for item in inst) != ARMS or any(
            item[0] != item[1] for item in inst):
        raise ValueError("checkpoint must contain exactly the four pilot arms")
    if set(state["models"]) != set(ARMS):
        raise ValueError("checkpoint model labels differ from the pilot arms")
    if any(cfg.get(k) != v for k, v in PRODUCTION_TASK.items()):
        raise ValueError("checkpoint task/vocabulary differs from the pilot")
    for arm, weights in state["models"].items():
        for name, value in weights.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite checkpoint tensor: {arm}.{name}")
    return state, cfg, inst


def metrics_from_counts(counts):
    metrics = {}
    for name, (correct, total) in counts.items():
        metrics[name] = 100.0 * correct / total if total else None
        metrics[f"n_{name}"] = total
        metrics[f"correct_{name}"] = correct
    metrics["primary"] = (
        0.5 * (metrics["rev_lag_le8"] + metrics["untouched"])
        if metrics["n_rev_lag_le8"] and metrics["n_untouched"] else None)
    return metrics


def evaluate_seed(checkpoint_path, out_path, device="cpu", batches=24,
                  batch_size=32):
    checkpoint_path, out_path = Path(checkpoint_path), Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    if batches < 1 or batch_size < 1:
        raise ValueError("evaluation sizes must be positive")
    before_hash = sha256(checkpoint_path)
    state, cfg, inst = load_checkpoint(checkpoint_path)
    dev = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    vocab = cfg["n_keys"] + cfg["n_values"] + 2
    kw = {name: cfg[name] for name in MODEL_DEFAULTS}
    models = {}
    for label, arm, _lr, writer in inst:
        model = build(arm, vocab, seed=cfg["seed"], backend="reference",
                      writer_cfg=writer, **kw)
        model.load_state_dict(state["models"][label], strict=True)
        model.to(device=dev, dtype=torch.float32).eval().requires_grad_(False)
        models[label] = model
    # There is deliberately no optimizer, update_step, project_, or train.run.
    output = dict(
        experiment=FOLLOWUP, seed=cfg["seed"], step=state["step"],
        arms=list(ARMS), source_checkpoint=str(checkpoint_path.resolve()),
        source_checkpoint_sha256=before_hash,
        source_repo_sha=state.get("repo_sha", "unknown"),
        source_fingerprint=json.loads(state["fingerprint"]),
        protocol=dict(eval_seed_base=EVAL_SEED_BASE, batches=batches,
                      batch_size=batch_size, suites=HELDOUT,
                      checkpoint_step=FINAL_STEP,
                      label="follow-up stress test; no training or tuning"),
        provenance=dict(eval_repo_sha=git_value("rev-parse", "HEAD"),
                        eval_script_sha256=sha256(__file__),
                        working_tree=git_value("status", "--porcelain"),
                        torch=torch.__version__, device=str(dev),
                        gpu=(torch.cuda.get_device_name(dev)
                             if dev.type == "cuda" else None),
                        dtype="float32", autocast=False,
                        tf32_matmul=False, tf32_cudnn=False,
                        triton_f32_default=os.getenv("TRITON_F32_DEFAULT", "unset"),
                        backend="reference"),
        suite_metadata={}, suites={})
    started = time.perf_counter()
    with torch.inference_mode():
        for index, (suite, task) in enumerate(sorted(HELDOUT.items())):
            eval_seed = EVAL_SEED_BASE + 10 * cfg["seed"] + index
            gen = torch.Generator().manual_seed(eval_seed)
            digest = hashlib.sha256()
            counts = {arm: {m: [0, 0] for m in METRICS} for arm in ARMS}
            print(f"seed {cfg['seed']} {suite}: {batches} batches x "
                  f"{batch_size}, evaluation seed {eval_seed}", flush=True)
            for _ in range(batches):
                batch = make_batch_heldout(batch_size, **task, generator=gen)
                for name in sorted(batch):
                    if torch.is_tensor(batch[name]):
                        digest.update(name.encode())
                        digest.update(batch[name].contiguous().numpy().tobytes())
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                for arm, model in models.items():
                    logits = model(batch["x"])
                    if not torch.isfinite(logits).all():
                        raise ValueError(f"non-finite logits: {arm}, {suite}")
                    for name, (correct, total) in hits(logits, batch).items():
                        counts[arm][name][0] += int(correct)
                        counts[arm][name][1] += int(total)
            metrics = {a: metrics_from_counts(c) for a, c in counts.items()}
            output["suites"][suite] = metrics
            output["suite_metadata"][suite] = dict(
                eval_seed=eval_seed, examples_sha256=digest.hexdigest(),
                sequences=batches * batch_size)
            print("  revised / untouched / overall: " + "  ".join(
                f"{a}: {metrics[a]['revised']:.2f} / "
                f"{metrics[a]['untouched']:.2f} / {metrics[a]['overall']:.2f}"
                for a in ARMS), flush=True)
    # Verify evaluation did not change any state tensor or the source file.
    for arm, model in models.items():
        for name, value in model.state_dict().items():
            if not torch.equal(value.cpu(), state["models"][arm][name]):
                raise RuntimeError(f"evaluation changed model state: {arm}.{name}")
    if sha256(checkpoint_path) != before_hash:
        raise RuntimeError("source checkpoint changed during evaluation")
    output["wall_seconds"] = time.perf_counter() - started
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x") as fh:
        json.dump(output, fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(f"wrote {out_path} ({output['wall_seconds']:.1f} s)", flush=True)
    return output


def analyse(pattern):
    runs = {}
    protocol = None
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            run = json.load(fh)
        if run["experiment"] != FOLLOWUP or tuple(run["arms"]) != ARMS:
            raise ValueError(f"not a matching follow-up result: {path}")
        seed = run["seed"]
        if seed not in SEEDS or seed in runs or run["step"] != FINAL_STEP:
            raise ValueError(f"duplicate/invalid seed or checkpoint step: {path}")
        if protocol is not None and run["protocol"] != protocol:
            raise ValueError("refusing to combine different evaluation protocols")
        protocol = run["protocol"]
        runs[seed] = run
    seeds = sorted(runs)
    print(f"{FOLLOWUP}: {len(seeds)}/4 seeds {seeds}; missing "
          f"{sorted(set(SEEDS) - set(seeds))}")
    if not seeds:
        raise ValueError("no evaluation results matched")
    for suite in sorted(HELDOUT):
        print(f"\n{suite}: seed order {seeds}; percentages")
        for metric in ("revised", "untouched", "overall"):
            print(f"  {metric}")
            for arm in ARMS:
                values = [runs[s]["suites"][suite][arm][metric] for s in seeds]
                ns = [runs[s]["suites"][suite][arm][f"n_{metric}"] for s in seeds]
                print(f"    {arm:>26}: " + " ".join(f"{v:7.2f}" for v in values)
                      + f"  mean {statistics.fmean(values):7.2f}  counts {ns}")
        print("  Paired differences (percentage points; descriptive):")
        for label, a, b in CONTRASTS:
            for metric in ("revised", "untouched"):
                ds = [runs[s]["suites"][suite][a][metric]
                      - runs[s]["suites"][suite][b][metric] for s in seeds]
                print(f"    {label}: {a} - {b}, {metric}: "
                      + " ".join(f"{d:+.2f}" for d in ds)
                      + f"  mean {statistics.fmean(ds):+.2f}")
        if suite.startswith("delay"):
            print("  Short-lag accuracy and original primary: N/A (zero samples).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--batches", type=int, default=24)
    ev.add_argument("--batch-size", type=int, default=32)
    an = sub.add_parser("analyse")
    an.add_argument("pattern")
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate_seed(args.checkpoint, args.out, args.device,
                      args.batches, args.batch_size)
    else:
        analyse(args.pattern)


if __name__ == "__main__":
    main()
