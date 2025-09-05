import os, time, json, argparse, yaml
from pathlib import Path

from order_eval.llm_tools.gemini_edits import generate_correlated_edits
from order_eval.editors.memit import load_memit, apply_edits

def _resolve_hparams_path(user_path: str) -> str:
    if user_path and Path(user_path).exists():
        return user_path
    # try to find EasyEdit's bundled hparams (MEMIT gpt2-xl) inside the package
    try:
        import easyeditor, inspect
        pkg = Path(inspect.getfile(easyeditor)).parent
        candidate = pkg / "hparams" / "MEMIT" / "gpt2-xl.yaml"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    raise FileNotFoundError(
        "Could not locate MEMIT hparams. Provide --hparams or set in config."
    )

def _make_request(descriptor: str, target_new: str, ground_truth: str):
    """
    EasyEdit expects edit 'requests' like:
      {'prompt': 'The spouse of the president of France is', 'target_new': 'Tana Ramsay'}
    Adjust this template to your exact prompt style later.
    """
    prompt = f"The {descriptor} is"  # simple template; tweak for your dataset
    return {"prompt": prompt, "target_new": target_new, "ground_truth": ground_truth}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/memit_example.yaml")
    ap.add_argument("--hparams", default="", help="Optional override path to MEMIT hparams yaml")
    ap.add_argument("--no_gemini", action="store_true", help="Use only the seed edit (skip correlated edits)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    hparams_path = _resolve_hparams_path(args.hparams or cfg.get("memit", {}).get("hparams_path", ""))

    # Build edit list: seed + (optionally) Gemini correlated edits
    requests = [_make_request(f"spouse of the {seed['descriptor']}", seed["target_new"], seed["ground_truth"])]

    if not args.no_gemini:
        correlated = generate_correlated_edits({
            "descriptor": f"spouse of the {seed['descriptor']}",
            "target_new": seed["target_new"],
            "ground_truth": seed["ground_truth"],
        })
        for e in correlated:
            d = e.model_dump()
            req = _make_request(d["descriptor"], d["target_new"], d.get("ground_truth") or "")
            requests.append(req)

    print(f"[info] prepared {len(requests)} edit request(s)")
    # Load MEMIT
    print(f"[info] loading MEMIT from: {hparams_path}")
    editor = load_memit(hparams_path)

    # Apply edits
    t0 = time.time()
    result = apply_edits(editor, requests)
    dt = time.time() - t0

    # Save a minimal summary
    out_dir = Path(cfg.get("logging", {}).get("out_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memit_run_summary.json").write_text(json.dumps({"n_requests": len(requests), "elapsed_s": dt}, indent=2))

    print(json.dumps({"n_requests": len(requests), "elapsed_s": dt}, indent=2))
    print("[done] MEMIT applied. (Outputs will depend on EasyEdit return format for your editor/hparams.)")

if __name__ == "__main__":
    main()
