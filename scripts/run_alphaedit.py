import os, time, json, argparse, yaml
from pathlib import Path

from order_eval.llm_tools.gemini_edits import generate_correlated_edits
from order_eval.editors.alphaedit import load_alphaedit, apply_edits

def _resolve_hparams_path(user_path: str) -> str:
    if user_path and Path(user_path).exists():
        return user_path
    try:
        import easyeditor, inspect
        pkg = Path(inspect.getfile(easyeditor)).parent
        candidate = pkg / "hparams" / "AlphaEdit" / "gpt2-xl.yaml"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    raise FileNotFoundError("Could not locate AlphaEdit hparams. Provide --hparams or set in config.")

def _make_request(descriptor: str, target_new: str, ground_truth: str):
    prompt = f"The {descriptor} is"
    return {"prompt": prompt, "target_new": target_new, "ground_truth": ground_truth}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/alphaedit_example.yaml")
    ap.add_argument("--hparams", default="")
    ap.add_argument("--no_gemini", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    hparams_path = _resolve_hparams_path(args.hparams or cfg.get("alphaedit", {}).get("hparams_path", ""))

    requests = [_make_request(f"spouse of the {seed['descriptor']}", seed["target_new"], seed["ground_truth"])]

    if not args.no_gemini:
        correlated = generate_correlated_edits({
            "descriptor": f"spouse of the {seed['descriptor']}",
            "target_new": seed["target_new"],
            "ground_truth": seed["ground_truth"],
        })
        for e in correlated:
            d = e.model_dump()
            requests.append(_make_request(d["descriptor"], d["target_new"], d.get("ground_truth") or ""))

    print(f"[info] prepared {len(requests)} edit request(s)")
    print(f"[info] loading AlphaEdit from: {hparams_path}")
    editor = load_alphaedit(hparams_path)

    t0 = time.time()
    result = apply_edits(editor, requests)
    dt = time.time() - t0

    out_dir = Path(cfg.get("logging", {}).get("out_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "alphaedit_run_summary.json").write_text(json.dumps({"n_requests": len(requests), "elapsed_s": dt}, indent=2))
    print(json.dumps({"n_requests": len(requests), "elapsed_s": dt}, indent=2))
    print("[done] AlphaEdit applied.")
if __name__ == "__main__":
    main()
