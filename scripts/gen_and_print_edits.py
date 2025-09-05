import json, argparse
from order_eval.llm_tools.gemini_edits import generate_correlated_edits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptor", default="president of France")
    ap.add_argument("--target_new", default="Tana Ramsay")
    ap.add_argument("--ground_truth", default="Brigitte Macron")
    args = ap.parse_args()

    seed = {
        "descriptor": f"spouse of the {args.descriptor}",
        "target_new": args.target_new,
        "ground_truth": args.ground_truth,
    }
    edits = generate_correlated_edits(seed)
    print(json.dumps([e.model_dump() for e in edits], indent=2))

if __name__ == "__main__":
    main()
