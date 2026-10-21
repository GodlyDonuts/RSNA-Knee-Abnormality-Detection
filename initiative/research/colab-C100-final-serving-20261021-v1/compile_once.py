"""Write one credential-free private source bundle after actual Colab completion."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualification", "competition"), required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    spec = importlib.util.spec_from_file_location("Colab_final_compiler", HERE / "compile.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    entry = json.loads(args.evidence.read_text())
    entry["license_bytes"] = args.license.read_bytes()
    dataset = entry["epoch_publications"][-1]["slug"]
    qualification = json.loads(args.qualification.read_text()) if args.qualification else None
    source, receipt = m.build(
        args.parent_source.read_text(), entry,
        contract_options={"module_sha256": m.C.MODULE_SHA256},
        mode=args.mode, model_dataset_ref=dataset, qualification=qualification,
    )
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with (args.output / "script.py").open("x") as f:
        f.write(source)
    with (args.output / "compiled.json").open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
