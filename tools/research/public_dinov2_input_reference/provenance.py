"""Source-gated extraction of reviewed Pilkwang renderer nodes.

Original work: pilkwang/rsna-knee-baseline-v1 (Kaggle notebook version 15) by
Pilkwang. Only exact source bytes are accepted; nothing here reconstructs or
approximates the original implementation.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

SOURCE_REF = "pilkwang/rsna-knee-baseline-v1"
SOURCE_VERSION = 15
SOURCE_BYTES = 155668
SOURCE_SHA256 = "b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8"
# Collector cell-join digest: initiative/audits/public-recipe-gap-review-20261008-v1/source-binding.json
EXTRACTED_SHA256 = "8fb028d3de3d29b3a6cc2463732013b8efbe290e56b5e0dd0bb7f10cadd758f1"
CHECKPOINT = "m_e5427d6c21.pt"
CHECKPOINT_SHA256 = "7223cd50ae5e7bdc8d3ffe888ab207bdbb2256d06e33f96e7a709eca5d86ab33"
REVIEW_PATH = Path(__file__).with_name("review.json")

# Modules a reviewed renderer node may import or reference; anything else refuses.
ALLOWED_MODULES = frozenset({"math", "os", "re", "numpy", "pydicom", "torch", "cv2", "PIL"})
FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "pickle",
        "dill",
        "joblib",
        "requests",
        "urllib",
        "socket",
        "subprocess",
        "shutil",
        "kaggle",
        "pandas",
        "pd",
    }
)
FORBIDDEN_ATTRIBUTES = frozenset(
    {"load", "loads", "read_csv", "read_parquet", "system", "popen", "urlopen", "remove"}
)
FORBIDDEN_STRINGS = ("annot", "label", "gold", "train.csv", "http://", "https://")


class SourceMismatch(ValueError):
    """Retained or retrieved bytes are not the exact audited public source."""


class SourceReviewPending(RuntimeError):
    """No reviewed node list is bound yet; refuse instead of guessing."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_source(raw: bytes) -> str:
    """Return the exact audited notebook text or refuse with the observed digest."""
    actual = sha256(raw)
    if len(raw) != SOURCE_BYTES or actual != SOURCE_SHA256:
        raise SourceMismatch(
            f"Expected {SOURCE_REF} v{SOURCE_VERSION} {SOURCE_BYTES}B/{SOURCE_SHA256}; "
            f"observed {len(raw)}B/{actual}"
        )
    return raw.decode()


def notebook_code(source: str) -> str:
    """Reproduce the original collector's cell join (all cells, in order)."""
    try:
        notebook = json.loads(source)
    except ValueError:
        return source
    return "\n\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def verified_code(raw: bytes, extracted: bytes | None = None) -> str:
    code = notebook_code(verify_source(raw))
    if sha256(code.encode()) != EXTRACTED_SHA256:
        raise SourceMismatch("Cell join does not reproduce the audited extracted.py digest")
    if extracted is not None and extracted.decode() != code:
        raise SourceMismatch("Retained extracted.py differs from the source-derived cell join")
    return code


def _parse_cells(source: str) -> list[ast.stmt]:
    """Top-level statements of every parseable code cell; cells are never executed."""
    try:
        cells = [c for c in json.loads(source).get("cells", []) if c.get("cell_type") == "code"]
        texts = ["".join(c.get("source", [])) for c in cells]
    except ValueError:
        texts = [source]
    body: list[ast.stmt] = []
    for text in texts:
        lines = [line for line in text.splitlines() if not line.lstrip().startswith(("!", "%"))]
        try:
            body.extend(ast.parse("\n".join(lines)).body)
        except SyntaxError:
            continue
    return body


def _defined(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {t.id for t in targets if isinstance(t, ast.Name)}
    return set()


def check_node(node: ast.stmt) -> None:
    """Refuse any reviewed node that could read labels, files, network or pickles."""
    for child in ast.walk(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in child.names] if isinstance(child, ast.Import) else []
            names.append(getattr(child, "module", None) or "")
            for name in filter(None, names):
                if name.split(".")[0] not in ALLOWED_MODULES:
                    raise ValueError(f"Reviewed node imports disallowed module {name}")
        elif isinstance(child, ast.Name) and child.id in FORBIDDEN_NAMES:
            raise ValueError(f"Reviewed node references forbidden name {child.id}")
        elif isinstance(child, ast.Attribute) and child.attr in FORBIDDEN_ATTRIBUTES:
            raise ValueError(f"Reviewed node references forbidden attribute {child.attr}")
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            lowered = child.value.lower()
            if any(token in lowered for token in FORBIDDEN_STRINGS):
                raise ValueError("Reviewed node contains a label/annotation/network string")


def extract_nodes(source: str, names: list[str]) -> tuple[str, str]:
    """Return exact reviewed node text (source order, last definition wins) and its digest."""
    wanted = set(names)
    if len(wanted) != len(names) or not wanted:
        raise ValueError("Reviewed node names must be unique and non-empty")
    chosen: dict[str, ast.stmt] = {}
    order: dict[str, int] = {}
    for index, node in enumerate(_parse_cells(source)):
        for name in _defined(node) & wanted:
            if len(_defined(node)) != 1:
                raise ValueError(f"Reviewed node {name} shares a multi-target assignment")
            chosen[name], order[name] = node, index
    if set(chosen) != wanted:
        raise ValueError(f"Reviewed nodes absent from source: {sorted(wanted - set(chosen))}")
    for node in chosen.values():
        check_node(node)
    text = "\n\n".join(ast.unparse(chosen[n]) for n in sorted(chosen, key=order.__getitem__))
    return text, sha256(text.encode())


def _guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
    if name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"Reference namespace refuses import of {name}")
    return builtins.__import__(name, *args, **kwargs)


def compile_nodes(text: str, preamble: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compile already-checked node text into an isolated namespace without other cells."""
    safe = {k: getattr(builtins, k) for k in dir(builtins) if k not in FORBIDDEN_NAMES}
    safe["__import__"] = _guarded_import
    namespace: dict[str, Any] = {"__builtins__": MappingProxyType(safe), "__name__": "pilkwang_v15"}
    namespace.update(preamble or {})
    tree = ast.parse(text)
    for node in tree.body:
        check_node(node)
    exec(compile(tree, f"<{SOURCE_REF} v{SOURCE_VERSION} reviewed>", "exec"), namespace)
    return namespace


def load_review(path: Path = REVIEW_PATH) -> dict:
    review = json.loads(path.read_text())
    if review.get("source_sha256") != SOURCE_SHA256:
        raise SourceMismatch("Review is bound to a different source digest")
    return review


def load_reference(raw: bytes, review: dict | None = None) -> tuple[dict[str, Any], dict]:
    """Exact source plus completed review -> reviewed renderer namespace and receipt."""
    review = load_review() if review is None else review
    code = verified_code(raw)
    if review.get("status") != "reviewed" or not review.get("nodes"):
        raise SourceReviewPending(
            "Renderer nodes are not reviewed against the exact source; provide the bounded "
            "source handoff and complete review.json before any comparison"
        )
    text, digest = extract_nodes(code, list(review["nodes"]))
    if digest != review.get("node_text_sha256"):
        raise SourceMismatch("Reviewed node text digest changed")
    preamble = {}
    for module in review.get("preamble_modules", []):
        alias, name = module.split("=") if "=" in module else (module, module)
        preamble[alias] = _guarded_import(name, fromlist=["_"] if "." in name else [])
    namespace = compile_nodes(text, preamble)
    receipt = {
        "source_ref": SOURCE_REF,
        "source_version": SOURCE_VERSION,
        "source_sha256": SOURCE_SHA256,
        "extracted_sha256": EXTRACTED_SHA256,
        "nodes": list(review["nodes"]),
        "node_text_sha256": digest,
        "notebook_top_level_executed": False,
    }
    return namespace, receipt
