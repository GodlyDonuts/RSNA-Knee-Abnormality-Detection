"""Prove that an external member differs only by two CSV output exports."""

from __future__ import annotations

import ast
import copy
import hashlib

HEADER = "%%writefile /tmp/raptor_run.py"
SUBMISSION = "/kaggle/working/submission.csv"
MEMBER = "/kaggle/working/raptor_sub.csv"
RAW = "/kaggle/working/raptor_raw.csv"
RAW_EXPORT = """_raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)
_raw.insert(0, "StudyInstanceUID", test_ids)
_raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)
"""


def tree_sha(tree):
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def one_function(tree, name):
    found = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name]
    if len(found) != 1:
        raise ValueError("One exact top-level function required: " + name)
    return found[0]


def output_assignment(function, path):
    matches = [
        x
        for x in function.body
        if isinstance(x, ast.Assign)
        and len(x.targets) == 1
        and isinstance(x.targets[0], ast.Name)
        and x.targets[0].id == "out"
        and isinstance(x.value, ast.Constant)
        and x.value.value == path
    ]
    if len(matches) != 1:
        raise ValueError("One exact output filename assignment required")
    return matches[0]


def inspect_member(parent, cell):
    header, sep, body = cell.partition("\n")
    if header != HEADER or not sep:
        raise ValueError("Exact original member script writefile boundary required")
    left, right = ast.parse(parent), ast.parse(body)
    p = one_function(left, "main")
    r = one_function(right, "main")
    expected = ast.parse(RAW_EXPORT).body
    sites = [
        i
        for i in range(len(r.body) - 2)
        if all(tree_sha(x) == tree_sha(y) for x, y in zip(r.body[i : i + 3], expected, strict=True))
    ]
    if len(sites) != 1:
        raise ValueError("One exact raw probability export, without new inference, required")
    i = sites[0]
    output_assignment(p, SUBMISSION)
    normalized = copy.deepcopy(right)
    n = one_function(normalized, "main")
    del n.body[i : i + 3]
    output_assignment(n, MEMBER).value = ast.Constant(value=SUBMISSION)
    if tree_sha(left) != tree_sha(normalized):
        raise ValueError("Whole original Python closure differs beyond the two CSV exports")
    return {
        "raw_export_insertion_index": i,
        "parent_module_AST_sha256": tree_sha(left),
        "normalized_R6_member_module_AST_sha256": tree_sha(normalized),
        "whole_module_equal_except_CSV_exports": True,
        "new_inference_or_model_calls": 0,
    }


def replace_node(source, node, replacement):
    lines = source.encode().splitlines(keepends=True)
    assert node.end_lineno is not None and node.end_col_offset is not None
    a = sum(map(len, lines[: node.lineno - 1])) + node.col_offset
    b = sum(map(len, lines[: node.end_lineno - 1])) + node.end_col_offset
    return (source.encode()[:a] + replacement.encode() + source.encode()[b:]).decode()


def export_as_member(parent, original_cell, paired_source):
    proof = inspect_member(parent, original_cell)
    tree = ast.parse(paired_source)
    f = one_function(tree, "main")
    p = one_function(ast.parse(parent), "main")
    i = proof["raw_export_insertion_index"]
    if len(f.body) != len(p.body) or i == 0 or tree_sha(f.body[i - 1]) != tree_sha(p.body[i - 1]):
        raise ValueError("Qualified bounded parent main and exact export position required")
    output_assignment(f, SUBMISSION).value = ast.Constant(value=MEMBER)
    f.body[i:i] = copy.deepcopy(ast.parse(RAW_EXPORT).body)
    changed = replace_node(
        paired_source, one_function(tree, "main"), ast.unparse(ast.fix_missing_locations(f))
    )
    tree = ast.parse(changed)
    f = one_function(tree, "_adapted_competition")
    paths = [n for n in ast.walk(f) if isinstance(n, ast.Constant) and n.value == SUBMISSION]
    if len(paths) != 1:
        raise ValueError("One exact standalone member CSV verification path required")
    paths[0].value = MEMBER
    changed = replace_node(
        changed,
        one_function(tree, "_adapted_competition"),
        ast.unparse(ast.fix_missing_locations(f)),
    )
    compile(changed, "paired_R6_member_script", "exec")
    return HEADER + "\n" + changed, proof
