"""Real CPU subprocess and numeric tests; synthetic data is not model acceptance."""

import ast
import copy
import csv
import importlib.util
import io
import json
import subprocess
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("pair_test_" + name, HERE / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S, C, F = load("supervisor"), load("compile"), load("fusion")


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    w = csv.writer(stream, lineterminator="\n")
    w.writerow(F.COLUMNS)
    for uid, value in rows:
        w.writerow([uid, *([value] * 12)])
    return stream.getvalue().encode()


@pytest.fixture
def case(tmp_path, monkeypatch):
    uids = ["1.2.3", "1.2.4", "1.2.5"]
    monkeypatch.setattr(S, "PUBLIC_ROSTER", S.digest(S.canonical(uids)))
    monkeypatch.setattr(C, "PUBLIC_ROSTER", S.PUBLIC_ROSTER)
    roster = tmp_path / "test.csv"
    roster.write_text("StudyInstanceUID\n" + "\n".join(uids) + "\n")
    references = {
        "newton": csv_bytes(zip(uids, (0.9, 0.1, 0.5))),
        "colab": csv_bytes(zip(reversed(uids), (0.6, 0.8, 0.2))),
    }
    children, proofs = {}, {}
    for role, model in zip(S.ROLES, ("a" * 64, "b" * 64)):
        children[role] = dict(
            model_sha256=model,
            candidate_sha256="c" * 64,
            runtime_and_serving={"torch": "synthetic"},
            public_CSV_sha256=S.digest(references[role]),
        )
        proofs[role] = dict(
            status="actual_predictive_exact_test_roster_submission_created",
            mode="competition",
            test_roster_sha256=S.PUBLIC_ROSTER,
            actual_test_studies=3,
            submission_csv_absent=False,
            CSV_bytes=len(references[role]),
            CSV_sha256=S.digest(references[role]),
            GPU_names=["Tesla T4", "Tesla T4"],
            torch="synthetic",
            models=[
                dict(
                    arm="C",
                    trained_export_sha256=model,
                    candidate_sha256="c" * 64,
                    precision_accepted=True,
                    memory_accepted=True,
                )
            ],
        )

    def sources(extras=None):
        extras = extras or {}
        return {
            role: (
                "from pathlib import Path\nimport time,os,sys,subprocess\n"
                + f"with Path('order').open('a') as f: f.write({role!r})\n"
                + f"Path({S.CHILD_ROOT!r}).mkdir()\n"
                + f"Path({(S.CHILD_ROOT + '/original.bin')!r}).write_bytes({role.encode()!r})\n"
                + extras.get(role, "")
                + f"Path('submission.csv').write_bytes({references[role]!r})\n"
                + f"Path({S.CHILD_RECEIPT!r}).write_text({json.dumps(proofs[role])!r})\n"
            ).encode()
            for role in S.ROLES
        }

    fusion = (HERE / "fusion.py").read_bytes()

    def binding(src):
        return dict(
            mode="candidate",
            total_watchdog_seconds=S.TOTAL_SECONDS,
            public_watchdog_seconds=S.PUBLIC_SECONDS,
            children=children,
            source_sha256={**{k: S.digest(v) for k, v in src.items()}, "fusion": S.digest(fusion)},
        )

    def run(src=None, bound=None):
        src = sources() if src is None else src
        return S.run(
            src, fusion, references, bound or binding(src), working=tmp_path, roster_path=roster
        )

    return locals()


def test_two_real_processes_archive_same_paths_and_align_probability_mean(case):
    receipt = case["run"]()
    root = case["tmp_path"]
    assert (root / "order").read_text() == "newtoncolab"
    for role in S.ROLES:
        archive = root / S.ROOT / role
        assert (archive / S.CHILD_ROOT / "original.bin").read_bytes() == role.encode()
        assert (archive / "submission.csv").read_bytes() == case["references"][role]
        assert (archive / S.CHILD_RECEIPT).is_file()
    values = F._read_csv((root / "submission.csv").read_bytes(), tuple(case["uids"]))
    assert [r[0] for r in values] == pytest.approx([0.55, 0.45, 0.55])
    assert receipt["stage_order"] == ["newton", "colab"]
    assert all(r["public_max_probability_difference"] == 0 for r in receipt["stages"])
    assert not any(uid in json.dumps(receipt) for uid in case["uids"])


def test_public_comparison_is_numeric_but_hidden_never_uses_reference(case):
    proof = copy.deepcopy(case["proofs"]["newton"])
    changed = csv_bytes(zip(case["uids"], (0.90001, 0.10001, 0.50001)))
    proof.update(CSV_bytes=len(changed), CSV_sha256=S.digest(changed))
    args = (proof, case["children"]["newton"], changed, case["uids"])
    assert S.check_child(*args, case["references"]["newton"], F._read_csv) == pytest.approx(1e-5)
    bad = csv_bytes(zip(case["uids"], (0.2, 0.1, 0.5)))
    proof.update(CSV_bytes=len(bad), CSV_sha256=S.digest(bad))
    with pytest.raises(ValueError, match="prediction difference"):
        S.check_child(proof, args[1], bad, args[3], case["references"]["newton"], F._read_csv)
    hidden = ["2.3.4", "2.3.5", "2.3.6"]
    hidden_csv = csv_bytes(zip(hidden, (0.2, 0.1, 0.5)))
    proof.update(
        CSV_bytes=len(hidden_csv),
        CSV_sha256=S.digest(hidden_csv),
        test_roster_sha256=S.digest(S.canonical(hidden)),
    )
    assert (
        S.check_child(proof, args[1], hidden_csv, hidden, b"not a reference", F._read_csv) is None
    )


@pytest.mark.parametrize("role", S.ROLES)
def test_failure_leaves_no_partial_final_and_preserves_completed_child(case, role):
    src = case["sources"](
        {role: "Path('submission.csv').write_text('partial')\nraise SystemExit(7)\n"}
    )
    with pytest.raises(ValueError, match="Child failed"):
        case["run"](src)
    root = case["tmp_path"]
    assert not (root / "submission.csv").exists()
    assert not (root / S.ACCEPTED).exists()
    if role == "colab":
        assert (root / S.ROOT / "newton/submission.csv").read_bytes() == case["references"][
            "newton"
        ]
    else:
        assert (root / "order").read_text() == "newton"


def test_shared_watchdog_kills_descendant_and_never_starts_second(case, monkeypatch):
    monkeypatch.setattr(S, "PUBLIC_SECONDS", 0.7)
    src = case["sources"](
        {
            "newton": (
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                "Path('grandchild.pid').write_text(str(p.pid))\n"
                "Path('submission.csv').write_text('partial')\ntime.sleep(60)\n"
            )
        }
    )
    start = time.monotonic()
    with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
        case["run"](src)
    assert time.monotonic() - start < 4
    root = case["tmp_path"]
    pid = (root / "grandchild.pid").read_text()
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", pid], capture_output=True, text=True
    ).stdout.strip()
    assert not state or state.startswith("Z")
    assert (root / "order").read_text() == "newton"
    assert not (root / "submission.csv").exists()


def test_watchdog_budget_is_not_reset_for_second_model(case, monkeypatch):
    monkeypatch.setattr(S, "PUBLIC_SECONDS", 0.7)
    src = case["sources"]({"newton": "time.sleep(.45)\n", "colab": "time.sleep(.45)\n"})
    with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
        case["run"](src)
    root = case["tmp_path"]
    assert (root / "order").read_text() == "newtoncolab"
    assert (root / S.ROOT / "newton/submission.csv").is_file()
    assert not (root / "submission.csv").exists()


def test_source_change_refused_before_launch(case):
    src = case["sources"]()
    bound = case["binding"](src)
    src["colab"] += b"\n"
    with pytest.raises(ValueError, match="source hash"):
        case["run"](src, bound)
    assert not (case["tmp_path"] / "order").exists()


@pytest.mark.parametrize("name", (S.CHILD_ROOT, S.CHILD_RECEIPT, "submission.csv", S.ROOT))
def test_stale_original_is_preserved_without_launch(case, name):
    path = case["tmp_path"] / name
    path.write_bytes(b"preserve original")
    with pytest.raises(ValueError, match="Fresh"):
        case["run"]()
    assert path.read_bytes() == b"preserve original"
    assert not (case["tmp_path"] / "order").exists()


@pytest.mark.parametrize("values", ((float("nan"), 0.2, 0.4), (1.1, 0.2, 0.4)))
def test_invalid_numeric_values_never_fused(case, values):
    with pytest.raises(ValueError, match="finite"):
        F.fuse_csv(csv_bytes(zip(case["uids"], values)), case["references"]["colab"], case["uids"])


def test_missing_duplicate_or_extra_rows_never_fused(case):
    for uids in (case["uids"][:2], [case["uids"][0]] * 3, case["uids"] + ["9.9.9"]):
        with pytest.raises(ValueError):
            F.fuse_csv(csv_bytes((u, 0.5) for u in uids), case["references"]["colab"], case["uids"])


def test_compiler_requires_both_original_final_acceptances(case, monkeypatch):
    items = {}
    for role in S.ROLES:
        chosen = dict(
            arm="C",
            architecture={"context_cap": 24},
            provenance={"epoch": 6, "completed_updates": 2916},
            model={"sha256": case["children"][role]["model_sha256"]},
        )
        public = copy.deepcopy(case["proofs"][role])
        public.update({k: "synthetic-" + k for k in C.FIELDS})
        public["models"][0]["candidate_sha256"] = C.digest(C.canonical(chosen))
        q = copy.deepcopy(public)
        q["status"] = "actual_trained_predictive_public_T4_qualification_accepted"
        b = dict(
            mode="competition",
            branch=C.BRANCHES[role],
            qualification=q,
            models=[chosen],
            **{k: public[k] for k in C.FIELDS[4:]},
        )
        source = ("BINDING = " + repr(b) + "\n").encode()
        # Original public references use the official order; runtime children
        # may emit a different order, which the fusion still aligns by UID.
        ref = (
            csv_bytes(zip(case["uids"], (0.2, 0.8, 0.6)))
            if role == "colab"
            else case["references"][role]
        )
        public.update(CSV_sha256=C.digest(ref), CSV_bytes=len(ref))
        items[role] = dict(
            source=source,
            actual_source_sha256=C.digest(source),
            qualification=q,
            public=public,
            public_csv=ref,
        )
    monkeypatch.setattr(C, "NEWTON_SOURCE", C.digest(items["newton"]["source"]))
    monkeypatch.setattr(C, "NEWTON_MODEL", "a" * 64)
    source, receipt = C.build(items)
    ast.parse(source)
    assert receipt["binding"]["weights"] == [0.5, 0.5]
    assert receipt["actual_combined_GPU_run"] is False
    assert receipt["competition_entry"] is False
    for role in S.ROLES:
        bad = copy.deepcopy(items)
        bad[role]["public"]["models"][0]["trained_export_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="trained child"):
            C.build(bad)
    with pytest.raises(ValueError, match="Two original"):
        C.build({"newton": items["newton"]})
