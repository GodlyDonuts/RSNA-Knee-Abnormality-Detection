"""Prepared replacement for the frozen portable Owner.run; no launch on import."""

import copy
import queue
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

SCIENCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
VERIFIED = "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"
PRESERVATION_END = datetime(2026, 10, 22, 12, tzinfo=timezone.utc).timestamp()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def run(owner, *, deadline_policy, placement_policy, once, audit_dir,
        clock=time.time, sleep=time.sleep):
    """Use original phase methods once; isolate C0 placement from B/C preservation.

    Root owns source freeze and exclusive idle-owner handoff before calling this.
    No GPU job, transfer or publication is restarted by this run replacement.
    """
    require(owner.payload["prepared"]["source_git"] == SCIENCE, "Frozen prepared science required")
    require(clock() < PRESERVATION_END, "Finite preservation deadline passed")
    require((owner.journal / "owner-intent.json").is_file(), "Original idle owner intent required")
    prior_phase = [
        path for path in owner.journal.iterdir()
        if (
            path.name.endswith(("-intent.json", "-result.json", "-transport.json"))
            and path.name != "owner-intent.json"
        ) or path.name.startswith(("deadline-", "known-dispatches-", "publication-unresolved"))
        or path.name == "owner-ended.json"
    ]
    require(not prior_phase, "Prephase-only successor refuses existing operation evidence")
    audit_dir = Path(audit_dir)
    owner.deadline = PRESERVATION_END
    original_invoke = owner.invoke.__func__
    original_transfer = owner.transfer.__func__
    budget = deadline_policy.Budget(
        deployment_wall_seconds=3 * 3600, scheduler_wait_seconds=2 * 3600,
        publication_return_seconds=3600, cache_admission_seconds=90 * 60,
        c0_transfer_seconds=150 * 60, c0_dispatch_control_seconds=45 * 60,
    )
    events = queue.Queue()
    dispatched, published, complete, failed = {}, set(), {}, {}
    observation_problems, observation_incidents = {}, {}
    placement_state, publication_problem, returned = None, None, None
    thread = None

    def emit(name, value):
        once(owner.journal / (name + ".json"), value)
        once(audit_dir / ("actual-" + name + ".json"), value)

    def deadline(stage):
        result = deadline_policy.assess_dispatch(
            datetime.fromtimestamp(clock(), timezone.utc), stage, budget,
            preservation_owner_expires_at=datetime.fromtimestamp(PRESERVATION_END, timezone.utc),
        )
        emit("deadline-" + stage, result)
        return result["deadline_allows_new_stage"]

    def pause():
        sleep(max(0, min(owner.args.poll_seconds, PRESERVATION_END - clock())))

    def invoke(target, node, mode, phase=None, extra=None):
        # No nested map or credential is shared between placement and publication.
        isolated = copy.copy(target)
        isolated.payload = copy.deepcopy(target.payload)
        require("token" not in isolated.payload, "Credentials belong only to original publish stdin")
        return original_invoke(isolated, node, mode, phase, copy.deepcopy(extra))

    def transfer(target, returning=False):
        isolated = copy.copy(target)
        isolated.payload = copy.deepcopy(target.payload)
        isolated.invoke = types.MethodType(
            lambda self, node, mode, phase=None, extra=None: invoke(self, node, mode, phase, extra),
            isolated,
        )
        return original_transfer(isolated, returning=returning)

    def known_jobs():
        return {a: copy.deepcopy(j) for value in dispatched.values() for a, j in value["jobs"].items()}

    def admit_dispatch(value, node, expected):
        require(value["status"] == "actual_predictive_placement_jobs_submitted_once"
                and value["source_git"] == SCIENCE
                and set(value["jobs"]) == set(expected), "Exact original dispatch result required")
        seen = {j["GPU_job"] for j in known_jobs().values()}
        for arm, job in value["jobs"].items():
            require(job["arm"] == arm and job["node"] == node
                    and job["source_git"] == SCIENCE
                    and isinstance(job["GPU_job"], str) and job["GPU_job"].isdecimal()
                    and job["GPU_job"] not in seen, "Independent actual arm/job/source required")
            seen.add(job["GPU_job"])
        emit("known-dispatches-" + ("C0" if node == "evc27" else "BC"), value)
        dispatched[node] = copy.deepcopy(value)
        owner.payload["all_dispatched_jobs"] = known_jobs()

    def finish(reason):
        result = {
            "status": "actual_resilient_predictive_owner_finished",
            "reason": reason, "science_git": SCIENCE,
            "placement_source_git": owner.args.source_git, "jobs": known_jobs(),
            "complete": sorted(complete), "failed": sorted(failed),
            "privately_verified_epochs": sorted(published), "C0_return": returned,
            "C0_placement": placement_state, "publication_problem": publication_problem,
            "unresolved_node_observations": copy.deepcopy(observation_problems),
            "placement_thread_alive": thread is not None and thread.is_alive(),
            "preservation_deadline_UTC": datetime.fromtimestamp(
                PRESERVATION_END, timezone.utc).isoformat(),
            "automatic_restart": False, "remote_jobs_cancelled": [],
            "clinical_accuracy_or_score": False,
        }
        once(owner.journal / "owner-result.json", result)
        once(audit_dir / "actual-owner-finished.json", result)
        return result

    # Repeated read-only waiting stops at the admission cutoff, not at noon.
    while clock() < PRESERVATION_END:
        prospective = deadline_policy.assess_dispatch(
            datetime.fromtimestamp(clock(), timezone.utc), "cache_admission", budget,
            preservation_owner_expires_at=datetime.fromtimestamp(PRESERVATION_END, timezone.utc),
        )
        if not prospective["deadline_allows_new_stage"]:
            emit("deadline-cache_admission", prospective)
            return finish("no_new_admission_after_dispatch_cutoff")
        receipts = owner.original.cache_receipts()
        if receipts is not None:
            owner.payload.update(raw_collected=receipts["raw"], context_collected=receipts["context"])
            break
        pause()
    else:
        return finish("finite_preservation_deadline_no_dispatch")
    require(deadline("cache_admission"), "Cache admission cutoff crossed before original phase")
    admitted = invoke(owner, "evc33", "admit", "admission-BC")
    require(admitted["arms"] == ["B", "C"], "B/C-only original admission required")
    if not deadline("bc_dispatch"):
        return finish("no_new_B_C_dispatch_after_cutoff")
    bc = invoke(owner, "evc33", "dispatch", "dispatch-BC", {"admitted": admitted})
    admit_dispatch(bc, "evc33", {"B", "C"})

    # The task has its own payload; only main merges an actual C0 dispatch.
    placement_owner = copy.copy(owner)
    placement_owner.payload = copy.deepcopy(owner.payload)

    def place_C0():
        phase = "contexts-C0"
        try:
            if not deadline("c0_transfer"):
                events.put(("not_launched", {"phase": phase, "reason": "launch_cutoff"}))
                return
            portable = transfer(placement_owner)
            placement_owner.payload["portable"] = portable
            phase = "dispatch-C0"
            if not deadline("c0_dispatch"):
                events.put(("not_launched", {
                    "phase": phase, "reason": "launch_cutoff", "original_transfer": portable,
                }))
                return
            actual = invoke(placement_owner, "evc27", "dispatch", "dispatch-C0")
            events.put(("dispatched", {"dispatch": actual, "portable": portable}))
        except Exception as error:
            events.put(("unresolved", {
                "phase": phase, "error_type": type(error).__name__,
            }))

    thread = threading.Thread(target=place_C0, name="original-C0-placement-once", daemon=True)
    thread.start()
    while clock() < PRESERVATION_END:
        try:
            kind, value = events.get_nowait()
        except queue.Empty:
            pass
        else:
            require(placement_state is None, "C0 placement may produce only one terminal event")
            if kind == "dispatched":
                try:
                    admit_dispatch(value["dispatch"], "evc27", {"C0"})
                    owner.payload["portable"] = copy.deepcopy(value["portable"])
                except Exception as error:
                    placement_state = placement_policy.continuation_after_exception(
                        known_jobs(), phase="dispatch-C0", error_type=type(error).__name__,
                    )
                    emit("C0-placement-unresolved", placement_state)
                else:
                    placement_state = {"status": "original_C0_dispatched_once"}
            elif kind == "not_launched":
                placement_state = {"status": "C0_not_launched_before_cutoff", **value}
                emit("C0-placement-not-launched", placement_state)
            else:
                require(kind == "unresolved", "Unknown placement result")
                placement_state = placement_policy.continuation_after_exception(
                    known_jobs(), phase=value["phase"], error_type=value["error_type"],
                )
                emit("C0-placement-unresolved", placement_state)

        arms = {}
        for node, receipt in dispatched.items():
            try:
                progress = invoke(owner, node, "progress", extra={"dispatched": receipt})
                require(set(progress["arms"]) == set(receipt["jobs"]), "Progress changed known cohort")
                for arm, item in progress["arms"].items():
                    require(item["GPU_job"] == receipt["jobs"][arm]["GPU_job"], "Wrong original job")
            except Exception as error:
                # A parser/numerical/transport exception is not a Slurm terminal.
                # Preserve its first evidence, retain original jobs, and retry
                # only this read-only observation in subsequent finite cycles.
                if node not in observation_problems:
                    incident = observation_incidents.get(node, 0) + 1
                    observation_incidents[node] = incident
                    problem = {
                        "status": "original_node_observation_unresolved",
                        "node": node, "incident": incident,
                        "error_type": type(error).__name__,
                        "known_original_jobs": copy.deepcopy(receipt["jobs"]),
                        "training_failure_inferred": False, "new_mutations": False,
                    }
                    observation_problems[node] = problem
                    emit(f"observation-unresolved-{node}-{incident:03d}", problem)
                continue
            if node in observation_problems:
                incident = observation_problems.pop(node)["incident"]
                emit(f"observation-recovered-{node}-{incident:03d}", {
                    "status": "original_node_observation_recovered",
                    "node": node, "incident": incident,
                    "known_original_jobs": copy.deepcopy(receipt["jobs"]),
                    "new_mutations": False,
                })
            arms.update(progress["arms"])
        for arm, item in arms.items():
            if owner.original.completed(item["slurm"]) and item["terminal"] and arm not in complete:
                complete[arm] = item
                once(audit_dir / ("actual-fixed-final-" + arm + ".json"), {
                    **item["terminal"], "actual_Slurm_terminal": item["slurm"],
                    "stderr_bytes": item["stderr_bytes"],
                    "placement_source_git": owner.args.source_git,
                })
            elif (owner.original.terminal(item["slurm"]) and
                  not owner.original.completed(item["slurm"]) and arm not in failed):
                failed[arm] = item
                once(audit_dir / ("actual-failure-" + arm + ".json"), item)

        candidate = owner.original.ready_epoch({"arms": arms}, published)
        if candidate and publication_problem is None:
            arm, epoch = candidate
            node = "evc27" if arm == "C0" else "evc33"
            try:
                actual = invoke(owner, node, "publish", f"publish-{arm}-{epoch:03d}",
                                {"arm": arm, "epoch": epoch, "dispatched": dispatched[node]})
                require(actual["status"] == VERIFIED and actual["objects"] == 5
                        and (actual["arm"], actual["epoch"], actual["source_git"]) ==
                        (arm, epoch, SCIENCE), "Exact original epoch preservation required")
            except Exception as error:
                publication_problem = {
                    "status": "original_publication_operation_unresolved",
                    "arm": arm, "epoch": epoch, "error_type": type(error).__name__,
                    "automatic_retry": False, "further_publication_mutations_stopped": True,
                    "already_dispatched_observation_continues": True,
                }
                emit("publication-unresolved", publication_problem)
            else:
                published.add((arm, epoch))
                sleep(0)  # Let placement report completion without serializing remote work.
                continue

        all_known_terminal = (
            set(arms) == set(known_jobs()) and not observation_problems
            and len(complete) + len(failed) == len(known_jobs())
        )
        if all_known_terminal and placement_state is not None:
            if "C0" in complete and all(("C0", epoch) in published for epoch in range(1, 7)):
                try:
                    returned = transfer(owner, returning=True)
                except Exception as error:
                    emit("C0-return-unresolved", placement_policy.continuation_after_exception(
                        known_jobs(), phase="return-C0", error_type=type(error).__name__,
                    ))
                    return finish("original_C0_final_return_unresolved_no_retry")
            return finish("known_jobs_terminal_preserve_unresolved_operations")
        pause()
    return finish("preservation_deadline_existing_jobs_require_read_only_continuation")
