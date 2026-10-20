"""Prepared qualification-only overlap; no execution or filesystem writes on import."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ORIGINAL_COMPILER_SHA = "9b1798683fe21cbc36cbceb48837ee7a4e7da3e464d238e683636163939d04c2"
ORIGINAL_SOURCE_SHA = "69f0d91347a49fde17b4293b3f21493a41bdd8f176aa6d77156540a04a161556"
MODEL_SHA = "d478a502e6fd33be2ad9d781acd91d3d2b6b252629262031fba8282a91cf5bca"
FUNCTION_NAME = "_adapted_qualify"

PARALLEL_FUNCTION = r'''
def _adapted_qualify():
    import tarfile
    from concurrent.futures import ThreadPoolExecutor

    started = time.monotonic()
    assert not Path('/kaggle/working/submission.csv').exists()
    code = {'__name__': 'actual_original_pixel_codec'}
    exec(compile(_ADAPTED_CODEC, 'actual_original_pixel_codec', 'exec'), code)
    mix = {'__name__': 'actual_checkpointed_whole94_encoder'}
    exec(compile(_ADAPTED_MIXIN, 'actual_checkpointed_whole94_encoder', 'exec'), mix)
    importance = {'__name__': 'actual_accepted_IID16_training_readout'}
    exec(compile(_ADAPTED_IMPORTANCE, 'actual_accepted_IID16_training_readout', 'exec'), importance)

    class TrainingGraph(importance['ImportanceReadout'], mix['CheckpointedEncode'], RaptorClassifier):
        pass

    p = _adapted_locate(
        'index.private.json', 21324,
        '3a1567989bf65779aeb19785dafdedef7386fb6bedba5ec45da6d3a67a9a79e3',
    )
    rows = json.loads(p.read_text())['records']
    assert len(rows) == 32
    common_startup_seconds = time.monotonic() - started

    def worker(i):
        device_started = time.monotonic()
        model = reference = None
        # no_grad is thread-local: the outer decorator does not cover these workers.
        with torch.cuda.device(i), torch.no_grad():
            device = 'cuda:' + str(i)
            try:
                model, res = load_model(str(_adapted_model_path), ARMS[0]['arch'], 384, device)
                reference = TrainingGraph(
                    build_backbone(ARMS[0]['arch'], pretrained=False), F_dim=1024
                ).to(device)
                reference.load_state_dict(model.state_dict(), strict=True)
                reference.eval()
                assert model.state_dict().keys() == reference.state_dict().keys()
                assert all(
                    torch.equal(t, reference.state_dict()[n])
                    for n, t in model.state_dict().items()
                )
                maximum = 0.0
                preparation_seconds = reference_forward_seconds = serving_forward_seconds = 0.0
                study_seconds = []
                # Independent handles prevent concurrent seek/read on a shared TarFile.
                with tarfile.open(p.parent / 'pixels.tar', 'r:') as tar:
                    members = {m.name: m for m in tar.getmembers()}
                    assert len(members) == 32
                    torch.cuda.synchronize(i)
                    startup_seconds = time.monotonic() - device_started
                    print(json.dumps({
                        'ranking_parallel_device_started': i,
                        'startup_seconds': startup_seconds,
                    }), flush=True)
                    for row in rows:
                        study_started = time.monotonic()
                        preparation_started = time.monotonic()
                        member = members[row['file']]
                        assert member.isreg() and member.size == row['bytes']
                        with tar.extractfile(member) as handle:
                            assert handle is not None
                            volume, mask = code['decode'](handle.read(), row['sha256'])
                        assert hashlib.sha256(volume.tobytes()).hexdigest() == row['volume_sha256']
                        assert hashlib.sha256(mask.tobytes()).hexdigest() == row['mask_sha256']
                        preparation_seconds += time.monotonic() - preparation_started
                        windows, values = [], []
                        for mirror in (False, True):
                            preparation_started = time.monotonic()
                            cpu = eval_windows(volume, mask, 94, 320, norm=NORM, mirror=mirror)
                            assert hashlib.sha256(cpu.contiguous().numpy().tobytes()).hexdigest() == row[
                                'original_mirrored_input_sha256' if mirror else 'original_plain_input_sha256'
                            ]
                            gpu = gpu_windows(volume, mask, 94, 320, device, norm=NORM, mirror=mirror)
                            assert torch.equal(cpu, gpu.cpu())
                            windows.append(gpu)
                            torch.cuda.synchronize(i)
                            preparation_seconds += time.monotonic() - preparation_started
                            forward_started = time.monotonic()
                            with torch.autocast('cuda', dtype=torch.float16):
                                logits = reference(gpu[None]).float()
                            assert logits.shape == (1, 12) and torch.isfinite(logits).all()
                            values.append(logits.sigmoid()[0].cpu().numpy())
                            torch.cuda.synchronize(i)
                            reference_forward_seconds += time.monotonic() - forward_started
                        forward_started = time.monotonic()
                        actual = infer_pair_probs(model, *windows, device)
                        torch.cuda.synchronize(i)
                        serving_forward_seconds += time.monotonic() - forward_started
                        expected = .5 * (values[0] + values[1])
                        error = float(np.max(np.abs(actual - expected)))
                        maximum = max(maximum, error)
                        assert error <= 1e-6, 'Served original class differs from complete trained-model graph'
                        del windows, cpu, gpu, logits
                        study_seconds.append(time.monotonic() - study_started)
                assert len(study_seconds) == 32
                record = {
                    'index': i, 'GPU': torch.cuda.get_device_name(i), 'MRI_studies': 32,
                    'both_original94_input_views_bit_equal': True,
                    'maximum_original_class_vs_training_graph_probability_error': maximum,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(i),
                    'complete_learned_graph_tensor_values_equal': True,
                    'startup_seconds': startup_seconds,
                    'preparation_seconds': preparation_seconds,
                    'reference_forward_seconds': reference_forward_seconds,
                    'serving_forward_seconds': serving_forward_seconds,
                    'elapsed_seconds': time.monotonic() - device_started,
                    'per_study_seconds': study_seconds,
                    'timing_includes_synchronization': True,
                }
                print(json.dumps({
                    'ranking_parallel_device_completed': i,
                    'MRI_studies': 32,
                    'elapsed_seconds': record['elapsed_seconds'],
                }), flush=True)
                return record
            finally:
                del model, reference
                torch.cuda.empty_cache()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, i) for i in range(2)]
        devices = [future.result() for future in futures]
    assert [d['index'] for d in devices] == [0, 1]
    assert not Path('/kaggle/working/submission.csv').exists()
    return {
        'status': 'actual_teacher_pair_final_epoch_bothT4_original32_MRI_and_training_graph_qualified',
        'epoch': _ADAPTED_TRAINING['epoch'], 'model_sha256': _ADAPTED_SHA,
        'training_source_sha256': _ADAPTED_TRAINING['source_sha256'],
        'devices': devices, 'qualified_studies': 32,
        'qualification_cannot_submit': True, 'competition_entry': False,
        'clinical_accuracy_claim': False, 'parallel_device_workers': 2,
        'common_startup_seconds': common_startup_seconds,
        'qualification_function_elapsed_seconds': time.monotonic() - started,
        'VM_startup_and_adapted_boot_timing_included': False,
        'runtime_admission_inferred': False,
    }
'''.lstrip()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _function(source):
    functions = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == FUNCTION_NAME
    ]
    if len(functions) != 1:
        raise ValueError("One exact original qualification function required")
    return functions[0]


def _original_function(compiler_source):
    if digest(compiler_source) != ORIGINAL_COMPILER_SHA:
        raise ValueError("Exact originally accepted compiler required")
    build = next(
        node for node in ast.parse(compiler_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "build"
    )
    assignments = [
        node for node in build.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "patch" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Exact original compiler patch required")
    patch = ast.literal_eval(assignments[0].value)
    patch = patch.replace("actual_matched_IID16_parent_epoch_", "actual_teacher_pair_final_epoch_")
    return ast.get_source_segment(patch, _function(patch))


def _replace_qualification(source, original_function):
    node = _function(source)
    if ast.get_source_segment(source, node) != original_function:
        raise ValueError("Original qualification body changed")
    if [ast.unparse(item) for item in node.decorator_list] != ["torch.no_grad()"]:
        raise ValueError("Original no_grad qualification decorator required")
    lines = source.splitlines(keepends=True)
    # Keep every original byte outside this one definition, including its decorator.
    result = "".join(lines[:node.lineno - 1]) + PARALLEL_FUNCTION + "".join(lines[node.end_lineno:])
    compile(result, "prepared_exact_rank_parallel_qualification", "exec")
    return result


def adapt(source, *, compiler_source):
    """Return a distinct compiled qualification and bindings; never write or execute it."""
    if digest(source) != ORIGINAL_SOURCE_SHA:
        raise ValueError("Exact original final ranking qualification source required")
    original_function = _original_function(compiler_source)
    result = _replace_qualification(source, original_function)
    metadata = {
        "status": "prepared_only_exact_final_ranking_parallel_T4_qualification",
        "original_compiler_sha256": ORIGINAL_COMPILER_SHA,
        "original_qualification_sha256": ORIGINAL_SOURCE_SHA,
        "model_sha256": MODEL_SHA,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_qualification_function_sha256": digest(original_function),
        "parallel_qualification_function_sha256": digest(PARALLEL_FUNCTION),
        "parallel_source_sha256": digest(result),
        "changed_function": FUNCTION_NAME,
        "device_workers": 2,
        "original_MRI_studies_per_device": 32,
        "original_views_per_study": 2,
        "original_windows_per_view": 94,
        "original_served_vs_training_probability_tolerance": 1e-6,
        "competition_source_changed": False,
        "new_model_or_pixel_downloads": False,
        "actual_T4_execution_or_runtime_acceptance": False,
        "launch_admitted": False,
    }
    return result, metadata
