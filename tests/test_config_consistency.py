"""The shell library and config.py must agree on where data lives.

`scripts/lib/common.sh` resolves the per-model data trees in bash so the
runners stay fast and dependency-free, while `iesseeg/config.py` resolves
them in Python for the library and the evaluator. That is the same
knowledge in two languages, so these tests execute the shell helpers and
compare their answers against the Python mapping. If someone adds a model
or renames a tree on one side only, this fails.
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from iesseeg import config  # noqa: E402

COMMON_SH = os.path.join(REPO_ROOT, "scripts", "lib", "common.sh")
FAKE_DATA_ROOT = "/tmp/iesseeg-consistency-check"

MODELS = sorted(config.MODEL_DATA_SUBDIR)


def run_helper(function, argument):
    """Source common.sh and call one of its helpers."""
    result = subprocess.run(
        ["bash", "-c", f'source "{COMMON_SH}" && {function} "{argument}"'],
        capture_output=True, text=True,
        env={**os.environ, "IESSEEG_DATA_ROOT": FAKE_DATA_ROOT,
             "IESSEEG_SCRATCH": "/tmp/iesseeg-consistency-check-scratch"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"{function}({argument}) failed: {result.stderr.strip()}")
    return result.stdout.strip()


@pytest.mark.parametrize("model", MODELS)
def test_training_tree_matches_python(model):
    shell = run_helper("model_data_dir", model)
    expected = os.path.join(FAKE_DATA_ROOT, config.MODEL_DATA_SUBDIR[model])
    assert shell == expected


@pytest.mark.parametrize("model", MODELS)
def test_test_tree_matches_python(model):
    shell = run_helper("model_test_dir", model)
    expected = os.path.join(FAKE_DATA_ROOT, config.MODEL_TEST_SUBDIR[model])
    assert shell == expected


@pytest.mark.parametrize("task", list(config.TASKS))
def test_label_key_matches_python(task):
    assert run_helper("label_key_for", task) == config.TASK_LABEL_COLUMN[task]


def test_both_sides_know_the_same_models():
    assert set(config.MODEL_DATA_SUBDIR) == set(config.MODEL_TEST_SUBDIR)
    for function in ("model_data_dir", "model_test_dir"):
        with pytest.raises(RuntimeError, match="unknown model"):
            run_helper(function, "not_a_model")


def test_unknown_task_is_rejected_by_shell():
    with pytest.raises(RuntimeError, match="unknown task"):
        run_helper("label_key_for", "not_a_task")


def test_every_registered_model_has_a_runner():
    """Each model in the aggregator maps to a directory that actually
    carries the two runner scripts run_benchmark.sh will invoke."""
    from iesseeg.evaluation.aggregate import MODEL_ORDER

    assert set(MODEL_ORDER) == set(config.MODEL_DATA_SUBDIR)
    assert set(MODEL_ORDER) == set(config.MODEL_BASELINE_DIR)

    for model in MODEL_ORDER:
        model_dir = os.path.join(REPO_ROOT, "baselines", config.MODEL_BASELINE_DIR[model])
        assert os.path.isdir(model_dir), f"{model}: missing {model_dir}"
        assert os.path.isfile(os.path.join(model_dir, "train_all.sh"))
        assert os.path.isfile(os.path.join(model_dir, "inference_all.sh"))


def test_run_benchmark_registers_every_model():
    """run_benchmark.sh declares its own model -> directory map in bash.

    It drives the whole sweep, so a model missing from it would simply
    never run rather than fail. Check it against the Python mapping.
    """
    script = open(os.path.join(REPO_ROOT, "scripts", "run_benchmark.sh")).read()

    from iesseeg.evaluation.aggregate import MODEL_ORDER

    for model in MODEL_ORDER:
        assert f"[{model}]=" in script, f"{model} is not registered in run_benchmark.sh"

    default_line = next(l for l in script.splitlines() if l.startswith("MODELS="))
    for model in MODEL_ORDER:
        assert model in default_line, f"{model} missing from run_benchmark.sh default MODELS"
