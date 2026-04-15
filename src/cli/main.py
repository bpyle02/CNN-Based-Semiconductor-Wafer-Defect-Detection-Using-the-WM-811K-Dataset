"""Root Typer application for ``wafer-cli``.

Usage::

    wafer-cli --help
    wafer-cli train --model cnn --epochs 10
    wafer-cli benchmark --model cnn
    wafer-cli eval --model cnn --checkpoint checkpoints/cnn_best.pth

Each subcommand below is implemented in its own module under
``src/cli/commands/`` and forwards to the existing script / module
entry point so flag parity with the legacy ``scripts/*.py`` CLIs is
preserved verbatim.
"""

from __future__ import annotations

import typer

from src.cli._common import FORWARD_CONTEXT_SETTINGS
from src.cli.commands import active_learn as _active_learn
from src.cli.commands import benchmark as _benchmark
from src.cli.commands import bootstrap as _bootstrap
from src.cli.commands import calibrate as _calibrate
from src.cli.commands import cross_validate as _cross_validate
from src.cli.commands import distill as _distill
from src.cli.commands import eval as _eval
from src.cli.commands import export_onnx as _export_onnx
from src.cli.commands import federated as _federated
from src.cli.commands import gradcam as _gradcam
from src.cli.commands import kaggle as _kaggle
from src.cli.commands import label_noise as _label_noise
from src.cli.commands import ood as _ood
from src.cli.commands import paper_figures as _paper_figures
from src.cli.commands import pr_ece as _pr_ece
from src.cli.commands import quantize as _quantize
from src.cli.commands import train as _train

app = typer.Typer(
    name="wafer-cli",
    help="Unified CLI for the WM-811K wafer defect detection project.",
    no_args_is_help=True,
    add_completion=False,
)


# Each subcommand is a pass-through wrapper: it accepts arbitrary extra
# args (``allow_extra_args=True, ignore_unknown_options=True``) and
# forwards them to the underlying script's ``main()``. This preserves
# 1:1 flag parity with the existing ``scripts/*.py`` CLIs.
_SUBCOMMANDS = [
    ("train", _train.train, "Train a wafer defect detection model."),
    ("eval", _eval.eval_cmd, "Evaluate a trained model (MC-Dropout + calibration audit)."),
    ("benchmark", _benchmark.benchmark, "Benchmark latency, throughput, and memory of a model."),
    ("distill", _distill.distill, "Knowledge-distillation training of a student from teacher(s)."),
    ("export-onnx", _export_onnx.export_onnx, "Export a trained checkpoint to ONNX."),
    ("quantize", _quantize.quantize, "Post-training quantization (dynamic / static / QAT)."),
    ("ood", _ood.ood, "Out-of-distribution / anomaly analysis."),
    ("calibrate", _calibrate.calibrate, "Fit temperature scaling and report calibration metrics."),
    ("bootstrap", _bootstrap.bootstrap, "Bootstrap confidence intervals for evaluation metrics."),
    ("gradcam", _gradcam.gradcam, "Generate Grad-CAM visualizations for misclassifications."),
    ("paper-figures", _paper_figures.paper_figures, "Regenerate the figures used in the paper."),
    ("cross-validate", _cross_validate.cross_validate, "K-fold cross-validation study."),
    ("federated", _federated.federated, "Federated-learning demo (FedAvg on WM-811K clients)."),
    ("active-learn", _active_learn.active_learn, "Active-learning acquisition-function study."),
    ("label-noise", _label_noise.label_noise, "Label-noise robustness study."),
    ("pr-ece", _pr_ece.pr_ece, "Per-class PR curves + Expected Calibration Error."),
]

for _name, _fn, _help in _SUBCOMMANDS:
    app.command(
        name=_name,
        help=_help,
        context_settings=FORWARD_CONTEXT_SETTINGS,
    )(_fn)


# ``kaggle`` is a nested Typer sub-app (commands: push, delete) rather
# than a single forwarding command, so it's registered via ``add_typer``.
app.add_typer(
    _kaggle.kaggle_app,
    name="kaggle",
    help="Publish or manage the Kaggle notebook kernel for this project.",
)


def run() -> None:  # pragma: no cover - thin delegate
    """Console-script entry point used by ``[project.scripts]``."""
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
