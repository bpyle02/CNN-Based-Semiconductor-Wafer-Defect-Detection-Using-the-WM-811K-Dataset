"""``wafer-cli kaggle`` — publish / delete the Kaggle notebook kernel.

This subcommand owns the logic previously in ``scripts/kaggle_push.py``.
Two nested commands are exposed:

- ``wafer-cli kaggle push``   — publish (or update) the notebook and
  kernel-metadata.json as a Kaggle Kernel.
- ``wafer-cli kaggle delete`` — delete the kernel (frees a session slot
  before republishing).

KGAT-token workaround
---------------------
Kaggle's newer ``KGAT_``-prefixed API tokens are **Bearer** tokens. The
Kaggle CLI / SDK default to Basic auth (username:password) which the
kernel endpoints reject with 401 for KGAT_ tokens. Setting
``KAGGLE_API_TOKEN`` *before* importing the SDK flips it onto the
Bearer-auth code path internally. We therefore set the env-var first
and only then import ``kaggle.api``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from src.cli._common import REPO_ROOT

kaggle_app = typer.Typer(
    name="kaggle",
    help="Publish or manage the Kaggle notebook kernel for this project.",
    no_args_is_help=True,
    add_completion=False,
)


def _load_credentials_and_set_env() -> Optional[Path]:
    """Locate ``~/.kaggle/kaggle.json`` and set KAGGLE_API_TOKEN.

    Returns the credentials path on success, or ``None`` if the file is
    missing (caller should report and exit non-zero).
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        typer.echo(
            f"error: {kaggle_json} not found. Create an API token at "
            "https://www.kaggle.com/settings/account and save it there.",
            err=True,
        )
        return None

    creds = json.loads(kaggle_json.read_text())
    # IMPORTANT: must be set BEFORE `from kaggle import api` anywhere in
    # this process — the SDK snapshots it at init time.
    os.environ["KAGGLE_API_TOKEN"] = creds["key"]
    return kaggle_json


def _kernel_slug_from_metadata(metadata_path: Path) -> Optional[str]:
    """Extract the ``owner/slug`` id from kernel-metadata.json."""
    if not metadata_path.exists():
        typer.echo(f"error: {metadata_path} not found.", err=True)
        return None
    try:
        meta = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        typer.echo(f"error: could not parse {metadata_path}: {exc}", err=True)
        return None
    slug = meta.get("id")
    if not slug:
        typer.echo(
            f"error: {metadata_path} has no 'id' field (expected 'owner/slug').",
            err=True,
        )
        return None
    return str(slug)


@kaggle_app.command("push")
def push() -> None:
    """Publish (or update) ``docs/kaggle_quickstart.ipynb`` as a Kaggle Kernel.

    Reads ``kernel-metadata.json`` at the repo root and calls Kaggle's
    ``kernels push`` under the hood. Re-run after any change to the
    notebook, metadata, or source the notebook imports — the URL stays
    stable so the README badge keeps working.
    """
    if _load_credentials_and_set_env() is None:
        raise typer.Exit(code=1)

    # Import AFTER setting the env var.
    from kaggle import api  # type: ignore[import-not-found]

    metadata_path = REPO_ROOT / "kernel-metadata.json"
    if not metadata_path.exists():
        typer.echo(f"error: {metadata_path} not found.", err=True)
        raise typer.Exit(code=1)

    resp = api.kernels_push(str(REPO_ROOT))
    error = getattr(resp, "error", None) or getattr(resp, "error_nullable", None)
    if error:
        typer.echo(f"push failed: {error}", err=True)
        raise typer.Exit(code=1)

    ref = getattr(resp, "ref", "?")
    url = getattr(resp, "url", None)
    version = getattr(resp, "version_number", None)
    typer.echo(f"OK — pushed version {version}")
    typer.echo(f"  ref: {ref}")
    typer.echo(f"  url: {url}")


@kaggle_app.command("delete")
def delete(
    slug: Optional[str] = typer.Option(
        None,
        "--slug",
        help=(
            "Kernel id as 'owner/slug'. Defaults to the 'id' field of "
            "kernel-metadata.json at the repo root."
        ),
    ),
) -> None:
    """Delete the Kaggle kernel to free up a session slot before republishing.

    Wraps ``kaggle.api.kernels_delete()``. Uses the slug from
    ``kernel-metadata.json`` unless ``--slug owner/slug`` is given.
    """
    if _load_credentials_and_set_env() is None:
        raise typer.Exit(code=1)

    if slug is None:
        slug = _kernel_slug_from_metadata(REPO_ROOT / "kernel-metadata.json")
        if slug is None:
            raise typer.Exit(code=1)

    # Import AFTER setting the env var.
    from kaggle import api  # type: ignore[import-not-found]

    # ``kernels_delete`` is not exposed by every SDK version; fall back
    # to the lower-level API call when the high-level wrapper is
    # missing.
    delete_fn = getattr(api, "kernels_delete", None)
    if delete_fn is None:
        delete_fn = getattr(api, "kernel_delete", None)
    if delete_fn is None:
        typer.echo(
            "error: installed kaggle SDK has no kernels_delete(); "
            "upgrade with `pip install -U kaggle`.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        owner, kernel = slug.split("/", 1)
    except ValueError:
        typer.echo(
            f"error: --slug must be 'owner/slug' (got {slug!r}).",
            err=True,
        )
        raise typer.Exit(code=1) from None

    try:
        resp = delete_fn(owner, kernel)
    except TypeError:
        # Some SDK versions accept the full slug as a single argument.
        resp = delete_fn(slug)

    error = getattr(resp, "error", None) or getattr(resp, "error_nullable", None)
    if error:
        typer.echo(f"delete failed: {error}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"OK — deleted kernel {slug}")


# Back-compat alias for the legacy script shim.
def main() -> int:
    """Entry point used by ``scripts/kaggle_push.py`` shim.

    Mirrors the legacy script behaviour: always runs ``push``.
    """
    try:
        push()
    except typer.Exit as exc:
        return int(exc.exit_code or 0)
    return 0


if __name__ == "__main__":  # pragma: no cover
    kaggle_app()
