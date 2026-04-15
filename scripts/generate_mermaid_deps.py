#!/usr/bin/env python3
"""Generate a Mermaid flowchart of internal module dependencies.

Parses all Python files in src/ and root-level scripts, extracts internal
import statements, and outputs a Mermaid diagram showing the dependency graph
grouped by subpackage.

Usage:
    python scripts/generate_mermaid_deps.py
    python scripts/generate_mermaid_deps.py --output docs/generated/dependency_graph.md
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Packages that get subgraphs
SUBPACKAGES = [
    "data",
    "models",
    "training",
    "analysis",
    "inference",
    "augmentation",
    "detection",
    "federated",
    "mlops",
]


def resolve_module_name(file_path: Path) -> Optional[str]:
    """Convert a file path to a dotted module name relative to the project root.

    Returns short names like 'data.dataset', 'models.cnn', 'config', 'train'.
    Returns None for __init__.py files (they represent the package itself).
    """
    try:
        rel = file_path.relative_to(SRC)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            # Package init -- use package name
            if len(parts) == 1:
                return None  # src/__init__.py itself, skip
            return ".".join(parts[:-1])
        return ".".join(parts)
    except ValueError:
        pass

    # Root-level file (train.py, setup.py, etc.)
    try:
        rel = file_path.relative_to(ROOT)
        parts = list(rel.with_suffix("").parts)
        if parts[0] == "scripts":
            return "scripts." + parts[-1]
        return parts[-1]
    except ValueError:
        return None


def normalize_import(module_str: str) -> Optional[str]:
    """Normalize an import string to a short internal module name.

    'src.models.cnn' -> 'models.cnn'
    'src.config'     -> 'config'
    'src.exceptions' -> 'exceptions'

    Returns None for external imports.
    """
    if module_str.startswith("src."):
        short = module_str[4:]  # strip 'src.'
        return short
    return None


def extract_imports(file_path: Path) -> list[str]:
    """Parse a Python file and return all internal import targets as short names."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                norm = normalize_import(alias.name)
                if norm:
                    imports.append(norm)

        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            module = node.module

            # Handle relative imports: resolve them to absolute
            if node.level > 0:
                try:
                    rel = file_path.relative_to(SRC)
                    pkg_parts = list(rel.parent.parts)
                    # level=1 means current package, level=2 means parent, etc.
                    base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                    if module:
                        full = ".".join(base_parts + [module])
                    else:
                        full = ".".join(base_parts)
                    imports.append(full)
                    continue
                except ValueError:
                    continue

            norm = normalize_import(module)
            if norm:
                imports.append(norm)

    return imports


def get_package(module_name: str) -> str:
    """Determine which subpackage a module belongs to.

    'models.cnn' -> 'models'
    'config'     -> 'root'
    'scripts.train' -> 'scripts'
    """
    parts = module_name.split(".")
    if parts[0] in SUBPACKAGES:
        return parts[0]
    if parts[0] == "scripts":
        return "scripts"
    return "root"


def make_node_id(module_name: str) -> str:
    """Create a valid Mermaid node ID from a module name.

    Prefixed with 'mod_' to avoid collisions with Mermaid subgraph IDs
    (e.g., subgraph 'data' vs node 'data').
    """
    return "mod_" + module_name.replace(".", "_")


def make_node_label(module_name: str) -> str:
    """Create a readable label for a node."""
    return module_name


def collect_all_files() -> list[Path]:
    """Collect all Python files to analyze."""
    files = []

    # Root-level entry points
    for f in ROOT.glob("*.py"):
        if f.name not in ("setup.py",):
            files.append(f)

    # All src/ Python files
    for f in SRC.rglob("*.py"):
        files.append(f)

    # scripts/
    scripts_dir = ROOT / "scripts"
    if scripts_dir.exists():
        for f in scripts_dir.glob("*.py"):
            if f.name != "generate_mermaid_deps.py":
                files.append(f)

    return sorted(set(files))


def build_dependency_graph() -> tuple[dict[str, set[str]], set[str]]:
    """Build the dependency graph.

    Returns:
        edges: dict mapping source_module -> set of target_modules
        all_modules: set of all module names that appear
    """
    edges: dict[str, set[str]] = defaultdict(set)
    all_modules: set[str] = set()

    for fpath in collect_all_files():
        source = resolve_module_name(fpath)
        if source is None:
            continue

        all_modules.add(source)
        targets = extract_imports(fpath)

        for target in targets:
            # Collapse to the most specific module that exists
            # e.g., if importing from 'models.cnn', keep that
            all_modules.add(target)
            if target != source:
                edges[source].add(target)

    return dict(edges), all_modules


def generate_mermaid(edges: dict[str, set[str]], all_modules: set[str]) -> str:
    """Generate a Mermaid flowchart string."""

    lines = ["flowchart LR"]

    # Group modules by package
    packages: dict[str, list[str]] = defaultdict(list)
    for mod in sorted(all_modules):
        pkg = get_package(mod)
        packages[pkg].append(mod)

    # Style definitions
    lines.append("")
    lines.append("    %% Style definitions")
    lines.append("    classDef entrypoint fill:#e1f5fe,stroke:#01579b,stroke-width:2px")
    lines.append("    classDef datamod fill:#e8f5e9,stroke:#2e7d32")
    lines.append("    classDef modelmod fill:#fff3e0,stroke:#e65100")
    lines.append("    classDef trainmod fill:#fce4ec,stroke:#b71c1c")
    lines.append("    classDef analysismod fill:#f3e5f5,stroke:#4a148c")
    lines.append("    classDef inframod fill:#e0f2f1,stroke:#004d40")
    lines.append("    classDef scriptmod fill:#f5f5f5,stroke:#616161,stroke-dasharray:5 5")

    # Package display names and style mapping
    pkg_meta = {
        "root": ("Root Modules", "entrypoint"),
        "data": ("Data", "datamod"),
        "models": ("Models", "modelmod"),
        "training": ("Training", "trainmod"),
        "analysis": ("Analysis", "analysismod"),
        "inference": ("Inference", "inframod"),
        "augmentation": ("Augmentation", "inframod"),
        "detection": ("Detection", "inframod"),
        "federated": ("Federated", "inframod"),
        "mlops": ("MLOps", "inframod"),
        "scripts": ("Scripts", "scriptmod"),
    }

    # Emit subgraphs
    for pkg_key in ["root", *SUBPACKAGES, "scripts"]:
        if pkg_key not in packages:
            continue
        mods = packages[pkg_key]
        display_name, style_class = pkg_meta.get(pkg_key, (pkg_key, "inframod"))
        sg_id = f"sg_{pkg_key}"  # prefix to avoid collision with node IDs

        lines.append("")
        lines.append(f"    subgraph {sg_id}[{display_name}]")
        for mod in sorted(mods):
            nid = make_node_id(mod)
            label = make_node_label(mod)
            lines.append(f'        {nid}["{label}"]')
        lines.append("    end")

        # Apply styles
        for mod in sorted(mods):
            nid = make_node_id(mod)
            lines.append(f"    class {nid} {style_class}")

    # Emit edges
    lines.append("")
    lines.append("    %% Dependencies")
    seen_edges: set[tuple[str, str]] = set()
    for source in sorted(edges.keys()):
        for target in sorted(edges[source]):
            src_id = make_node_id(source)
            tgt_id = make_node_id(target)
            edge_key = (src_id, tgt_id)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                # Use dotted arrows for cross-package, solid for within-package
                src_pkg = get_package(source)
                tgt_pkg = get_package(target)
                if src_pkg == tgt_pkg:
                    lines.append(f"    {src_id} --> {tgt_id}")
                else:
                    lines.append(f"    {src_id} -.-> {tgt_id}")

    lines.append("")
    return "\n".join(lines)


def generate_markdown(mermaid_text: str) -> str:
    """Wrap the mermaid diagram in a markdown document."""
    return (
        "# Module Dependency Graph\n"
        "\n"
        "Internal import dependencies across all project modules.\n"
        "Solid arrows indicate within-package imports; "
        "dashed arrows indicate cross-package imports.\n"
        "\n"
        "```mermaid\n"
        f"{mermaid_text}\n"
        "```\n"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Mermaid dependency diagram for the project."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write markdown output to this file (default: stdout)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw Mermaid text without markdown wrapping",
    )
    args = parser.parse_args()

    edges, all_modules = build_dependency_graph()
    mermaid_text = generate_mermaid(edges, all_modules)

    if args.raw:
        output = mermaid_text
    else:
        output = generate_markdown(mermaid_text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote dependency graph to {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
