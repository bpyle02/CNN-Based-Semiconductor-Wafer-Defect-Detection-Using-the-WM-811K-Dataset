#!/usr/bin/env python3
"""
Parse all Python files and generate a Graphviz DOT dependency graph.

Scans src/ and root-level scripts, extracts internal import relationships
using the ast module, and produces a DOT file with clustered subgraphs
per package.

Usage:
    python scripts/generate_dot_deps.py
    # Output: docs/generated/dependency_graph.dot
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "generated"
OUTPUT_DOT = OUTPUT_DIR / "dependency_graph.dot"

# Packages inside src/ that get their own cluster
SRC_PACKAGES = [
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

# Colors per package cluster (pastel-ish for readability)
CLUSTER_COLORS = {
    "data": "#E8F5E9",  # green
    "models": "#E3F2FD",  # blue
    "training": "#FFF3E0",  # orange
    "analysis": "#F3E5F5",  # purple
    "inference": "#E0F7FA",  # cyan
    "augmentation": "#FFF9C4",  # yellow
    "detection": "#FFEBEE",  # red
    "federated": "#F1F8E9",  # lime
    "mlops": "#EFEBE9",  # brown
}

CLUSTER_BORDER_COLORS = {
    "data": "#4CAF50",
    "models": "#2196F3",
    "training": "#FF9800",
    "analysis": "#9C27B0",
    "inference": "#00BCD4",
    "augmentation": "#FFC107",
    "detection": "#F44336",
    "federated": "#8BC34A",
    "mlops": "#795548",
}


def discover_python_files() -> Dict[str, List[Path]]:
    """Find all Python files grouped by category.

    Returns a dict with keys:
      - 'root': root-level scripts like train.py
      - 'src_root': src/*.py (non-package modules)
      - each package name: src/<pkg>/*.py
      - 'scripts': scripts/*.py
    """
    files: Dict[str, List[Path]] = defaultdict(list)

    # Root-level .py files (only train.py, setup.py, etc.)
    for py in PROJECT_ROOT.glob("*.py"):
        files["root"].append(py)

    # src/ root-level modules (config.py, exceptions.py, model_registry.py, __init__.py)
    for py in SRC_DIR.glob("*.py"):
        files["src_root"].append(py)

    # src/<package>/*.py
    for pkg in SRC_PACKAGES:
        pkg_dir = SRC_DIR / pkg
        if pkg_dir.is_dir():
            for py in pkg_dir.glob("*.py"):
                files[pkg].append(py)

    # scripts/*.py
    for py in SCRIPTS_DIR.glob("*.py"):
        # skip self
        if py.name == "generate_dot_deps.py":
            continue
        files["scripts"].append(py)

    return files


def file_to_module(filepath: Path) -> str:
    """Convert a file path to a dotted module name.

    Examples:
        src/models/cnn.py       -> src.models.cnn
        src/config.py           -> src.config
        train.py                -> train
        scripts/dashboard.py    -> scripts.dashboard
    """
    try:
        rel = filepath.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return filepath.stem

    parts = list(rel.with_suffix("").parts)
    # Drop __init__ — represent the package itself
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def extract_imports(filepath: Path) -> Set[str]:
    """Parse a Python file with ast and return all imported module names."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def resolve_relative_import(filepath: Path, module_str: str, level: int) -> str:
    """Resolve a relative import to an absolute module path."""
    # For relative imports parsed by ast, node.module may already be the
    # relative part. We need the package context.
    pkg_dir = filepath.parent
    for _ in range(level - 1):
        pkg_dir = pkg_dir.parent
    try:
        rel = pkg_dir.resolve().relative_to(PROJECT_ROOT)
        base = ".".join(rel.parts)
    except ValueError:
        base = ""
    if module_str:
        return f"{base}.{module_str}" if base else module_str
    return base


def extract_imports_detailed(filepath: Path) -> Set[str]:
    """Parse a file and return fully-qualified imported module names,
    resolving relative imports."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import
                resolved = resolve_relative_import(filepath, node.module or "", node.level)
                imports.add(resolved)
            elif node.module:
                imports.add(node.module)
    return imports


def normalize_to_known_module(raw_import: str, known_modules: Set[str]) -> str | None:
    """Given a raw import string, find the best matching known module.

    Strategy: try the full string, then progressively strip the last
    component (since 'from src.models.cnn import WaferCNN' gives us
    'src.models.cnn' which is a known module, but 'from src.models import X'
    gives 'src.models' which maps to src/models/__init__.py).
    """
    candidate = raw_import
    while candidate:
        if candidate in known_modules:
            return candidate
        # Strip last component
        if "." in candidate:
            candidate = candidate.rsplit(".", 1)[0]
        else:
            break
    return None


def build_dependency_graph(
    file_groups: Dict[str, List[Path]],
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """Build the full dependency graph.

    Returns:
        modules: dict mapping module_name -> group_key
        edges: list of (source_module, target_module) tuples
    """
    # Build registry of all known modules
    modules: Dict[str, str] = {}  # module_name -> group
    module_files: Dict[str, Path] = {}  # module_name -> filepath

    for group, paths in file_groups.items():
        for fp in paths:
            mod = file_to_module(fp)
            if mod:
                modules[mod] = group
                module_files[mod] = fp

    known = set(modules.keys())
    edges: List[Tuple[str, str]] = []
    seen_edges: Set[Tuple[str, str]] = set()

    for mod_name, filepath in module_files.items():
        raw_imports = extract_imports_detailed(filepath)
        for raw in raw_imports:
            target = normalize_to_known_module(raw, known)
            if target and target != mod_name:
                edge = (mod_name, target)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(edge)

    return modules, edges


def module_short_name(mod: str) -> str:
    """Shorten module name for display.

    src.models.cnn -> models.cnn
    src.config     -> src.config
    train          -> train
    scripts.dashboard -> scripts.dashboard
    """
    return mod


def dot_node_id(mod: str) -> str:
    """Convert a module name to a valid DOT node ID."""
    return mod.replace(".", "_")


def generate_dot(
    modules: Dict[str, str],
    edges: List[Tuple[str, str]],
) -> str:
    """Generate the Graphviz DOT source."""

    lines = []
    lines.append("digraph DependencyGraph {")
    lines.append("    // Graph settings")
    lines.append("    rankdir=LR;")
    lines.append('    fontname="Times";')
    lines.append("    fontsize=14;")
    lines.append('    label="\\nCNN Wafer Defect Detection - Internal Dependency Graph\\n";')
    lines.append("    labelloc=t;")
    lines.append("    compound=true;")
    lines.append("    nodesep=0.4;")
    lines.append("    ranksep=1.0;")
    lines.append("")
    lines.append("    // Default node style")
    lines.append('    node [shape=box, style="filled,rounded", fontname="Times", fontsize=10];')
    lines.append('    edge [color="#555555", arrowsize=0.7];')
    lines.append("")

    # Group modules by their group key
    groups: Dict[str, List[str]] = defaultdict(list)
    for mod, group in modules.items():
        groups[group].append(mod)

    # Sort modules within each group
    for g in groups:
        groups[g].sort()

    # Emit clustered subgraphs for src packages
    for pkg in SRC_PACKAGES:
        pkg_mods = [m for m in groups.get(pkg, []) if m != f"src.{pkg}"]
        init_mod = f"src.{pkg}"
        all_mods = []
        if init_mod in modules:
            all_mods.append(init_mod)
        all_mods.extend(pkg_mods)

        if not all_mods:
            continue

        bg = CLUSTER_COLORS.get(pkg, "#F5F5F5")
        border = CLUSTER_BORDER_COLORS.get(pkg, "#999999")

        lines.append(f"    subgraph cluster_{pkg} {{")
        lines.append(f'        label="{pkg}";')
        lines.append(f'        style="filled,rounded";')
        lines.append(f'        fillcolor="{bg}";')
        lines.append(f'        color="{border}";')
        lines.append(f'        fontname="Times";')
        lines.append(f"        fontsize=12;")
        lines.append("")
        for mod in all_mods:
            nid = dot_node_id(mod)
            display = mod.split(".")[-1]
            if display == "__init__":
                display = f"{pkg} (init)"
            lines.append(f'        {nid} [label="{display}", fillcolor="white"];')
        lines.append("    }")
        lines.append("")

    # src root modules (config.py, exceptions.py, model_registry.py, __init__.py)
    src_root_mods = groups.get("src_root", [])
    if src_root_mods:
        lines.append("    subgraph cluster_src_root {")
        lines.append('        label="src (root)";')
        lines.append('        style="filled,rounded";')
        lines.append('        fillcolor="#ECEFF1";')
        lines.append('        color="#607D8B";')
        lines.append('        fontname="Times";')
        lines.append("        fontsize=12;")
        lines.append("")
        for mod in src_root_mods:
            nid = dot_node_id(mod)
            display = mod.split(".")[-1]
            if display == "src":
                display = "src (init)"
            lines.append(f'        {nid} [label="{display}", fillcolor="white"];')
        lines.append("    }")
        lines.append("")

    # Root-level scripts (train.py, setup.py)
    root_mods = groups.get("root", [])
    if root_mods:
        lines.append("    // Root-level scripts")
        for mod in root_mods:
            nid = dot_node_id(mod)
            lines.append(
                f'    {nid} [label="{mod}.py", fillcolor="#BBDEFB", '
                f'style="filled,bold", shape=box];'
            )
        lines.append("")

    # Scripts
    script_mods = groups.get("scripts", [])
    if script_mods:
        lines.append("    subgraph cluster_scripts {")
        lines.append('        label="scripts/";')
        lines.append('        style="filled,rounded";')
        lines.append('        fillcolor="#F5F5F5";')
        lines.append('        color="#9E9E9E";')
        lines.append('        fontname="Times";')
        lines.append("        fontsize=12;")
        lines.append("")
        for mod in script_mods:
            nid = dot_node_id(mod)
            display = mod.split(".")[-1]
            lines.append(f'        {nid} [label="{display}", fillcolor="#E1BEE7"];')
        lines.append("    }")
        lines.append("")

    # Edges
    lines.append("    // Dependencies")
    for src, tgt in sorted(edges):
        src_id = dot_node_id(src)
        tgt_id = dot_node_id(tgt)
        lines.append(f"    {src_id} -> {tgt_id};")

    # Legend
    lines.append("")
    lines.append("    // Legend")
    lines.append("    subgraph cluster_legend {")
    lines.append('        label="Legend";')
    lines.append('        style="filled,rounded";')
    lines.append('        fillcolor="white";')
    lines.append('        color="#BDBDBD";')
    lines.append('        fontname="Helvetica Bold";')
    lines.append("        fontsize=11;")
    lines.append("        node [shape=box, width=1.2, fontsize=9];")

    legend_items = [
        ("leg_root", "Root Script", "#BBDEFB"),
        ("leg_script", "Script", "#E1BEE7"),
        ("leg_src", "src module", "white"),
    ]
    for pkg in SRC_PACKAGES:
        legend_items.append((f"leg_{pkg}", pkg, CLUSTER_COLORS[pkg]))

    for nid, label, color in legend_items:
        lines.append(f'        {nid} [label="{label}", fillcolor="{color}"];')

    # invisible edges to keep legend vertical
    for i in range(len(legend_items) - 1):
        lines.append(f"        {legend_items[i][0]} -> {legend_items[i+1][0]} " f"[style=invis];")

    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning Python files in {PROJECT_ROOT} ...")
    file_groups = discover_python_files()

    total = sum(len(v) for v in file_groups.values())
    print(f"Found {total} Python files across {len(file_groups)} groups")
    for group, paths in sorted(file_groups.items()):
        print(f"  {group}: {len(paths)} files")

    print("\nBuilding dependency graph ...")
    modules, edges = build_dependency_graph(file_groups)
    print(f"  {len(modules)} modules, {len(edges)} internal dependency edges")

    print("\nGenerating DOT ...")
    dot_source = generate_dot(modules, edges)

    OUTPUT_DOT.write_text(dot_source, encoding="utf-8")
    print(f"Written: {OUTPUT_DOT}")
    print(f"  ({len(dot_source)} bytes, {dot_source.count(chr(10))+1} lines)")


if __name__ == "__main__":
    main()
