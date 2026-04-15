#!/usr/bin/env python3
"""Generate a module-level dependency graph for the src package using static AST analysis.

Produces DOT and SVG files without importing any project modules (avoids segfaults
from heavy C-extension libraries like PyTorch).

Includes a pure-Python SVG renderer as fallback when graphviz ``dot`` binary is
unavailable or broken.

Usage:
    python scripts/generate_depgraph.py

Outputs:
    docs/generated/pydeps_dependency_graph.dot   (always)
    docs/generated/pydeps_dependency_graph.svg   (always -- pure-Python fallback)
"""

import ast
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "generated"


def get_module_name(filepath: Path) -> str:
    """Convert file path to dotted module name relative to project root."""
    rel = filepath.relative_to(PROJECT_ROOT)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def extract_imports(filepath: Path) -> list[str]:
    """Extract all import targets from a Python file using AST (no execution)."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def resolve_to_src_module(import_name: str, all_modules: set[str]) -> str | None:
    """Resolve an import name to a known src module, or None if external."""
    # Direct match
    if import_name in all_modules:
        return import_name
    # Check if it's a sub-attribute of a known module
    parts = import_name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in all_modules:
            return candidate
    return None


# Subpackage color mapping
SUBPACKAGE_COLORS = {
    "src.data": "#4E79A7",
    "src.models": "#F28E2B",
    "src.training": "#E15759",
    "src.analysis": "#76B7B2",
    "src.inference": "#59A14F",
    "src.augmentation": "#EDC948",
    "src.detection": "#B07AA1",
    "src.federated": "#FF9DA7",
    "src.mlops": "#9C755F",
    "src": "#BAB0AC",
}


def get_color(module: str) -> str:
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        key = ".".join(parts[:i])
        if key in SUBPACKAGE_COLORS:
            return SUBPACKAGE_COLORS[key]
    return "#BAB0AC"


def get_subpackage(module: str) -> str:
    parts = module.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return module


def build_dependency_graph() -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Build the internal dependency graph from static analysis."""
    # Collect all src modules
    all_modules: set[str] = set()
    file_to_module: dict[Path, str] = {}

    for py_file in sorted(SRC_DIR.rglob("*.py")):
        mod_name = get_module_name(py_file)
        all_modules.add(mod_name)
        file_to_module[py_file] = mod_name

    # Build edges: module -> set of internal dependencies
    internal_deps: dict[str, set[str]] = defaultdict(set)
    # Also track external deps per module
    external_deps: dict[str, set[str]] = defaultdict(set)

    for py_file, mod_name in file_to_module.items():
        raw_imports = extract_imports(py_file)
        for imp in raw_imports:
            resolved = resolve_to_src_module(imp, all_modules)
            if resolved and resolved != mod_name:
                internal_deps[mod_name].add(resolved)
            elif not resolved:
                # Track top-level external package
                external_deps[mod_name].add(imp.split(".")[0])

    return all_modules, dict(internal_deps), dict(external_deps)


def generate_dot(
    all_modules: set[str],
    internal_deps: dict[str, set[str]],
    external_deps: dict[str, set[str]],
    *,
    cluster: bool = True,
    include_external: bool = False,
) -> str:
    """Generate DOT graph source."""
    lines = [
        "digraph src_dependencies {",
        "    rankdir=LR;",
        '    fontname="Helvetica";',
        '    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];',
        '    edge [color="#666666", arrowsize=0.7];',
        "",
    ]

    # Filter to only modules that have edges (either as source or target)
    active_modules = set()
    for src, targets in internal_deps.items():
        active_modules.add(src)
        active_modules.update(targets)

    # Also include modules with no deps but that exist
    active_modules.update(all_modules)

    # Remove bare __init__ packages that just re-export (keep them if they have unique deps)
    init_only = set()
    for m in list(active_modules):
        parts = m.split(".")
        if len(parts) == 2 and m in all_modules:
            # This is a subpackage __init__ -- keep it
            pass

    if cluster:
        # Group by subpackage
        subpackages: dict[str, list[str]] = defaultdict(list)
        for m in sorted(active_modules):
            subpackages[get_subpackage(m)].append(m)

        for i, (subpkg, members) in enumerate(sorted(subpackages.items())):
            color = get_color(subpkg)
            lines.append(f"    subgraph cluster_{i} {{")
            lines.append(f'        label="{subpkg}";')
            lines.append(f'        style="rounded,filled";')
            lines.append(f'        fillcolor="{color}22";')
            lines.append(f'        color="{color}";')
            for m in sorted(members):
                short_label = m.split(".")[-1]
                node_id = m.replace(".", "_")
                lines.append(f'        {node_id} [label="{short_label}", fillcolor="{color}44"];')
            lines.append("    }")
            lines.append("")
    else:
        for m in sorted(active_modules):
            color = get_color(m)
            short_label = m.replace("src.", "")
            node_id = m.replace(".", "_")
            lines.append(f'    {node_id} [label="{short_label}", fillcolor="{color}44"];')
        lines.append("")

    # Add external nodes if requested
    if include_external:
        ext_all = set()
        for deps in external_deps.values():
            ext_all.update(deps)
        if ext_all:
            lines.append("    subgraph cluster_external {")
            lines.append('        label="External";')
            lines.append('        style="dashed,rounded";')
            lines.append('        color="#999999";')
            for ext in sorted(ext_all):
                node_id = f"ext_{ext}"
                lines.append(
                    f'        {node_id} [label="{ext}", fillcolor="#EEEEEE", shape=ellipse];'
                )
            lines.append("    }")
            lines.append("")

    # Edges
    for src, targets in sorted(internal_deps.items()):
        src_id = src.replace(".", "_")
        for tgt in sorted(targets):
            tgt_id = tgt.replace(".", "_")
            lines.append(f"    {src_id} -> {tgt_id};")

    if include_external:
        for src, deps in sorted(external_deps.items()):
            src_id = src.replace(".", "_")
            for ext in sorted(deps):
                ext_id = f"ext_{ext}"
                lines.append(f'    {src_id} -> {ext_id} [style=dashed, color="#CCCCCC"];')

    lines.append("}")
    return "\n".join(lines)


def generate_simplified_dot(
    all_modules: set[str],
    internal_deps: dict[str, set[str]],
) -> str:
    """Generate a simplified DOT graph at the subpackage level only."""
    # Aggregate to subpackage level
    pkg_deps: dict[str, set[str]] = defaultdict(set)

    for src, targets in internal_deps.items():
        src_pkg = get_subpackage(src)
        for tgt in targets:
            tgt_pkg = get_subpackage(tgt)
            if src_pkg != tgt_pkg:
                pkg_deps[src_pkg].add(tgt_pkg)

    all_pkgs = set()
    for m in all_modules:
        all_pkgs.add(get_subpackage(m))

    lines = [
        "digraph src_packages {",
        "    rankdir=LR;",
        '    fontname="Helvetica";',
        '    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=12, penwidth=2];',
        '    edge [color="#666666", arrowsize=0.8, penwidth=1.5];',
        "",
    ]

    for pkg in sorted(all_pkgs):
        color = get_color(pkg)
        label = pkg.replace("src.", "") if pkg != "src" else "src (root)"
        node_id = pkg.replace(".", "_")
        lines.append(f'    {node_id} [label="{label}", fillcolor="{color}88"];')

    lines.append("")

    for src_pkg, targets in sorted(pkg_deps.items()):
        src_id = src_pkg.replace(".", "_")
        for tgt_pkg in sorted(targets):
            tgt_id = tgt_pkg.replace(".", "_")
            lines.append(f"    {src_id} -> {tgt_id};")

    lines.append("}")
    return "\n".join(lines)


def dot_to_svg(dot_path: Path, svg_path: Path) -> bool:
    """Render DOT to SVG using graphviz. Returns True on success."""
    try:
        result = subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and svg_path.exists() and svg_path.stat().st_size > 0:
            return True
        print(f"  dot failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("  graphviz 'dot' binary not found.", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("  dot rendering timed out.", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"  dot error: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Pure-Python SVG renderer (fallback when graphviz dot is broken/missing)
# ---------------------------------------------------------------------------


def _estimate_text_width(text: str, font_size: float = 10) -> float:
    """Rough estimate of text width in pixels (monospace-ish approximation)."""
    return len(text) * font_size * 0.6


def render_svg_fallback(
    nodes: list[dict],
    edges: list[tuple[str, str]],
    clusters: dict[str, list[str]] | None = None,
    title: str = "src dependencies",
) -> str:
    """Render a dependency graph to SVG using pure Python.

    Args:
        nodes: list of dicts with keys: id, label, color
        edges: list of (source_id, target_id) tuples
        clusters: optional mapping of cluster_label -> [node_ids]
        title: graph title
    """
    node_map = {n["id"]: n for n in nodes}

    # Layout: arrange clusters in columns, nodes within cluster in a column
    if clusters:
        cluster_list = sorted(clusters.items())
    else:
        cluster_list = [("all", [n["id"] for n in nodes])]

    pad_x, pad_y = 40, 60
    node_h = 30
    node_spacing_y = 16
    cluster_spacing_x = 60
    cluster_pad = 20
    font_size = 11

    # Calculate node widths
    node_widths: dict[str, float] = {}
    for n in nodes:
        node_widths[n["id"]] = max(80, _estimate_text_width(n["label"], font_size) + 24)

    # Position nodes by cluster
    node_positions: dict[str, tuple[float, float]] = {}
    cluster_rects: list[tuple[str, float, float, float, float, str]] = []

    cur_x = pad_x
    max_y = 0

    for cluster_label, member_ids in cluster_list:
        members = [m for m in member_ids if m in node_map]
        if not members:
            continue

        max_node_w = max(node_widths.get(m, 80) for m in members)
        cluster_w = max_node_w + 2 * cluster_pad

        cluster_top = pad_y
        cur_y = cluster_top + cluster_pad + 22  # space for cluster label

        for mid in sorted(members):
            nw = node_widths.get(mid, 80)
            cx = cur_x + cluster_pad + max_node_w / 2
            cy = cur_y + node_h / 2
            node_positions[mid] = (cx, cy)
            cur_y += node_h + node_spacing_y

        cluster_bottom = cur_y + cluster_pad
        cluster_h = cluster_bottom - cluster_top

        color = node_map[members[0]]["color"] if members else "#BAB0AC"
        cluster_rects.append((cluster_label, cur_x, cluster_top, cluster_w, cluster_h, color))

        if cluster_bottom > max_y:
            max_y = cluster_bottom

        cur_x += cluster_w + cluster_spacing_x

    svg_w = cur_x + pad_x
    svg_h = max_y + pad_y

    # Build SVG
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" '
        f'width="{svg_w}" height="{svg_h}">',
        "<defs>",
        '  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">',
        '    <polygon points="0 0, 10 3.5, 0 7" fill="#666" />',
        "  </marker>",
        "</defs>",
        f'<rect width="{svg_w}" height="{svg_h}" fill="white" />',
        f'<text x="{svg_w/2}" y="24" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" '
        f'font-size="14" font-weight="bold" fill="#333">{title}</text>',
    ]

    # Draw cluster backgrounds
    for label, cx, cy, cw, ch, color in cluster_rects:
        svg_lines.append(
            f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="8" ry="8" '
            f'fill="{color}22" stroke="{color}" stroke-width="1.5" />'
        )
        svg_lines.append(
            f'<text x="{cx + cw/2}" y="{cy + 16}" text-anchor="middle" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="10" fill="{color}" '
            f'font-weight="bold">{label}</text>'
        )

    # Draw edges (behind nodes)
    for src_id, tgt_id in edges:
        if src_id not in node_positions or tgt_id not in node_positions:
            continue
        x1, y1 = node_positions[src_id]
        x2, y2 = node_positions[tgt_id]

        # Shorten line to stop at node border
        nw_src = node_widths.get(src_id, 80)
        nw_tgt = node_widths.get(tgt_id, 80)

        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1:
            continue

        # Determine exit/entry points based on relative positions
        # Simple approach: use right/left sides for horizontal, top/bottom for vertical
        if abs(dx) > abs(dy):
            # Horizontal-dominant: exit from side
            if dx > 0:
                sx = x1 + nw_src / 2
                ex = x2 - nw_tgt / 2
            else:
                sx = x1 - nw_src / 2
                ex = x2 + nw_tgt / 2
            sy = y1
            ey = y2
        else:
            # Vertical-dominant: exit from top/bottom
            sx = x1
            ex = x2
            if dy > 0:
                sy = y1 + node_h / 2
                ey = y2 - node_h / 2
            else:
                sy = y1 - node_h / 2
                ey = y2 + node_h / 2

        # Bezier curve for cleaner look
        ctrl_x = (sx + ex) / 2
        ctrl_y1 = sy
        ctrl_y2 = ey
        svg_lines.append(
            f'<path d="M {sx:.1f} {sy:.1f} C {ctrl_x:.1f} {ctrl_y1:.1f}, '
            f'{ctrl_x:.1f} {ctrl_y2:.1f}, {ex:.1f} {ey:.1f}" '
            f'fill="none" stroke="#888" stroke-width="1.2" marker-end="url(#arrowhead)" />'
        )

    # Draw nodes
    for n in nodes:
        nid = n["id"]
        if nid not in node_positions:
            continue
        cx, cy = node_positions[nid]
        nw = node_widths[nid]
        color = n["color"]

        rx = nw / 2
        ry = node_h / 2
        svg_lines.append(
            f'<rect x="{cx - rx}" y="{cy - ry}" width="{nw}" height="{node_h}" '
            f'rx="5" ry="5" fill="{color}66" stroke="{color}" stroke-width="1.2" />'
        )
        svg_lines.append(
            f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="{font_size}" '
            f'fill="#333">{n["label"]}</text>'
        )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def render_graph_as_svg(
    all_modules: set[str],
    internal_deps: dict[str, set[str]],
    svg_path: Path,
    *,
    simplified: bool = False,
) -> None:
    """Build node/edge data and render to SVG via pure Python."""
    if simplified:
        # Subpackage-level graph
        pkg_deps: dict[str, set[str]] = defaultdict(set)
        for src, targets in internal_deps.items():
            src_pkg = get_subpackage(src)
            for tgt in targets:
                tgt_pkg = get_subpackage(tgt)
                if src_pkg != tgt_pkg:
                    pkg_deps[src_pkg].add(tgt_pkg)

        all_pkgs = set()
        for m in all_modules:
            all_pkgs.add(get_subpackage(m))

        nodes = []
        for pkg in sorted(all_pkgs):
            color = get_color(pkg)
            label = pkg.replace("src.", "") if pkg != "src" else "src (root)"
            nodes.append({"id": pkg, "label": label, "color": color})

        edges = []
        for src_pkg, targets in pkg_deps.items():
            for tgt_pkg in targets:
                edges.append((src_pkg, tgt_pkg))

        svg_content = render_svg_fallback(nodes, edges, title="src subpackage dependencies")
    else:
        # Full module-level graph with clusters
        subpackages: dict[str, list[str]] = defaultdict(list)
        nodes = []
        for m in sorted(all_modules):
            color = get_color(m)
            short_label = m.split(".")[-1]
            nodes.append({"id": m, "label": short_label, "color": color})
            subpackages[get_subpackage(m)].append(m)

        edges = []
        for src, targets in internal_deps.items():
            for tgt in targets:
                edges.append((src, tgt))

        clusters = dict(sorted(subpackages.items()))
        svg_content = render_svg_fallback(
            nodes, edges, clusters=clusters, title="src module dependencies"
        )

    svg_path.write_text(svg_content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Analyzing src/ package imports (static AST analysis)...")
    all_modules, internal_deps, external_deps = build_dependency_graph()

    print(f"  Found {len(all_modules)} modules")
    edge_count = sum(len(v) for v in internal_deps.values())
    print(f"  Found {edge_count} internal dependency edges")
    ext_count = sum(len(v) for v in external_deps.values())
    print(f"  Found {ext_count} external dependency references")

    # --- Generate DOT files (always) ---

    # 1. Full detailed graph (clustered)
    dot_full = generate_dot(all_modules, internal_deps, external_deps, cluster=True)
    dot_full_path = OUTPUT_DIR / "pydeps_dependency_graph.dot"
    dot_full_path.write_text(dot_full, encoding="utf-8")
    print(f"\n  DOT: {dot_full_path}")

    # 2. Simplified subpackage-level graph
    dot_simple = generate_simplified_dot(all_modules, internal_deps)
    dot_simple_path = OUTPUT_DIR / "pydeps_dependency_graph_simplified.dot"
    dot_simple_path.write_text(dot_simple, encoding="utf-8")
    print(f"  DOT: {dot_simple_path}")

    # 3. Full graph with external deps
    dot_ext = generate_dot(
        all_modules, internal_deps, external_deps, cluster=True, include_external=True
    )
    dot_ext_path = OUTPUT_DIR / "pydeps_dependency_graph_with_externals.dot"
    dot_ext_path.write_text(dot_ext, encoding="utf-8")
    print(f"  DOT: {dot_ext_path}")

    # --- Generate SVGs ---

    svg_full_path = OUTPUT_DIR / "pydeps_dependency_graph.svg"
    svg_simple_path = OUTPUT_DIR / "pydeps_dependency_graph_simplified.svg"
    svg_ext_path = OUTPUT_DIR / "pydeps_dependency_graph_with_externals.svg"

    # Try graphviz dot first
    dot_ok = dot_to_svg(dot_full_path, svg_full_path)
    if dot_ok:
        print(f"\n  SVG (via dot): {svg_full_path}")
        dot_to_svg(dot_simple_path, svg_simple_path)
        print(f"  SVG (via dot): {svg_simple_path}")
        dot_to_svg(dot_ext_path, svg_ext_path)
        print(f"  SVG (via dot): {svg_ext_path}")
    else:
        print("\n  Graphviz dot unavailable/broken -- using pure-Python SVG renderer.")
        render_graph_as_svg(all_modules, internal_deps, svg_full_path, simplified=False)
        print(f"  SVG (fallback): {svg_full_path}")

        render_graph_as_svg(all_modules, internal_deps, svg_simple_path, simplified=True)
        print(f"  SVG (fallback): {svg_simple_path}")

        # For the externals graph, generate full-module view (same as full for fallback)
        render_graph_as_svg(all_modules, internal_deps, svg_ext_path, simplified=False)
        print(f"  SVG (fallback): {svg_ext_path}")

    # Summary
    print("\n--- Subpackage dependency summary ---")
    pkg_deps: dict[str, set[str]] = defaultdict(set)
    for src, targets in internal_deps.items():
        src_pkg = get_subpackage(src)
        for tgt in targets:
            tgt_pkg = get_subpackage(tgt)
            if src_pkg != tgt_pkg:
                pkg_deps[src_pkg].add(tgt_pkg)

    for pkg in sorted(pkg_deps):
        targets = sorted(pkg_deps[pkg])
        label = pkg.replace("src.", "")
        deps_str = ", ".join(t.replace("src.", "") for t in targets)
        print(f"  {label} -> {deps_str}")

    print(f"\nTo improve SVG quality, install graphviz system binary:")
    print(f"  Windows: winget install Graphviz.Graphviz")
    print(f"  Then re-run: python scripts/generate_depgraph.py")


if __name__ == "__main__":
    main()
