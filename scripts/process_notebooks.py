"""
Process solution notebooks: create filtered copies and executed outputs.

Overview
--------
This script performs two tasks for every Jupyter notebook found in a source
directory (default: `solutions`):

1) Filtered copy → writes to `notebooks/` by
    - Removing markdown cells whose first non-empty line starts with "**A:**" (also supports "A:")
    - Keeping code cells whose first non-empty line starts with "# Solution" but EMPTYING
      their content (source cleared, outputs removed, execution count reset)

2) Executed copy → writes to `solutions-output/` by executing the original
     notebook and saving the outputs. The source notebooks remain unchanged.

Kernel selection
----------------
By default, the script attempts to auto-detect a Jupyter kernel that matches
the currently running Python interpreter. If it cannot find a matching kernel,
it falls back to `python3`. You can override this with `--kernel-name`.

Typical usage
-------------
Run inside your desired Python/conda environment so the execution uses the
same interpreter and packages:

    # Use conda base and run filtering + execution
    conda activate base
    python notebooks-deeprl/scripts/process_notebooks.py

Filtering only (no execution):

    python notebooks-deeprl/scripts/process_notebooks.py --no-execute

Exclude some notebooks from execution (still filtered):

    python notebooks-deeprl/scripts/process_notebooks.py \
        --exclude "11-Keras" "13-PPO"

Custom folders (relative paths are resolved against the project root that
contains this script’s `scripts/` directory):

    python notebooks-deeprl/scripts/process_notebooks.py \
        --source-dir solutions \
        --filtered-dir notebooks \
        --executed-dir solutions-output

Force a specific kernel (optional):

    jupyter kernelspec list
    python notebooks-deeprl/scripts/process_notebooks.py --kernel-name python3

Recursive search through subfolders of the source directory:

    python notebooks-deeprl/scripts/process_notebooks.py --recursive

Arguments
---------
- --source-dir     : Source notebooks directory (default: solutions)
- --filtered-dir   : Output folder for filtered notebooks (default: notebooks)
- --executed-dir   : Output folder for executed notebooks (default: solutions-output)
- --exclude        : One or more notebook names to skip executing (still filtered)
- --recursive      : Recurse into subdirectories of the source directory
- --timeout        : Per-cell execution timeout in seconds (default: 1200)
- --kernel-name    : Kernel to run notebooks with (auto-detect if omitted)
- --no-execute     : Only produce filtered copies; skip execution entirely

Behavior summary
----------------
- Source notebooks are updated (only) to ensure the first cell is an "Open in Colab" badge with a correct link.
- Filtered notebooks are written to the filtered directory with the same
    relative paths as in the source.
- Executed notebooks are written to the executed directory with outputs
    preserved, unless excluded or `--no-execute` is set.

Colab badge cell
----------------
Before creating filtered and executed copies, the script ensures the first
cell is a markdown "Open in Colab" badge cell with metadata id
`view-in-github`. If the first cell does not have this id, a new first cell is
inserted. The badge link is updated per output so that it points to the exact
location of that specific notebook in the GitHub repository, e.g.:

    https://colab.research.google.com/github/Olimaol/notebooks-deeprl/blob/main/<folder>/<file>.ipynb
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Iterable, List, Set, Optional

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager


def read_notebook(path: Path):
    return nbformat.read(path, as_version=4)


def write_notebook(nb, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)


def make_colab_badge_cell(repo_path: str) -> nbformat.NotebookNode:
    # repo_path should be like "folder/name.ipynb" relative to repo root
    href = (
        f'<a href="https://colab.research.google.com/github/Olimaol/notebooks-deeprl/blob/main/{repo_path}" '
        f'target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )
    cell = nbformat.v4.new_markdown_cell(source=href)
    # Ensure required metadata
    cell.metadata = cell.metadata or {}
    cell.metadata["colab_type"] = "text"
    cell.metadata["id"] = "view-in-github"
    return cell


def ensure_and_update_colab_badge(nb: nbformat.NotebookNode, repo_path: str) -> None:
    # Ensure first cell exists and has the id 'view-in-github'; otherwise insert a new one
    if not nb.cells:
        nb.cells = [make_colab_badge_cell(repo_path)]
        return
    first = nb.cells[0]
    meta = getattr(first, "metadata", {}) or {}
    if meta.get("id") != "view-in-github":
        nb.cells.insert(0, make_colab_badge_cell(repo_path))
        return
    # Update existing badge cell with fresh link
    nb.cells[0] = make_colab_badge_cell(repo_path)


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def cell_should_be_dropped(cell) -> bool:
    ctype = cell.get("cell_type")
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    header = first_nonempty_line(source)

    if ctype == "markdown":
        # Drop markdown cells that start with "**A:**" or "A:"
        return header.startswith("**A:**") or header.startswith("A:")

    # Do not drop code cells anymore; solution cells will be emptied instead
    return False


def filter_notebook_cells(nb) -> None:
    new_cells = []
    for cell in nb.cells:
        if cell_should_be_dropped(cell):
            continue
        ctype = cell.get("cell_type")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        header = first_nonempty_line(source)
        if ctype == "code" and header.startswith("# Solution"):
            cell["source"] = ""
            cell["outputs"] = []
            cell["execution_count"] = None
        new_cells.append(cell)
    nb.cells = new_cells


def collect_notebooks(src_dir: Path, recursive: bool = True) -> List[Path]:
    pattern = "**/*.ipynb" if recursive else "*.ipynb"
    return sorted([p for p in src_dir.glob(pattern) if p.is_file()])


def execute_notebook(
    nb_path: Path, timeout: int = 1200, kernel_name: Optional[str] = None
) -> nbformat.NotebookNode:
    nb = read_notebook(nb_path)
    # Ensure relative paths in notebook (e.g., data loads) resolve from its directory
    resources = {"metadata": {"path": str(nb_path.parent)}}

    # Default to python3; typically resolves to the current env's kernel
    if kernel_name is None:
        kernel_name = "python3"

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources=resources,
        allow_errors=False,
        terminate_kernel=True,
    )
    client.execute()
    return nb


def normalize_names(names: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for n in names:
        n = n.strip()
        if not n:
            continue
        base = os.path.basename(n)
        if not base.endswith(".ipynb"):
            base += ".ipynb"
        out.add(base)
    return out


def process(
    source_dir: Path,
    filtered_dir: Path,
    executed_dir: Path,
    exclude_execute: Set[str],
    recursive: bool,
    timeout: int,
    kernel_name: Optional[str],
    do_execute: bool,
):
    notebooks = collect_notebooks(source_dir, recursive=recursive)
    if not notebooks:
        print(f"No notebooks found in {source_dir}")
        return

    print(f"Found {len(notebooks)} notebooks in {source_dir}")
    filtered_dir.mkdir(parents=True, exist_ok=True)
    executed_dir.mkdir(parents=True, exist_ok=True)

    for nb_path in notebooks:
        rel = nb_path.relative_to(source_dir)
        filtered_out = filtered_dir / rel
        executed_out = executed_dir / rel
        filtered_out.parent.mkdir(parents=True, exist_ok=True)
        executed_out.parent.mkdir(parents=True, exist_ok=True)

        # 0) Ensure original notebook has the Colab badge as first cell and update its link
        try:
            nb_src = read_notebook(nb_path)
            src_repo_path = str(
                nb_path.resolve().relative_to(Path(__file__).resolve().parents[1])
            )
            ensure_and_update_colab_badge(nb_src, src_repo_path)
            write_notebook(nb_src, nb_path)
        except Exception as e:
            print(
                f"WARNING: could not ensure Colab badge for {rel}: {e}", file=sys.stderr
            )

        # 1) Create filtered copy (ensure/update badge, then filter solution cells)
        nb_filtered = read_notebook(nb_path)
        # For filtered copy, the path to use in the badge is its repo-relative path
        filtered_repo_path = str(
            (filtered_out).relative_to(Path(__file__).resolve().parents[1])
        )
        ensure_and_update_colab_badge(nb_filtered, filtered_repo_path)
        filter_notebook_cells(nb_filtered)
        write_notebook(nb_filtered, filtered_out)
        print(f"Filtered -> {filtered_out}")

        # 2) Execute original notebook and save with outputs (unless excluded)
        if not do_execute:
            print(f"Skip execute (disabled): {rel}")
        elif rel.name in exclude_execute:
            print(f"Skip execute (excluded): {rel}")
        else:
            try:
                # Execute from original, then ensure/update badge for the executed copy
                executed_nb = execute_notebook(
                    nb_path, timeout=timeout, kernel_name=kernel_name
                )
                executed_repo_path = str(
                    (executed_out).relative_to(Path(__file__).resolve().parents[1])
                )
                ensure_and_update_colab_badge(executed_nb, executed_repo_path)
                write_notebook(executed_nb, executed_out)
                print(f"Executed -> {executed_out}")
            except Exception as e:
                print(f"ERROR executing {rel}: {e}", file=sys.stderr)
                # Still continue to next notebooks


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter solution notebooks and execute copies with outputs. "
            "Creates a filtered copy that removes answer markdown and empties solution code cells, "
            "and an executed copy with outputs."
        )
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("solutions"),
        help="Directory containing source solution notebooks (default: solutions)",
    )
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        default=Path("notebooks"),
        help="Directory to write filtered notebooks without solutions (default: notebooks)",
    )
    parser.add_argument(
        "--executed-dir",
        type=Path,
        default=Path("solutions-output"),
        help="Directory to write executed notebooks with outputs (default: solutions-output)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help=(
            "Notebook names to skip executing (basename or path; repeatable). "
            "These will still be filtered into the filtered-dir."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process subfolders of source-dir",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Execution timeout per cell in seconds (default: 1200)",
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        default=None,
        help=(
            "Kernel name to use for execution. If omitted, the script "
            "tries to auto-detect a kernel matching the current Python "
            "executable; otherwise falls back to 'python3'."
        ),
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Do not execute notebooks; only create filtered copies.",
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None):
    ns = parse_args(argv or sys.argv[1:])

    # Resolve directories relative to this script's project root (parent of scripts/)
    # so defaults like "solutions" map to notebooks-deeprl/solutions regardless of CWD.
    script_root = Path(__file__).resolve().parents[1]
    source_dir = (
        (script_root / ns.source_dir)
        if not ns.source_dir.is_absolute()
        else ns.source_dir
    )
    filtered_dir = (
        (script_root / ns.filtered_dir)
        if not ns.filtered_dir.is_absolute()
        else ns.filtered_dir
    )
    executed_dir = (
        (script_root / ns.executed_dir)
        if not ns.executed_dir.is_absolute()
        else ns.executed_dir
    )

    exclude_set = normalize_names(ns.exclude)

    # Resolve kernel name: prefer user-provided, else auto-detect matching current interpreter
    def autodetect_kernel_for_current_python() -> str:
        try:
            ksm = KernelSpecManager()
            specs = ksm.get_all_specs()  # {name: {resource_dir, spec}}
            exe = os.path.realpath(sys.executable)
            for name, info in specs.items():
                argv = info.get("spec", {}).get("argv", [])
                if not argv:
                    continue
                cmd0 = os.path.realpath(argv[0]) if os.path.isabs(argv[0]) else argv[0]
                # Prefer exact path match; otherwise compare basenames as fallback
                if cmd0 == exe or os.path.basename(cmd0) == os.path.basename(exe):
                    return name
        except Exception:
            pass
        return "python3"

    effective_kernel = ns.kernel_name or autodetect_kernel_for_current_python()

    print("Settings:")
    print(f"  Source dir   : {source_dir}")
    print(f"  Filtered dir : {filtered_dir}")
    print(f"  Executed dir : {executed_dir}")
    if exclude_set:
        print(f"  Exclude exec : {sorted(exclude_set)}")
    else:
        print("  Exclude exec : []")
    print(f"  Recursive    : {ns.recursive}")
    print(f"  Timeout      : {ns.timeout}s")
    print(f"  Kernel name  : {effective_kernel}")
    print(f"  Execute      : {not ns.no_execute}")

    process(
        source_dir=source_dir,
        filtered_dir=filtered_dir,
        executed_dir=executed_dir,
        exclude_execute=exclude_set,
        recursive=ns.recursive,
        timeout=ns.timeout,
        kernel_name=effective_kernel,
        do_execute=not ns.no_execute,
    )


if __name__ == "__main__":
    main()
