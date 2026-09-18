#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

MANIFEST_REL = Path("publication/PUBLIC_SNAPSHOT_CODE_MANIFEST.txt")
EXPECTED_CODE_FILES = 160
BUILDER_REL = Path("scripts/release/build_public_snapshot.py")

EXTRA_FILES = (
    ".gitattributes",
    ".gitignore",
    "README.md",
    "publication/CITATION.cff",
    "publication/CLAIMS_AND_EVIDENCE.md",
    "publication/EVIDENCE_SUMMARY.md",
    "publication/REFERENCES.bib",
    "publication/REPRODUCIBILITY.md",
    "publication/PROVENANCE_HISTORY.md",
    "publication/PHASE2_PROVENANCE_REVIEW.md",
    "publication/APACHE_RELEASE_SCOPE.md",
    "publication/RELEASE_SCOPE_HOLD.md",
    "publication/SUPPORTED_ENTRY_POINTS.md",
    "publication/PUBLIC_RELEASE_AUDIT.md",
    "publication/PUBLICATION_ARTIFACT_WORDING_REVIEW.md",
    "publication/LICENSE_AND_NOTICE_REVIEW.md",
    "docs/README.md",
    "docs/WeaveCube_Technical_Report_v1.pdf",
    "presentation/WeaveCube_Core_Deck_v1.pptx",
    "presentation/README.md",
    "tests/test_domino_phase2.py",
    MANIFEST_REL.as_posix(),
    BUILDER_REL.as_posix(),
)

ROOT_MAPPED_FILES = (
    ("publication/public_snapshot/LICENSE", "LICENSE"),
    ("publication/public_snapshot/NOTICE", "NOTICE"),
    ("publication/public_snapshot/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
)

FORBIDDEN_CODE_MARKERS = (
    "twophase.",
    "kociemba_p1_runtime_v36",
    "audit_gx31_nissy_index_translation",
)

def _rmtree_onexc(func, path, exc_info) -> None:
    """Retry Windows deletions after clearing the read-only attribute.

    Git object files may be read-only on Windows.  A stale public-snapshot
    repository must still be replaceable without weakening source-tree
    safety.  If the retry fails (for example because another process has the
    file open), propagate the original error.
    """
    exc = exc_info if isinstance(exc_info, BaseException) else exc_info[1]
    if not isinstance(exc, PermissionError):
        raise exc

    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    func(path)


def remove_existing_destination(dest: Path) -> None:
    try:
        shutil.rmtree(dest, onexc=_rmtree_onexc)
    except PermissionError as exc:
        raise SystemExit(
            "cannot replace destination; Windows is still locking a file under "
            f"{dest}. Close shells/editors using that tree or choose a fresh "
            "destination path, then retry."
        ) from exc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def copy_one(src_root: Path, dst_root: Path, rel: str) -> dict[str, object]:
    src = src_root / rel
    if not src.is_file():
        raise FileNotFoundError(f"manifest file missing: {src}")
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": rel, "bytes": dst.stat().st_size, "sha256": sha256(dst)}

def _module_candidates(current_rel: str, node: ast.AST) -> list[str]:
    """Resolve concrete project-local Python module files.

    Deliberately do not require package __init__.py files.  The sanitized
    snapshot may use namespace-package directories to avoid importing broad
    historical package surfaces.  Runtime import smoke remains the final
    authority for supported entry points.
    """
    current = Path(current_rel)
    candidates: list[str] = []

    def add_module(module: str) -> None:
        if not module:
            return
        parts = module.split(".")
        if parts[0] in {"cubelab", "cubelab_global_backbone"}:
            candidates.append("src/" + "/".join(parts) + ".py")
        elif len(parts) == 1:
            candidates.append("scripts/" + parts[0] + ".py")

    if isinstance(node, ast.Import):
        for alias in node.names:
            add_module(alias.name)

    elif isinstance(node, ast.ImportFrom):
        level = int(node.level or 0)
        module = node.module or ""

        if level:
            if not current_rel.startswith("src/"):
                return candidates

            package_parts = list(current.parent.parts[1:])  # drop leading src
            up = max(0, level - 1)
            if up:
                package_parts = package_parts[:-up]

            base = list(package_parts)
            if module:
                base.extend(module.split("."))

            if module and base:
                candidates.append("src/" + "/".join(base) + ".py")

            # "from .pkg import submodule" and "from . import submodule"
            # may resolve to a concrete sibling/child module.
            for alias in node.names:
                alias_base = list(base if module else package_parts)
                alias_base.append(alias.name)
                candidates.append("src/" + "/".join(alias_base) + ".py")

        else:
            add_module(module)

            # For "from cubelab.pkg import submodule", also test whether the
            # imported name is itself a concrete module file.
            if module.startswith("cubelab") or module.startswith("cubelab_global_backbone"):
                for alias in node.names:
                    add_module(module + "." + alias.name)

    return candidates


def verify_manifest_not_ignored(source: Path, code_files: list[str]) -> None:
    """Fail if source .gitignore rules would hide an approved public code file."""
    proc = subprocess.run(
        ["git", "-C", str(source), "check-ignore", "--no-index", "--stdin"],
        input="\n".join(code_files) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # git check-ignore returns 0 when at least one path matched, 1 when none matched.
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            "git check-ignore failed: "
            + (proc.stderr.strip() or f"returncode={proc.returncode}")
        )
    ignored = sorted({
        line.strip().replace("\\", "/")
        for line in proc.stdout.splitlines()
        if line.strip()
    })
    if ignored:
        print(json.dumps({"manifest_paths_ignored_by_gitignore": ignored}, indent=2))
        raise SystemExit(
            "approved public code manifest contains paths hidden by .gitignore"
        )


def verify_static_project_import_closure(
    source: Path,
    code_files: list[str],
    tracked_paths: set[str],
) -> None:
    """Fail if an obvious local import resolves to an exact tracked file but is absent.

    Git's exact tracked path set is authoritative here. This avoids Windows
    case-insensitive false positives such as an imported symbol `MOVES`
    being misread as the tracked module `moves.py`.
    """
    approved = set(code_files)
    missing: dict[str, list[str]] = {}

    for rel in code_files:
        if not rel.endswith(".py"):
            continue
        path = source / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
        except SyntaxError:
            # compileall remains the syntax authority; do not hide the actual syntax error here.
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for candidate in _module_candidates(rel, node):
                if candidate in tracked_paths and candidate not in approved:
                    missing.setdefault(rel, []).append(candidate)

    if missing:
        normalized = {
            key: sorted(set(values))
            for key, values in sorted(missing.items())
        }
        print(json.dumps({"missing_project_imports": normalized}, indent=2))
        raise SystemExit("public snapshot manifest is not closed under obvious project-local imports")

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a history-free WeaveCube public snapshot from an approved manifest."
    )
    ap.add_argument("--source", required=True, help="Private WeaveCube working tree")
    ap.add_argument("--dest", required=True, help="New/empty public snapshot directory")
    ap.add_argument(
        "--replace-dest",
        action="store_true",
        help="Delete destination first if it exists. Never touches source.",
    )
    ns = ap.parse_args()

    source = Path(ns.source).resolve()
    dest = Path(ns.dest).resolve()

    if not (source / ".git").exists():
        raise SystemExit(f"source is not a Git working tree: {source}")
    if source == dest or source in dest.parents:
        raise SystemExit("destination must be outside the private source repository")

    if dest.exists():
        if not ns.replace_dest:
            if any(dest.iterdir()):
                raise SystemExit(f"destination exists and is not empty: {dest}")
        else:
            remove_existing_destination(dest)
    dest.mkdir(parents=True, exist_ok=True)

    manifest_path = source / MANIFEST_REL
    if not manifest_path.is_file():
        raise SystemExit(f"missing approved manifest: {manifest_path}")
    code_files = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(code_files) != len(set(code_files)):
        raise SystemExit("duplicate paths in public code manifest")
    if len(code_files) != EXPECTED_CODE_FILES:
        raise SystemExit(f"expected {EXPECTED_CODE_FILES} code files, found {len(code_files)}")
    bad_roots = [p for p in code_files if not (p.startswith("src/") or p.startswith("scripts/"))]
    if bad_roots:
        raise SystemExit(f"code manifest contains unexpected roots: {bad_roots[:10]}")

    tracked_paths = set(
        subprocess.run(
            ["git", "-C", str(source), "ls-files"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.splitlines()
    )

    verify_manifest_not_ignored(source, code_files)

    verify_static_project_import_closure(
        source,
        code_files,
        tracked_paths,
    )

    source_head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()

    files = code_files + [p for p in EXTRA_FILES if p not in code_files]
    receipts = [copy_one(source, dest, rel) for rel in files]

    for source_rel, dest_rel in ROOT_MAPPED_FILES:
        src = source / source_rel
        if not src.is_file():
            raise FileNotFoundError(f"mapped snapshot file missing: {src}")
        dst = dest / dest_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        receipts.append({
            "path": dest_rel,
            "source_path": source_rel,
            "bytes": dst.stat().st_size,
            "sha256": sha256(dst),
        })

    # Release-code grep gate. Documentation is intentionally allowed to discuss prior art.
    code_roots = ("src/", "scripts/")
    violations: list[dict[str, object]] = []
    for rel in code_files:
        if not rel.startswith(code_roots):
            continue
        path = dest / rel
        if path.suffix.lower() != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in FORBIDDEN_CODE_MARKERS:
            if marker.lower() in text.lower():
                violations.append({"path": rel, "marker": marker})

    if violations:
        print(json.dumps({"forbidden_code_markers": violations}, indent=2))
        raise SystemExit("public snapshot contains forbidden historical integration markers")

    # Assert no copied Git history.
    if (dest / ".git").exists():
        raise SystemExit("destination unexpectedly contains .git")

    receipt = {
        "schema": "weavecube.public-snapshot.v1",
        "source": str(source),
        "source_head": source_head,
        "code_manifest": MANIFEST_REL.as_posix(),
        "code_files": len(code_files),
        "total_files": len(receipts),
        "license": "Apache-2.0",
        "root_mapped_files": [
            {"source": source_rel, "destination": dest_rel}
            for source_rel, dest_rel in ROOT_MAPPED_FILES
        ],
        "files": receipts,
    }
    (dest / "PUBLIC_SNAPSHOT_BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("# WeaveCube public snapshot build")
    print(f"source      : {source}")
    print(f"destination : {dest}")
    print(f"source HEAD : {source_head}")
    print(f"code files  : {len(code_files)}")
    print(f"total files : {len(receipts)}")
    print("git history : NOT COPIED")
    print("license     : Apache-2.0")
    print("notice pack : PASS")
    print("ignore gate : PASS")
    print("import gate : PASS")
    print("marker gate : PASS")
    print("BUILD: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
