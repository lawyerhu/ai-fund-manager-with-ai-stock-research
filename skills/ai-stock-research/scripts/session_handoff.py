"""Session handoff: the current conversation model performs the stock research.

`prepare` exports an immutable historical baseline into a new run directory.
The current conversation then does the research itself (ranking, supplemental
deep research, incumbent comparison) and saves its decision plus evidence.
`finalize` validates coverage, evidence references and selection state, then
records the new selection.

This entry point never calls a model API. It imports only the project's
standalone `src/session_research.py` validator; the project runtime, `.env`, the
production database, the Risk Engine and any broker are never imported, read or
contacted, and no order is ever sent. Reasoning effort is not exposed by a
session, so it is recorded as `NOT_EXPOSED_BY_SESSION` rather than assumed.
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
from pathlib import Path
import sys

from research_model import (
    SESSION_MODEL_ENV,
    manifest_fields,
    session_model,
)


class NoProjectRuntimeImports(importlib.abc.MetaPathFinder):
    """The handoff must never import the project runtime or its trading code."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src" or fullname.startswith("src."):
            raise ImportError(f"Session research handoff blocks project runtime import: {fullname}")
        return None


def load_handoff_module(project):
    """Load the project's standalone session validator by file path."""
    path = Path(project).resolve() / "src" / "session_research.py"
    if not path.is_file():
        raise ValueError(f"Project does not provide the session research module: {path}")
    spec = importlib.util.spec_from_file_location("session_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "finalize"])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--session-model", default=None,
                        help="Declared by the active session; not an API model-id verification. "
                             f"Falls back to {SESSION_MODEL_ENV} or the built-in default.")
    parser.add_argument("--source-result", type=Path, help="Completed unified-research result.json")
    parser.add_argument("--previous-result", type=Path, help="Previous COMPLETE result.json")
    parser.add_argument("--verification-packet", type=Path, help="Same-day verified evidence packet")
    parser.add_argument("--output-root", type=Path, help="Root directory for the new run")
    parser.add_argument("--run-directory", type=Path, help="Run directory created by prepare")
    parser.add_argument("--decision", type=Path, help="Session-produced decision JSON")
    parser.add_argument("--evidence", type=Path, help="Session-produced evidence packet JSON")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    project = args.project.resolve()
    declared = session_model(args.session_model)
    sys.meta_path.insert(0, NoProjectRuntimeImports())
    module = load_handoff_module(project)
    if args.mode == "prepare":
        missing = [name for name in ("source_result", "previous_result", "verification_packet", "output_root")
                   if getattr(args, name) is None]
        if missing:
            parser.error("prepare requires " + ", ".join("--" + name.replace("_", "-") for name in missing))
        namespace = argparse.Namespace(
            source_result=args.source_result,
            previous_result=args.previous_result,
            verification_packet=args.verification_packet,
            output_root=args.output_root,
            session_model=declared,
        )
        output = module.prepare(namespace)
    else:
        missing = [name for name in ("run_directory", "decision", "evidence") if getattr(args, name) is None]
        if missing:
            parser.error("finalize requires " + ", ".join("--" + name.replace("_", "-") for name in missing))
        output = module.finalize(argparse.Namespace(
            run_directory=args.run_directory,
            decision=args.decision,
            evidence=args.evidence,
        )) or args.run_directory.resolve()
    print("OUTPUT_DIRECTORY=" + str(output), flush=True)
    print("RESEARCH_ROLE=" + str(manifest_fields(declared)), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Session handoff stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
