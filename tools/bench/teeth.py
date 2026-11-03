"""Teeth check for the chokepoint changes (probe-branch tooling only).

For each mutant: copy the tree at HEAD (git archive), weaken one guard, run
the committed tests that pin it, and require them to FAIL; then require the
same tests to PASS on the unmutated tree. Prints one row per mutant.

    python tools/bench/teeth.py --base <develop sha> [--only NAME ...]
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _archive(ref, dest):
    data = subprocess.run(
        ["git", "-C", ROOT, "archive", ref], capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


def _show(ref, path):
    return subprocess.run(
        ["git", "-C", ROOT, "show", f"{ref}:{path}"], capture_output=True, check=True
    ).stdout


def _append(rel, code):
    def patch(tree, _base):
        with open(os.path.join(tree, rel), "a", encoding="utf-8") as f:
            f.write("\n\n# teeth mutant\n" + code + "\n")

    return patch


def _from_base(rel):
    def patch(tree, base):
        with open(os.path.join(tree, rel), "wb") as f:
            f.write(_show(base, rel))

    return patch


PICKLE_TESTS = [
    "nltk/test/unit/test_attack_pickle_ext_scan.py::test_base0_int_line_no_longer_hides_ext",
    "nltk/test/unit/test_attack_pickle_ext_scan.py::test_scan_never_stops_on_argument_content",
]
JSON_TESTS = [
    "nltk/test/unit/test_attack_json_depth_scan.py::test_deep_document_after_any_string_is_refused",
    "nltk/test/unit/test_attack_json_depth_scan.py::test_hidden_deep_document_cannot_crash_a_raised_recursion_limit",
]

MUTANTS = {
    "picklesec-develop-genops-scan": (_from_base("nltk/picklesec.py"), PICKLE_TESTS),
    "picklesec-no-ext-scan": (
        _append(
            "nltk/picklesec.py",
            "def _reject_extension_opcodes(pickle_source):\n    return None",
        ),
        PICKLE_TESTS
        + [
            "nltk/test/unit/test_pickle_allowlist_security.py::TestExtensionOpcodeBypassBlocked"
        ],
    ),
    "jsontags-develop-scan": (_from_base("nltk/jsontags.py"), JSON_TESTS),
    "jsontags-no-depth-scan": (
        _append(
            "nltk/jsontags.py",
            "def _utf8_depth(text, limit):\n    return 0\n\n"
            "def _scan_json_depth(data, limit):\n    return 0",
        ),
        JSON_TESTS,
    ),
}


def _pytest(tree, tests):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", *tests],
        cwd=tree,
        capture_output=True,
        text=True,
        env=env,
    )
    tail = [l for l in proc.stdout.splitlines() if l.strip()][-1:] or [""]
    return proc.returncode, tail[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--work", default=os.path.join(ROOT, "teeth_trees"))
    args = p.parse_args()
    shutil.rmtree(args.work, ignore_errors=True)
    clean = os.path.join(args.work, "clean")
    _archive("HEAD", clean)
    ok = True
    print("| mutant | tests | unmutated | mutant | verdict |")
    print("|---|---|---|---|---|")
    for name, (patch, tests) in MUTANTS.items():
        if args.only and name not in args.only:
            continue
        tree = os.path.join(args.work, name)
        _archive("HEAD", tree)
        patch(tree, args.base)
        rc_clean, tail_clean = _pytest(clean, tests)
        rc_mut, tail_mut = _pytest(tree, tests)
        verdict = "TEETH" if rc_clean == 0 and rc_mut != 0 else "NO TEETH"
        ok &= verdict == "TEETH"
        print(
            f"| {name} | {len(tests)} | rc={rc_clean} {tail_clean} | rc={rc_mut} {tail_mut} | {verdict} |",
            flush=True,
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
