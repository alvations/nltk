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


def _replace(rel, old, new):
    def patch(tree, _base):
        path = os.path.join(tree, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert text.count(old) == 1, (rel, old)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new))

    return patch


def _from_ref(ref, rel):
    def patch(tree, _base):
        with open(os.path.join(tree, rel), "wb") as f:
            f.write(_show(ref, rel))

    return patch


def _from_base(rel):
    def patch(tree, base):
        with open(os.path.join(tree, rel), "wb") as f:
            f.write(_show(base, rel))

    return patch


U = "nltk/test/unit"
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
    "picklesec-no-frame-rule": (
        _from_ref("b9386c7d1", "nltk/picklesec.py"),
        [
            f"{U}/test_attack_pickle_ext_scan.py::test_argument_across_a_frame_end_cannot_hide_ext"
        ],
    ),
    "jsontags-develop-scan": (_from_base("nltk/jsontags.py"), JSON_TESTS),
    "jsontags-walk-skips-every-container": (
        _append("nltk/jsontags.py", "def _finite_numbers(values):\n    return True"),
        [f"{U}/test_jsontags_fast_paths.py::test_non_finite_weight_is_still_refused"],
    ),
    "jsontags-bracket-count-zero": (
        _append(
            "nltk/jsontags.py", "def _bracket_depth(residue, limit):\n    return 0"
        ),
        [
            f"{U}/test_jsontags_fast_paths.py::test_bracket_count_returns_what_the_loop_returns"
        ]
        + JSON_TESTS[:1],
    ),
    "jsontags-read-ignores-cap": (
        _append(
            "nltk/jsontags.py",
            "def _read_capped(reader, size):\n    return reader.read()",
        ),
        [
            f"{U}/test_capped_reads_memory.py::test_capped_json_read_refuses_and_reads_exactly_as_one_read"
        ],
    ),
    "jsontags-read-reserves-request": (
        _append("nltk/jsontags.py", "_READ_STEP = 10**12"),
        [
            f"{U}/test_capped_reads_memory.py::test_a_small_json_file_no_longer_reserves_the_cap"
        ],
    ),
    "data-read-reserves-request": (
        _append("nltk/data.py", "_READ_STEP = 10**12"),
        [
            f"{U}/test_capped_reads_memory.py::test_a_small_json_file_no_longer_reserves_the_cap"
        ],
    ),
    "termsec-no-dangerous-search": (
        _append("nltk/termsec.py", "_PRINTABLE_BUT_DANGEROUS_SORTED = ()"),
        [f"{U}/test_termsec_fast_paths.py::test_every_code_point_alone"],
    ),
    "termsec-separators-unsanitised": (
        _append(
            "nltk/termsec.py",
            "class _Everything:\n    def __contains__(self, item):\n        return True\n\n\n"
            "_CLEAN_SEPARATORS = _Everything()",
        ),
        [
            f"{U}/test_termsec_fast_paths.py::test_default_separators_are_printed_as_before"
        ],
    ),
    "pathsec-within-any-ignores-parents": (
        _append(
            "nltk/pathsec.py",
            "def _within_any(target, roots):\n    return target in roots",
        ),
        [f"{U}/test_pathsec_fast_paths.py::test_containment_matches_is_relative_to"],
    ),
    "pathsec-within-any-always": (
        _append("nltk/pathsec.py", "def _within_any(target, roots):\n    return True"),
        [f"{U}/test_pathsec_fast_paths.py::test_containment_matches_is_relative_to"],
    ),
    "pathsec-scheme-always-skipped": (
        _append("nltk/pathsec.py", "def _may_name_a_scheme(raw):\n    return False"),
        [f"{U}/test_pathsec_fast_paths.py::test_url_shapes_are_still_refused"],
    ),
    "punkt-numeric-shortcut-on-digits": (
        _append(
            "nltk/tokenize/punkt.py",
            "_NOT_NUMBER_START = frozenset(map(chr, range(0x80)))",
        ),
        [
            f"{U}/test_punkt_fast_paths.py::test_numeric_type_matches_the_regex_for_every_first_character"
        ],
    ),
    "punkt-ellipsis-shortcut-always": (
        _replace("nltk/tokenize/punkt.py", 'self.tok[:2] != ".."', "True"),
        [f"{U}/test_punkt_fast_paths.py::test_is_ellipsis_matches_the_regex"],
    ),
    "punkt-initial-shortcut-always": (
        _replace("nltk/tokenize/punkt.py", 'self.tok[1:2] != "."', "True"),
        [f"{U}/test_punkt_fast_paths.py::test_is_initial_matches_the_regex"],
    ),
    "destructive-needs-never-present": (
        _append(
            "nltk/tokenize/destructive.py",
            '_NEEDS = {key: "\\x00" for key in _NEEDS}',
        ),
        [
            f"{U}/test_destructive_fast_paths.py::test_no_match_without_a_needed_character"
        ],
    ),
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
