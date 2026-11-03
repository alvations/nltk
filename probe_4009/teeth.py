"""Teeth of the #4009 tests: run them against develop's redos.py and two mutants.

Usage: python probe_4009/teeth.py <develop_redos.py> <work_dir>
Run from the branch checkout. Exits non-zero unless every variant fails exactly
where it must: develop on the five consumer tests, the per-search mutant on the
bounded-total test, the no-skip mutant on the equivalence tests; and the branch
itself passes all.
"""

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEVELOP_REDOS, WORK = Path(sys.argv[1]), Path(sys.argv[2])
REPO = Path.cwd()
TEST = "nltk/test/unit/test_redos.py"
CLASS = "TestFinditerChargesMatchingOnly"

SKIP_LINES = """                    if not batch and m.start() == m.end() == empty_at:
                        continue
"""
BATCH_START = '        bound.update(kwargs)\n        pos, endpos = bound.get("pos"), bound.get("endpos")'
BATCH_END = "            size = min(2 * size, _FINDITER_BATCH)\n"
PER_SEARCH = '''        bound.update(kwargs)
        pos, endpos = bound.get("pos"), bound.get("endpos")
        empty_at = None
        while True:
            try:
                it = self._rx.finditer(string, pos, endpos, timeout=t)
                m = next(it, None)
                if m is not None and m.start() == m.end() == empty_at:
                    m = next(it, None)
            except TimeoutError:
                raise self._fail(t) from None
            if m is None:
                return
            yield m
            pos = m.end()
            empty_at = pos if m.start() == pos else None
'''


def variant(name, redos_src, test_src):
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(REPO / "nltk", root / "nltk")
    for f in ("pytest.ini", "setup.cfg", "pyproject.toml", "conftest.py"):
        if (REPO / f).exists():
            shutil.copy(REPO / f, root / f)
    (root / "nltk/redos.py").write_text(redos_src, encoding="utf-8")
    (root / TEST).write_text(test_src, encoding="utf-8")
    xml = root / "junit.xml"
    cmd = [sys.executable, "-m", "pytest", TEST, "-q", "-p", "no:cacheprovider",
           "-k", CLASS, "-n", "auto", f"--junitxml={xml}"]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    print(f"== {name}: exit {proc.returncode}\n{proc.stdout[-1500:]}")
    failed, total = set(), 0
    for case in ET.parse(xml).iter("testcase"):
        total += 1
        if case.find("failure") is not None or case.find("error") is not None:
            failed.add(case.get("name"))
    return failed, total


branch_redos = (REPO / "nltk/redos.py").read_text(encoding="utf-8")
branch_test = (REPO / TEST).read_text(encoding="utf-8")
assert branch_redos.count(SKIP_LINES) == 1
start = branch_redos.index(BATCH_START)
end = branch_redos.index(BATCH_END) + len(BATCH_END)
# develop has no _FINDITER_BATCH: the batch parameter has nothing to patch there.
develop_test = branch_test.replace("[1, 3, redos._FINDITER_BATCH]", "[1, 3, 256]").replace(
    'monkeypatch.setattr(redos, "_FINDITER_BATCH", batch)',
    'monkeypatch.setattr(redos, "_FINDITER_BATCH", batch, raising=False)',
)

results = {
    "branch": variant("branch", branch_redos, branch_test),
    "develop": variant("develop", DEVELOP_REDOS.read_text(encoding="utf-8"), develop_test),
    "persearch": variant("persearch", branch_redos[:start] + PER_SEARCH + branch_redos[end:], branch_test),
    "noskip": variant("noskip", branch_redos.replace(SKIP_LINES, ""), branch_test),
}
checks = {
    "branch passes all": not results["branch"][0] and results["branch"][1] > 0,
    "develop fails all 5 consumer tests and nothing else": (
        len(results["develop"][0]) == 5
        and all(n.startswith("test_consumer_work_between_items_is_not_charged") for n in results["develop"][0])
    ),
    "per-search mutant fails the bounded-total test": (
        "test_total_matching_time_is_still_bounded" in results["persearch"][0]
    ),
    "no-skip mutant fails equivalence": any(
        n.startswith("test_same_matches_as_one_scanner") for n in results["noskip"][0]
    ),
}
print("\n| variant | tests | failed |\n|---|---|---|")
for name, (failed, total) in results.items():
    print(f"| {name} | {total} | {len(failed)}: {', '.join(sorted(failed))[:600]} |")
print()
for label, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}: {label}")
sys.exit(0 if all(checks.values()) else 1)
