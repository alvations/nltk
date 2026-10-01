"""Runner diagnostics for the suite's CPU-time scaling tests.

Measures, on the machine it runs on:
  1. the spread of CPU time per fixed unit of work, idle and beside two busy
     siblings (xdist-like load), with the time series so the switching
     structure is visible: a bimodal spread means heterogeneous core rates
     or steal counted as guest CPU time;
  2. the relextract linear op at a ladder of sizes;
  3. the CoNLL SRL fixed sink and the pre-fix rescan on the current grid
     design (words == predicates) and on a constant-words design;
  4. two ways of taking the small/big scaling ratio, over repeated trials:
     the current helper (min of single runs on each side) and a block form
     (the small side timed as four calls in one block, so both sides are
     about as long and sample the same rate windows).
Prints plain text; nothing is asserted.
"""

import multiprocessing
import os
import platform
import random
import subprocess
import sys
import tempfile
import time

ROOT = os.environ.get("DIAG_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, ROOT)

INF = float("inf")


def _busy(seconds):
    deadline = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < deadline:
        for _ in range(1000):
            x += 1


def cpu_and_wall(fn, *args):
    c0, w0 = time.process_time(), time.perf_counter()
    fn(*args)
    return time.process_time() - c0, time.perf_counter() - w0


def best(fn, *args, reps=3):
    cpu = wall = INF
    for _ in range(reps):
        c, w = cpu_and_wall(fn, *args)
        cpu, wall = min(cpu, c), min(wall, w)
    return cpu, wall


def _div(a, b):
    return a / b if b else INF


def header():
    print("=" * 72)
    print(platform.platform(), "python", sys.version.split()[0])
    print("cpu_count", os.cpu_count(), "machine", platform.machine())
    if sys.platform == "darwin":
        for key in (
            "machdep.cpu.brand_string",
            "hw.ncpu",
            "hw.perflevel0.logicalcpu",
            "hw.perflevel1.logicalcpu",
            "hw.l2cachesize",
            "hw.memsize",
        ):
            try:
                out = subprocess.run(
                    ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
                )
                print("  %s = %s" % (key, out.stdout.strip() or out.stderr.strip()))
            except Exception as exc:
                print("  %s ? %s" % (key, exc))
    print("=" * 72)


# ---- 1. CPU time per unit of work ------------------------------------------
def _unit():
    x = 0
    for _ in range(1_000_000):
        x += 1


def spread(label, chunks=120):
    cpus, walls = [], []
    for _ in range(chunks):
        c, w = cpu_and_wall(_unit)
        cpus.append(c)
        walls.append(w)
    lo = min(cpus) or 1e-9
    series = "".join(
        "." if c < 1.3 * lo else (":" if c < 1.7 * lo else "#") for c in cpus
    )
    scpu, swall = sorted(cpus), sorted(walls)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    print(
        "[spread %-12s] cpu/unit  min %.4f p10 %.4f p50 %.4f p90 %.4f max %.4f"
        % (label, scpu[0], pct(scpu, 0.1), pct(scpu, 0.5), pct(scpu, 0.9), scpu[-1])
    )
    print(
        "[spread %-12s] wall/unit min %.4f p10 %.4f p50 %.4f p90 %.4f max %.4f"
        % (
            label,
            swall[0],
            pct(swall, 0.1),
            pct(swall, 0.5),
            pct(swall, 0.9),
            swall[-1],
        )
    )
    print(
        "[spread %-12s] cpu max/min %.2fx  cpu p90/p10 %.2fx  wall p50/cpu p50 %.2fx"
        % (
            label,
            _div(scpu[-1], scpu[0]),
            _div(pct(scpu, 0.9), pct(scpu, 0.1)),
            _div(pct(swall, 0.5), pct(scpu, 0.5)),
        )
    )
    print("[spread %-12s] series (.<1.3x min, :<1.7x, # above): %s" % (label, series))


# ---- 2. relextract -----------------------------------------------------------
def _relextract_op():
    from nltk.sem.relextract import semi_rel2reldict, tree2semi_rel
    from nltk.test.unit.test_attack_relextract_expanded import _chunk_tree

    rng = random.Random(0)

    def op(n):
        pairs = tree2semi_rel(_chunk_tree(rng, n))
        assert len(semi_rel2reldict(pairs)) == max(0, len(pairs) - 2)

    return op


def relextract_ladder(label):
    op = _relextract_op()
    res = {}
    for n in (750, 1500, 3000, 6000, 12000, 24000):
        res[n] = best(op, n)
        print("[relextract %-6s] n=%6d cpu %.4f wall %.4f" % (label, n, *res[n]))
    for a in (750, 1500, 3000, 6000):
        b = 4 * a
        print(
            "[relextract %-6s] %d->%d cpu ratio %.2fx (floored %.2fx)  wall ratio %.2fx"
            % (
                label,
                a,
                b,
                _div(res[b][0], res[a][0]),
                _div(res[b][0], max(res[a][0], 0.1)),
                _div(res[b][1], res[a][1]),
            )
        )


# ---- 3. CoNLL SRL -----------------------------------------------------------
def _grid_square(n):
    grid = []
    for i in range(n):
        row = ["w", "NN", "*", "verb.01", "p"]
        row += ["(V*)" if i == j else "(A1*)" for j in range(n)]
        grid.append(row)
    return grid


def _grid_const_words(predicates, words):
    grid = []
    for i in range(words):
        if i < predicates:
            row = ["w", "NN", "*", "verb.01", "p"]
        else:
            row = ["w", "NN", "*", "-", "-"]
        row += ["(V*)" if i == j else "(A1*)" for j in range(predicates)]
        grid.append(row)
    return grid


def _srl_reader():
    from nltk.corpus.reader.conll import ConllCorpusReader
    from nltk.test.unit.security_probes._base import register_data_root

    box = tempfile.mkdtemp()
    undo = register_data_root(box)
    root = os.path.join(box, "corpus")
    os.makedirs(root)
    open(os.path.join(root, "srl.conll"), "w").close()
    reader = ConllCorpusReader(root, ["srl.conll"], ("words", "pos", "tree", "srl"))
    return reader, undo


def srl(label):
    from nltk.test.unit.test_attack_corpus_readers_expanded import (
        _reference_srl_instances,
    )

    reader, undo = _srl_reader()

    def run(name, grids, pairs):
        res = {}
        for key, g in grids.items():
            f = best(lambda: reader._get_srl_instances(g, False), reps=2)[0]
            r = best(lambda: _reference_srl_instances(reader, g), reps=2)[0]
            res[key] = (f, r)
            print("[srl %-6s %s] %s fixed %.4f ref %.4f" % (label, name, key, f, r))
        for a, b in pairs:
            print(
                "[srl %-6s %s] %s->%s fixed %.1fx (floored %.1fx)  ref %.1fx (floored %.1fx)"
                % (
                    label,
                    name,
                    a,
                    b,
                    _div(res[b][0], res[a][0]),
                    _div(res[b][0], max(res[a][0], 0.1)),
                    _div(res[b][1], res[a][1]),
                    _div(res[b][1], max(res[a][1], 0.1)),
                )
            )

    try:
        run("square", {n: _grid_square(n) for n in (70, 280)}, [(70, 280)])
        for words in (600, 800):
            run(
                "W%d" % words,
                {"P%d" % p: _grid_const_words(p, words) for p in (70, 280)},
                [("P70", "P280")],
            )
    finally:
        undo()


# ---- 4. ratio methods over trials -------------------------------------------
def current_ratio(op, small, big, reps=3, floor=0.1):
    cs = cb = INF
    for _ in range(reps):
        cs = min(cs, cpu_and_wall(op, small)[0])
        cb = min(cb, cpu_and_wall(op, big)[0])
    for _ in range(reps):
        cs = min(cs, cpu_and_wall(op, small)[0])
    return cb / max(cs, floor), cs, cb


def block_ratio(op, small, big, reps=3, floor=0.1):
    cs = cb = INF

    def block():
        for _ in range(4):
            op(small)

    for _ in range(reps):
        cs = min(cs, cpu_and_wall(block)[0] / 4)
        cb = min(cb, cpu_and_wall(op, big)[0])
    return cb / max(cs, floor), cs, cb


def trials(label, name, op, small, big, n, expect):
    for method in (current_ratio, block_ratio):
        ratios = []
        t0 = time.perf_counter()
        for _ in range(n):
            ratios.append(method(op, small, big)[0])
        ratios.sort()
        print(
            "[ratio %-6s %-18s %-7s] %s: min %.2fx med %.2fx max %.2fx  over 8: %d/%d  (%.0fs)"
            % (
                label,
                name,
                method.__name__.split("_")[0],
                expect,
                ratios[0],
                ratios[len(ratios) // 2],
                ratios[-1],
                sum(r >= 8 for r in ratios),
                n,
                time.perf_counter() - t0,
            )
        )


def ratio_trials(label):
    from nltk.test.unit.test_attack_corpus_readers_expanded import (
        _reference_srl_instances,
    )

    trials(label, "relextract 3k/12k", _relextract_op(), 3000, 12000, 8, "linear")
    reader, undo = _srl_reader()
    try:
        grids = {p: _grid_const_words(p, 600) for p in (70, 280)}
        trials(
            label,
            "srl W600 fixed",
            lambda p: reader._get_srl_instances(grids[p], False),
            70,
            280,
            6,
            "linear",
        )
        trials(
            label,
            "srl W600 ref",
            lambda p: _reference_srl_instances(reader, grids[p]),
            70,
            280,
            3,
            "quadratic",
        )
    finally:
        undo()


def main():
    header()
    which = sys.argv[1:] or ["spread", "relextract", "srl", "ratios"]
    loaded = "load" in which
    siblings = []
    if "spread" in which:
        spread("idle")
    if loaded:
        ctx = multiprocessing.get_context("spawn")
        siblings = [ctx.Process(target=_busy, args=(900,)) for _ in range(2)]
        for p in siblings:
            p.start()
        time.sleep(1)
        if "spread" in which:
            spread("2 siblings")
    label = "loaded" if siblings else "idle"
    try:
        if "relextract" in which:
            relextract_ladder(label)
        if "srl" in which:
            srl(label)
        if "ratios" in which:
            ratio_trials(label)
    finally:
        for p in siblings:
            p.terminate()


if __name__ == "__main__":
    main()
