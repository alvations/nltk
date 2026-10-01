"""Runner diagnostics for the suite's CPU-time scaling tests.

Measures, on the machine it runs on:
  1. the spread of CPU time per fixed unit of work, idle and beside two busy
     siblings (xdist-like load): a bimodal spread means heterogeneous cores
     or steal counted as guest CPU time;
  2. the relextract linear op at a ladder of sizes, idle and under load;
  3. the CoNLL SRL fixed sink and the pre-fix rescan on the current grid
     design (words == predicates) and on a constant-words design.
Prints plain text; nothing is asserted.
"""

import multiprocessing
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.environ.get("DIAG_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, ROOT)


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
    cpu = wall = float("inf")
    for _ in range(reps):
        c, w = cpu_and_wall(fn, *args)
        cpu, wall = min(cpu, c), min(wall, w)
    return cpu, wall


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
            "hw.perflevel0.l2cachesize",
            "hw.perflevel1.l2cachesize",
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
    for _ in range(400_000):
        x += 1


def spread(label, chunks=150):
    cpus, walls = [], []
    for _ in range(chunks):
        c, w = cpu_and_wall(_unit)
        cpus.append(c)
        walls.append(w)
    cpus.sort()
    walls.sort()

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    print(
        "[spread %-12s] cpu/unit  min %.4f p10 %.4f p50 %.4f p90 %.4f max %.4f"
        % (label, cpus[0], pct(cpus, 0.1), pct(cpus, 0.5), pct(cpus, 0.9), cpus[-1])
    )
    print(
        "[spread %-12s] wall/unit min %.4f p10 %.4f p50 %.4f p90 %.4f max %.4f"
        % (label, walls[0], pct(walls, 0.1), pct(walls, 0.5), pct(walls, 0.9), walls[-1])
    )
    print(
        "[spread %-12s] cpu max/min %.2fx   cpu p90/p10 %.2fx   wall p50/cpu p50 %.2fx"
        % (
            label,
            cpus[-1] / cpus[0],
            pct(cpus, 0.9) / pct(cpus, 0.1),
            pct(walls, 0.5) / pct(cpus, 0.5),
        )
    )


# ---- 2. relextract -----------------------------------------------------------
def relextract_ladder(label):
    from nltk.sem.relextract import semi_rel2reldict, tree2semi_rel
    from nltk.test.unit.test_attack_relextract_expanded import _chunk_tree

    rng = random.Random(0)

    def op(n):
        pairs = tree2semi_rel(_chunk_tree(rng, n))
        assert len(semi_rel2reldict(pairs)) == max(0, len(pairs) - 2)

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
                res[b][0] / res[a][0],
                res[b][0] / max(res[a][0], 0.1),
                res[b][1] / res[a][1],
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


def srl(label):
    from nltk.corpus.reader.conll import ConllCorpusReader
    from nltk.test.unit.test_attack_corpus_readers_expanded import (
        _reference_srl_instances,
    )

    box = tempfile.mkdtemp()
    open(os.path.join(box, "srl.conll"), "w").close()
    reader = ConllCorpusReader(box, ["srl.conll"], ("words", "pos", "tree", "srl"))

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
                    res[b][0] / res[a][0],
                    res[b][0] / max(res[a][0], 0.1),
                    res[b][1] / res[a][1],
                    res[b][1] / max(res[a][1], 0.1),
                )
            )

    run(
        "square",
        {n: _grid_square(n) for n in (70, 140, 280)},
        [(70, 280), (70, 140), (140, 280)],
    )
    for words in (400, 600):
        run(
            "W%d" % words,
            {
                "P%d" % p: _grid_const_words(p, words)
                for p in (35, 70, 140, 280)
                if p <= words
            },
            [("P35", "P140"), ("P70", "P280"), ("P70", "P140"), ("P140", "P280")],
        )


def main():
    header()
    which = sys.argv[1:] or ["spread", "relextract", "srl"]
    if "spread" in which:
        spread("idle")
    siblings = []
    if "load" in which or not sys.argv[1:]:
        ctx = multiprocessing.get_context("spawn")
        siblings = [ctx.Process(target=_busy, args=(600,)) for _ in range(2)]
        for p in siblings:
            p.start()
        time.sleep(1)
        if "spread" in which:
            spread("2 siblings")
    try:
        if "relextract" in which:
            relextract_ladder("loaded" if siblings else "idle")
        if "srl" in which:
            srl("loaded" if siblings else "idle")
    finally:
        for p in siblings:
            p.terminate()
    if siblings and "relextract" in which:
        relextract_ladder("idle")


if __name__ == "__main__":
    main()
