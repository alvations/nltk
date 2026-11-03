"""Chokepoint cost benchmark: time and memory of real NLTK workloads per tree.

Probe-branch tooling only (never part of an upstream PR). Usage:

    python chokepoint_bench.py drive --tree NAME=DIR ... --out results.json
    python chokepoint_bench.py child --tree DIR --workload W --variant V --mode M

Each (workload, tree, variant, mode) runs in a fresh child process that puts
DIR first on sys.path, so `import nltk` loads that tree. Time mode reports the
minimum CPU time (time.process_time) over --repeat runs after one warm-up;
mem mode reports the tracemalloc peak of one run and the growth of the
process's peak RSS across it. The driver rotates the tree order per workload
so the trees run back to back in the same conditions.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))


def _peak_rss():
    """Peak resident set size of this process in bytes."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024
    except ImportError:
        import psutil

        return psutil.Process().memory_info().peak_wset


def _digest(value):
    return hashlib.sha256(repr(value).encode("utf-8", "surrogatepass")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Workloads. Each builder returns {"tree": fn, "stdlib": fn or None}; the setup
# work (reading corpora, building inputs) happens in the builder, untimed.
# --------------------------------------------------------------------------

W = {}


def workload(name, group):
    def deco(fn):
        W[name] = (group, fn)
        return fn

    return deco


def _gutenberg(fid):
    from nltk.corpus import gutenberg

    return gutenberg.raw(fid)


PUNKT_FILES = [
    "austen-emma.txt",
    "austen-persuasion.txt",
    "carroll-alice.txt",
    "chesterton-thursday.txt",
    "melville-moby_dick.txt",
    "shakespeare-hamlet.txt",
]


def _brown_sents(n, categories="news"):
    from nltk.corpus import brown

    return [" ".join(s) for s in brown.sents(categories=categories)[:n]]


@workload("redos.sent_tokenize_files", "redos")
def _w(_):
    from nltk.tokenize import sent_tokenize

    texts = [_gutenberg(f) for f in PUNKT_FILES]
    return {"tree": lambda: [len(sent_tokenize(t)) for t in texts], "stdlib": None}


@workload("redos.sent_tokenize_paras", "redos")
def _w(_):
    from nltk.tokenize import sent_tokenize

    paras = [p for p in _gutenberg("austen-emma.txt").split("\n\n") if p.strip()]
    return {"tree": lambda: [sent_tokenize(p) for p in paras], "stdlib": None}


@workload("redos.word_tokenize_sents", "redos")
def _w(_):
    from nltk.tokenize import word_tokenize

    sents = _brown_sents(3000)
    return {"tree": lambda: [word_tokenize(s) for s in sents], "stdlib": None}


@workload("redos.treebank_word", "redos")
def _w(_):
    from nltk.tokenize import TreebankWordTokenizer

    sents = _brown_sents(3000)
    tok = TreebankWordTokenizer()
    return {"tree": lambda: [tok.tokenize(s) for s in sents], "stdlib": None}


@workload("redos.regexp_family", "redos")
def _w(_):
    from nltk.tokenize import (
        BlanklineTokenizer,
        RegexpTokenizer,
        WhitespaceTokenizer,
        wordpunct_tokenize,
    )

    text = _gutenberg("austen-emma.txt")
    ws, bl = WhitespaceTokenizer(), BlanklineTokenizer()
    rx = RegexpTokenizer(r"\w+|[^\w\s]+")

    def run():
        return (
            wordpunct_tokenize(text),
            ws.tokenize(text),
            bl.tokenize(text),
            rx.tokenize(text),
            list(ws.span_tokenize(text)),
        )

    return {"tree": run, "stdlib": None}


@workload("redos.regexp_chunk", "redos")
def _w(_):
    from nltk import RegexpParser
    from nltk.corpus import conll2000

    sents = conll2000.tagged_sents("test.txt")[:1000]
    cp = RegexpParser("NP: {<DT>?<JJ>*<NN.*>+}\nPP: {<IN><NP>}\nVP: {<VB.*><NP|PP>*}")
    return {"tree": lambda: [str(cp.parse(s)) for s in sents], "stdlib": None}


@workload("redos.tweet_tokenize", "redos")
def _w(_):
    from nltk.corpus import twitter_samples
    from nltk.tokenize import TweetTokenizer

    tweets = twitter_samples.strings("tweets.20150430-223406.json")[:3000]
    tok = TweetTokenizer()
    return {"tree": lambda: [tok.tokenize(t) for t in tweets], "stdlib": None}


@workload("redos.toktok", "redos")
def _w(_):
    from nltk.tokenize import ToktokTokenizer

    sents = _brown_sents(3000)
    tok = ToktokTokenizer()
    return {"tree": lambda: [tok.tokenize(s) for s in sents], "stdlib": None}


@workload("redos.pos_tag", "redos")
def _w(_):
    from nltk import pos_tag
    from nltk.corpus import brown

    sents = [list(s) for s in brown.sents(categories="news")[:1000]]
    pos_tag(["warm", "up"])
    return {"tree": lambda: [pos_tag(s) for s in sents], "stdlib": None}


@workload("redos.match_micro", "redos")
def _w(_):
    import re

    from nltk import redos

    pat, s = r"^-?[\.,]?\d[\d,\.-]*\.?$", "fine."
    rx, std = redos.compile(pat), re.compile(pat)
    n = 100_000

    def run():
        m = rx.match
        for _ in range(n):
            m(s)
        return n

    def run_std():
        m = std.match
        for _ in range(n):
            m(s)
        return n

    return {"tree": run, "stdlib": run_std}


@workload("redos.clock_micro", "redos")
def _w(_):
    import regex

    pat, s = r"^-?[\.,]?\d[\d,\.-]*\.?$", "fine."
    rx = regex.compile(pat)
    n = 100_000

    def run():
        m = rx.match
        for _ in range(n):
            m(s, timeout=5.0)
        return n

    def run_plain():
        m = rx.match
        for _ in range(n):
            m(s)
        return n

    return {"tree": run, "stdlib": run_plain}


@workload("redos.helper_micro", "redos")
def _w(_):
    import re

    from nltk import redos

    n = 50_000
    if not hasattr(redos, "sub"):
        return None

    def run():
        for _ in range(n):
            redos.sub(r"\s+", " ", "a  b")
        return n

    def run_std():
        for _ in range(n):
            re.sub(r"\s+", " ", "a  b")
        return n

    return {"tree": run, "stdlib": run_std}


@workload("corpus.treebank_parsed", "redos")
def _w(_):
    from nltk.corpus import treebank

    fids = treebank.fileids()[:60]
    return {
        "tree": lambda: [str(t) for f in fids for t in treebank.parsed_sents(f)],
        "stdlib": None,
    }


@workload("corpus.brown_tagged", "redos")
def _w(_):
    from nltk.corpus import brown

    return {
        "tree": lambda: list(brown.tagged_words(categories=["news", "editorial"])),
        "stdlib": None,
    }


@workload("corpus.conll2000_chunked", "redos")
def _w(_):
    from nltk.corpus import conll2000

    return {
        "tree": lambda: [str(t) for t in conll2000.chunked_sents("test.txt")],
        "stdlib": None,
    }


@workload("corpus.gutenberg_words", "redos")
def _w(_):
    from nltk.corpus import gutenberg

    fids = ["austen-emma.txt", "carroll-alice.txt", "shakespeare-hamlet.txt"]
    return {"tree": lambda: [list(gutenberg.words(f)) for f in fids], "stdlib": None}


# ------------------------------------------------------------------ JSON


def _perceptron_files(lang):
    import nltk

    base = nltk.data.find(f"taggers/averaged_perceptron_tagger_{lang}/")
    return [
        os.path.join(str(base), f"averaged_perceptron_tagger_{lang}.{a}.json")
        for a in ("weights", "tagdict", "classes")
    ]


def _json_stdlib(paths):
    def run():
        out = []
        for p in paths:
            with open(p, "rb") as f:
                out.append(json.load(f))
        return [len(x) for x in out]

    return run


@workload("json.perceptron_eng", "json")
def _w(_):
    from nltk.tag import PerceptronTagger

    return {
        "tree": lambda: len(PerceptronTagger().model.weights),
        "stdlib": _json_stdlib(_perceptron_files("eng")),
    }


@workload("json.perceptron_rus", "json")
def _w(_):
    from nltk.tag import PerceptronTagger

    return {
        "tree": lambda: len(PerceptronTagger(lang="rus").model.weights),
        "stdlib": _json_stdlib(_perceptron_files("rus")),
    }


@workload("json.twitter_strings", "json")
def _w(_):
    import nltk
    from nltk.corpus import twitter_samples

    path = str(nltk.data.find("corpora/twitter_samples/negative_tweets.json"))

    def run_std():
        with open(path, "rb") as f:
            return [json.loads(line)["text"] for line in f]

    return {
        "tree": lambda: twitter_samples.strings("negative_tweets.json"),
        "stdlib": run_std,
    }


@workload("json.safe_loads_weights", "json")
def _w(_):
    from nltk import jsontags

    if not hasattr(jsontags, "safe_json_loads"):
        return None
    path = _perceptron_files("eng")[0]
    with open(path, "rb") as f:
        raw = f.read()
    return {
        "tree": lambda: len(jsontags.safe_json_loads(raw)),
        "stdlib": lambda: len(json.loads(raw)),
    }


# ------------------------------------------------------------------ pickle


def _plain_object():
    return {
        f"w{i}": [i, i / 7.0, f"tag{i % 40}", (i, "x" * (i % 5)), None, i % 2 == 0]
        for i in range(150_000)
    }


@workload("pickle.restricted_plain", "pickle")
def _w(_):
    import io
    import pickle

    from nltk.picklesec import RestrictedUnpickler

    data = pickle.dumps(_plain_object(), protocol=4)
    return {
        "tree": lambda: len(RestrictedUnpickler(io.BytesIO(data)).load()),
        "stdlib": lambda: len(pickle.loads(data)),
    }


@workload("pickle.restricted_proto2", "pickle")
def _w(_):
    import io
    import pickle

    from nltk.picklesec import RestrictedUnpickler

    data = pickle.dumps(_plain_object(), protocol=2)
    return {
        "tree": lambda: len(RestrictedUnpickler(io.BytesIO(data)).load()),
        "stdlib": lambda: len(pickle.loads(data)),
    }


@workload("pickle.data_load", "pickle")
def _w(_):
    import pickle

    import nltk

    root = os.environ["NLTK_DATA"]
    os.makedirs(os.path.join(root, "bench"), exist_ok=True)
    path = os.path.join(root, "bench", "plain.pickle")
    data = pickle.dumps(_plain_object(), protocol=4)
    if not os.path.exists(path) or open(path, "rb").read() != data:
        with open(path, "wb") as f:
            f.write(data)
        os.chmod(path, 0o600)

    def run_std():
        with open(path, "rb") as f:
            return len(pickle.load(f))

    return {
        "tree": lambda: len(nltk.data.load("bench/plain.pickle", cache=False)),
        "stdlib": run_std,
    }


@workload("pickle.allowlist_model", "pickle")
def _w(_):
    import io
    import pickle
    from collections import defaultdict

    from nltk import picklesec

    counts = defaultdict(int)
    for i in range(200_000):
        counts[(f"w{i % 5000}", f"t{i % 45}")] += i
    data = pickle.dumps(counts, protocol=4)
    allowed = {("collections", "defaultdict"), ("builtins", "int")}

    def run():
        return len(
            picklesec.allowlisted_pickle_load(io.BytesIO(data), allowed_globals=allowed)
        )

    return {"tree": run, "stdlib": lambda: len(pickle.loads(data))}


# ------------------------------------------------------------------ pathsec


def _reader_open(reader, fileids):
    def run():
        total = 0
        for f in fileids:
            with reader.open(f) as s:
                total += len(s.read())
        return total

    return run


def _builtin_open(reader, fileids):
    paths = [str(reader.abspath(f)) for f in fileids]

    def run():
        total = 0
        for p in paths:
            with open(p, "rb") as s:
                total += len(s.read())
        return total

    return run


@workload("pathsec.open_brown", "pathsec")
def _w(_):
    from nltk.corpus import brown

    fids = brown.fileids()
    return {"tree": _reader_open(brown, fids), "stdlib": _builtin_open(brown, fids)}


@workload("pathsec.open_treebank", "pathsec")
def _w(_):
    from nltk.corpus import treebank

    fids = treebank.fileids()
    return {
        "tree": _reader_open(treebank, fids),
        "stdlib": _builtin_open(treebank, fids),
    }


@workload("pathsec.validate_path_micro", "pathsec")
def _w(_):
    from nltk import pathsec
    from nltk.corpus import brown

    paths = [str(brown.abspath(f)) for f in brown.fileids()] * 4
    return {
        "tree": lambda: [pathsec.validate_path(p) for p in paths],
        "stdlib": lambda: [os.path.realpath(p) for p in paths],
    }


@workload("pathsec.open_micro", "pathsec")
def _w(_):
    from nltk import pathsec
    from nltk.corpus import gutenberg

    paths = [str(gutenberg.abspath(f)) for f in gutenberg.fileids()] * 40

    def run():
        for p in paths:
            pathsec.open(p, "rb").close()
        return len(paths)

    def run_std():
        for p in paths:
            open(p, "rb").close()
        return len(paths)

    return {"tree": run, "stdlib": run_std}


@workload("pathsec.data_find", "pathsec")
def _w(_):
    import nltk

    def run():
        for _ in range(20):
            nltk.data.find("corpora/gutenberg")
            nltk.data.find("corpora/treebank/combined/wsj_0001.mrg")
            nltk.data.find("tokenizers/punkt_tab/english/")
        return 60

    return {"tree": run, "stdlib": None}


@workload("pathsec.punkt_load_langs", "pathsec")
def _w(_):
    from nltk.tokenize.punkt import PunktTokenizer

    langs = ["english", "german", "french", "spanish", "italian", "dutch"]
    return {
        "tree": lambda: [len(PunktTokenizer(l)._params.abbrev_types) for l in langs],
        "stdlib": None,
    }


# ------------------------------------------------------------------ xmlsec


def _brown_tei_paths():
    import nltk

    base = str(nltk.data.find("corpora/brown_tei/"))
    return sorted(os.path.join(base, f) for f in os.listdir(base) if f.endswith(".xml"))


@workload("xml.brown_tei_files", "xml")
def _w(_):
    from xml.etree import ElementTree

    from nltk import xmlsec

    paths = _brown_tei_paths()

    def run():
        return sum(len(list(xmlsec.parse(p).iter())) for p in paths)

    def run_std():
        return sum(len(list(ElementTree.parse(p).iter())) for p in paths)

    return {"tree": run, "stdlib": run_std}


@workload("xml.brown_tei_concat", "xml")
def _w(_):
    from xml.etree import ElementTree

    from nltk import xmlsec

    parts = []
    for p in _brown_tei_paths():
        with open(p, "rb") as f:
            part = f.read()
        if part.startswith(b"<?xml"):
            part = part.split(b"?>", 1)[1]
        parts.append(part)
    doc = b"<corpus>" + b"".join(parts) + b"</corpus>"
    return {
        "tree": lambda: len(list(xmlsec.fromstring(doc).iter())),
        "stdlib": lambda: len(list(ElementTree.fromstring(doc).iter())),
    }


@workload("xml.brown_tei_defused_only", "xml")
def _w(_):
    from xml.etree import ElementTree

    try:
        from defusedxml.ElementTree import parse as defused_parse
    except ImportError:
        return None
    paths = _brown_tei_paths()
    return {
        "tree": lambda: sum(len(list(defused_parse(p).iter())) for p in paths),
        "stdlib": lambda: sum(len(list(ElementTree.parse(p).iter())) for p in paths),
    }


@workload("xml.brown_tei_screen_cparse", "xml")
def _w(_):
    from xml.etree import ElementTree

    from nltk import xmlsec

    if not hasattr(xmlsec, "_check_tree"):
        return None
    docs = []
    for p in _brown_tei_paths():
        with open(p, "rb") as f:
            docs.append(f.read())

    def run():
        total = 0
        for d in docs:
            xmlsec._screen(d)
            root = ElementTree.fromstring(d)
            xmlsec._check_tree(root)
            total += len(list(root.iter()))
        return total

    return {
        "tree": run,
        "stdlib": lambda: sum(
            len(list(ElementTree.fromstring(d).iter())) for d in docs
        ),
    }


@workload("xml.fromstring_small", "xml")
def _w(_):
    from xml.etree import ElementTree

    from nltk import xmlsec

    docs = [b"<doc n='%d'><a k='v'>t</a><b/></doc>" % i for i in range(3000)]
    return {
        "tree": lambda: [xmlsec.fromstring(d).tag for d in docs],
        "stdlib": lambda: [ElementTree.fromstring(d).tag for d in docs],
    }


# ------------------------------------------------------------------ termsec


class _Sink:
    def write(self, s):
        return len(s)

    def flush(self):
        pass


def _udhr_tokens(n):
    from nltk.corpus import udhr

    toks = []
    for f in (
        "German_Deutsch-Latin1",
        "French_Francais-Latin1",
        "Russian-UTF8",
        "Greek_Ellinika-UTF8",
        "Japanese_Nihongo-UTF8",
        "Arabic_Alarabia-Arabic",
    ):
        try:
            toks.extend(udhr.words(f))
        except Exception:
            continue
    return (toks * (n // max(1, len(toks)) + 1))[:n]


@workload("termsec.print_ascii", "termsec")
def _w(_):
    from nltk.corpus import brown

    words = list(brown.words(categories="news"))[:100_000]
    sink = _Sink()
    try:
        from nltk.termsec import safe_print
    except ImportError:
        return None

    def run():
        for w in words:
            safe_print(w, file=sink)
        return len(words)

    def run_std():
        for w in words:
            print(w, file=sink)
        return len(words)

    return {"tree": run, "stdlib": run_std}


@workload("termsec.print_nonascii", "termsec")
def _w(_):
    words = _udhr_tokens(100_000)
    sink = _Sink()
    try:
        from nltk.termsec import safe_print
    except ImportError:
        return None

    def run():
        for w in words:
            safe_print(w, file=sink)
        return len(words)

    def run_std():
        for w in words:
            print(w, file=sink)
        return len(words)

    return {"tree": run, "stdlib": run_std}


@workload("termsec.sanitize_large", "termsec")
def _w(_):
    try:
        from nltk.termsec import sanitize_terminal
    except ImportError:
        return None
    ascii_text = _gutenberg("austen-emma.txt")
    uni_text = " ".join(_udhr_tokens(200_000))
    return {
        "tree": lambda: (
            len(sanitize_terminal(ascii_text)),
            len(sanitize_terminal(uni_text)),
        ),
        "stdlib": lambda: (len(str(ascii_text)), len(str(uni_text))),
    }


@workload("csvsec.write_rows", "termsec")
def _w(_):
    import csv

    from nltk.corpus import brown

    try:
        from nltk.csvsec import SafeCsvWriter
    except ImportError:
        return None
    rows = list(brown.tagged_words(categories="news"))[:60_000]
    rows += [(w, "X") for w in _udhr_tokens(40_000)]

    def run():
        SafeCsvWriter(_Sink()).writerows(rows)
        return len(rows)

    def run_std():
        csv.writer(_Sink()).writerows(rows)
        return len(rows)

    return {"tree": run, "stdlib": run_std}


# ------------------------------------------------------------------ data.py


@workload("data.udhr_raw_all", "data")
def _w(_):
    from nltk.corpus import udhr

    fids = udhr.fileids()

    def run():
        out = 0
        for f in fids:
            try:
                out += len(udhr.raw(f))
            except Exception as exc:  # a codec the tree refuses
                out += len(type(exc).__name__)
        return out

    return {"tree": run, "stdlib": None}


@workload("data.seekable_lines", "data")
def _w(_):
    from nltk.corpus import gutenberg
    from nltk.data import SeekableUnicodeStreamReader

    path = str(gutenberg.abspath("austen-emma.txt"))
    with open(path, "rb") as f:
        raw = f.read()

    def run():
        import io

        r = SeekableUnicodeStreamReader(io.BytesIO(raw), "utf8")
        n = 0
        while True:
            pos = r.tell()
            line = r.readline()
            if not line:
                return n, pos
            n += 1

    def run_std():
        import io

        return sum(1 for _ in io.TextIOWrapper(io.BytesIO(raw), "utf8")), 0

    return {"tree": run, "stdlib": run_std}


# --------------------------------------------------------------------------


def child(args):
    sys.path.insert(0, args.tree)
    import nltk

    assert os.path.realpath(os.path.dirname(os.path.dirname(nltk.__file__))) == (
        os.path.realpath(args.tree)
    ), nltk.__file__
    group, builder = W[args.workload]
    t0 = time.process_time()
    try:
        fns = builder(None)
    except Exception as exc:
        print(json.dumps({"setup_error": f"{type(exc).__name__}: {exc}"[:300]}))
        return
    setup_cpu = time.process_time() - t0
    if fns is None or fns.get(args.variant) is None:
        print(json.dumps({"skipped": True}))
        return
    fn = fns[args.variant]
    out = {"setup_cpu": round(setup_cpu, 4), "nltk": nltk.__version__}
    try:
        if args.mode == "time":
            result = fn()  # warm-up, and the output we digest
            out["digest"] = _digest(result)
            cpu, wall = [], []
            for _ in range(args.repeat):
                gc.collect()
                c0, w0 = time.process_time(), time.perf_counter()
                fn()
                cpu.append(time.process_time() - c0)
                wall.append(time.perf_counter() - w0)
            out.update(cpu_min=min(cpu), cpu_all=cpu, wall_min=min(wall))
        else:
            gc.collect()
            rss0 = _peak_rss()
            tracemalloc.start()
            fn()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            out.update(tracemalloc_peak=peak, rss_peak_growth=_peak_rss() - rss0)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    print(json.dumps(out))


def run_child(tree, workload_name, variant, mode, repeat, timeout):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "child",
        "--tree",
        tree,
        "--workload",
        workload_name,
        "--variant",
        variant,
        "--mode",
        mode,
        "--repeat",
        str(repeat),
    ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=HERE
        )
    except subprocess.TimeoutExpired:
        return {"error": f"child exceeded {timeout}s"}
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        return {"error": f"rc={proc.returncode}: {proc.stderr[-400:]}"}
    return json.loads(lines[-1])


def _load_average():
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


def drive(args):
    trees = dict(t.split("=", 1) for t in args.tree)
    names = [n for n in W if not args.only or any(o in n for o in args.only)]
    results = {
        "platform": platform.platform(),
        "python": sys.version,
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "trees": trees,
        "load_start": _load_average(),
        "rows": [],
    }
    order = list(trees)
    for i, name in enumerate(names):
        rot = order[i % len(order) :] + order[: i % len(order)]
        variants = [(t, "tree") for t in rot] + [(order[0], "stdlib")]
        for tname, variant in variants:
            row = {
                "workload": name,
                "group": W[name][0],
                "tree": tname,
                "variant": variant,
                "load": _load_average(),
            }
            row["time"] = run_child(
                trees[tname], name, variant, "time", args.repeat, args.timeout
            )
            if row["time"].get("skipped"):
                continue
            row["mem"] = run_child(trees[tname], name, variant, "mem", 1, args.timeout)
            label = tname if variant == "tree" else "stdlib"
            print(f"{name:30s} {label:8s} " + _fmt(row), flush=True)
            results["rows"].append(row)
    results["load_end"] = _load_average()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print_table(results)


def _fmt(row):
    t, m = row["time"], row.get("mem", {})
    if "error" in t or "setup_error" in t:
        return "ERROR " + (t.get("error") or t.get("setup_error"))[:120]
    return (
        f"cpu {t['cpu_min']:8.3f}s  wall {t['wall_min']:8.3f}s  "
        f"tm_peak {m.get('tracemalloc_peak', 0) / 2**20:8.1f}MiB  "
        f"rss+ {m.get('rss_peak_growth', 0) / 2**20:7.1f}MiB  "
        f"digest {t.get('digest')}"
    )


def print_table(results):
    rows = results["rows"]
    print(f"\n{results['platform']} | Python {results['python'].split()[0]}")
    trees = list(results["trees"])
    head = "| workload | " + " | ".join(f"{t} cpu s" for t in trees)
    head += " | stdlib cpu s | " + " | ".join(f"{t} tm MiB" for t in trees)
    head += " | stdlib tm MiB | digests equal |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    by = {}
    for r in rows:
        key = r["tree"] if r["variant"] == "tree" else "stdlib"
        by.setdefault(r["workload"], {})[key] = r
    for name, cells in by.items():

        def cpu(k):
            t = cells.get(k, {}).get("time", {})
            return f"{t['cpu_min']:.3f}" if "cpu_min" in t else ("ERR" if t else "-")

        def mem(k):
            m = cells.get(k, {}).get("mem", {})
            return (
                f"{m['tracemalloc_peak'] / 2**20:.1f}"
                if "tracemalloc_peak" in m
                else "-"
            )

        digests = {
            cells[t]["time"].get("digest")
            for t in trees
            if t in cells and t != "v3103" and t != "v3104"
        }
        line = f"| {name} | " + " | ".join(cpu(t) for t in trees)
        line += f" | {cpu('stdlib')} | " + " | ".join(mem(t) for t in trees)
        line += f" | {mem('stdlib')} | {'yes' if len(digests) == 1 else 'NO'} |"
        print(line)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("child")
    c.add_argument("--tree", required=True)
    c.add_argument("--workload", required=True)
    c.add_argument("--variant", default="tree")
    c.add_argument("--mode", default="time")
    c.add_argument("--repeat", type=int, default=3)
    d = sub.add_parser("drive")
    d.add_argument("--tree", action="append", required=True)
    d.add_argument("--only", nargs="*")
    d.add_argument("--repeat", type=int, default=3)
    d.add_argument("--timeout", type=int, default=900)
    d.add_argument("--out", default="bench.json")
    t = sub.add_parser("table")
    t.add_argument("json")
    args = p.parse_args()
    if args.cmd == "child":
        child(args)
    elif args.cmd == "drive":
        drive(args)
    else:
        with open(args.json) as f:
            print_table(json.load(f))


if __name__ == "__main__":
    main()
