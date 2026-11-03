"""Punkt on the ten longest English Wikipedia articles (Special:LongPages, 2026-11-03).

Usage:
  python probe_4009/wiki.py fetch <data_dir>
  python probe_4009/wiki.py run <data_dir> <label>=<nltk_dir> ... > results.md

fetch: downloads each pinned revision's wikitext (checked against its sha256) and
its rendered text (all visible text, tables included). run: tokenizes every page,
all ten concatenated, in both forms, with each nltk tree in a fresh process; the
label "reference" runs develop with the timeout disabled, the output every other
tree must reproduce.
"""

import hashlib
import html.parser
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "nltk-punkt-long-text-probe/1.0 (https://github.com/nltk/nltk/issues/4009)"
HERE = Path(__file__).resolve().parent
BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br", "table",
         "ul", "ol", "dd", "dt", "caption"}

CHILD = r"""
import hashlib, json, sys, time
sys.path.insert(0, sys.argv[1])
import nltk
from nltk.tokenize import sent_tokenize
if sys.argv[3] == "1":
    from nltk import redos
    redos.DEFAULT_TIMEOUT = None
text = open(sys.argv[2], encoding="utf-8").read()
t = time.process_time()
try:
    sents = sent_tokenize(text)
    out = {"status": "ok", "sents": len(sents),
           "sha256": hashlib.sha256("\n\x00".join(sents).encode()).hexdigest()}
except TimeoutError:
    out = {"status": "TimeoutError"}
out["cpu"] = round(time.process_time() - t, 2)
out["nltk"] = nltk.__version__
print(json.dumps(out))
"""


def get(params):
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception as e:
            print("retry", attempt, e, file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    raise SystemExit("fetch failed")


class Visible(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self.skip += 1
        elif tag in BLOCK:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append("\t")

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self.skip -= 1
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def fetch(data):
    data.mkdir(parents=True, exist_ok=True)
    pages = json.loads((HERE / "pages.json").read_text())["pages"]
    for p in pages:
        rev = get({"action": "query", "prop": "revisions", "revids": p["revid"],
                   "rvprop": "content", "rvslots": "main"})["query"]["pages"][0]["revisions"][0]
        wikitext = rev["slots"]["main"]["content"]
        digest = hashlib.sha256(wikitext.encode()).hexdigest()
        if digest != p["wikitext_sha256"]:
            raise SystemExit(f"sha256 mismatch for {p['title']}: {digest}")
        parsed = get({"action": "parse", "oldid": p["revid"], "prop": "text"})["parse"]["text"]
        v = Visible()
        v.feed(parsed)
        (data / f"{p['rank']:02d}.wikitext.txt").write_text(wikitext, encoding="utf-8")
        (data / f"{p['rank']:02d}.rendered.txt").write_text("".join(v.out), encoding="utf-8")
        time.sleep(1)
    for kind in ("wikitext", "rendered"):
        joined = "\n\n".join((data / f"{p['rank']:02d}.{kind}.txt").read_text(encoding="utf-8")
                             for p in pages)
        (data / f"all.{kind}.txt").write_text(joined, encoding="utf-8")
    print("fetched and verified", len(pages), "pages")


def run(data, trees):
    pages = json.loads((HERE / "pages.json").read_text())["pages"]
    names = [f"{p['rank']:02d}" for p in pages] + ["all"]
    titles = {f"{p['rank']:02d}": p["title"] for p in pages} | {"all": "all ten concatenated"}
    rows, bad = [], []
    for name in names:
        for kind in ("rendered", "wikitext"):
            path = data / f"{name}.{kind}.txt"
            chars = len(path.read_text(encoding="utf-8"))
            res = {}
            for label, tree in trees:
                unbounded = "1" if label == "reference" else "0"
                out = subprocess.run([sys.executable, "-I", "-c", CHILD, tree, str(path), unbounded],
                                     capture_output=True, text=True)
                res[label] = json.loads(out.stdout) if out.returncode == 0 else {
                    "status": "crash", "cpu": None, "err": out.stderr[-300:]}
            ref = res["reference"]
            for label, r in res.items():
                if label in ("reference", "3.10.3"):
                    continue
                if r["status"] == "ok" and r["sha256"] != ref.get("sha256"):
                    bad.append(f"{label} output differs from reference on {name}.{kind}")
            if res.get("branch", {}).get("status") != "ok":
                bad.append(f"branch did not complete {name}.{kind}: {res.get('branch')}")
            rows.append((name, titles[name], kind, chars, res))
    labels = [label for label, _ in trees]
    print("| page | form | chars | " + " | ".join(labels) + " |")
    print("|---|---|---|" + "---|" * len(labels))
    for name, title, kind, chars, res in rows:
        cells = []
        for label in labels:
            r = res[label]
            cells.append(f"{r['status']} {r['cpu']}s" + (f" ({r['sents']} sents)" if r["status"] == "ok" else ""))
        print(f"| {name} {title} | {kind} | {chars:,} | " + " | ".join(cells) + " |")
    print()
    print("\n".join(f"FAIL: {b}" for b in bad) or "PASS: branch completes every text with the reference output")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(Path(sys.argv[2]))
    else:
        trees = [tuple(a.split("=", 1)) for a in sys.argv[3:]]
        sys.exit(run(Path(sys.argv[2]), trees))
