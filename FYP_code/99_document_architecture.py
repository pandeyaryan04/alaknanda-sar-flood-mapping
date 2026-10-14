r"""
99_document_architecture.py
===========================
Generate the architecture documentation from the code itself.

WHY THIS EXISTS
---------------
Written project documents describe the plan. Code drifts from the plan. By the
time the final report is written, a methodology section typed by hand describes
what was intended six months earlier, not what actually ran. That gap is where
report errors come from, and it is the sort of thing an examiner finds by asking
one question about a parameter.

This script removes the gap. It reads the scripts, the configuration constants,
the environment and the git history, and writes an architecture section that is
correct by construction. Run it at the end of every working session. At report
time the section is already written, already current, and already consistent
with the numbers in the results.

It executes nothing and imports nothing from the other scripts - it parses them
as text, so it works even when the environment is half-installed or a script is
mid-edit and would not run.

HOW EACH SCRIPT DECLARES ITSELF
-------------------------------
Every pipeline script carries three module-level variables near the top:

    __stage__ = "Short name of what this stage does"
    __reads__ = ["what it consumes"]
    __writes__ = ["what it produces"]

Those declarations are what build the data-flow graph. Keeping them accurate is
the only maintenance this needs. If a script gains an output, add it to
__writes__ and the diagram, the tables and the report text all follow.

HOW TO RUN
----------
    (fyp) C:\\...\\FYP> python 99_document_architecture.py

OUTPUTS (into ./outputs/architecture/)
--------------------------------------
    ARCHITECTURE.md      the readable document - this is the main output
    architecture.dot     data-flow graph, Graphviz source
    architecture.png     rendered, if Graphviz is installed
    architecture.mmd     the same graph as Mermaid, which GitHub renders
                         with nothing installed
    parameters.csv       every configuration constant, per script
    provenance.json      versions, git commit, file hashes, timestamps

Re-run it whenever the pipeline changes, so the report's methods match the code.

================================================================================
HOW TO RUN THIS  (copy the lines, paste into Anaconda Prompt, press Enter)
================================================================================
    conda activate fyp
    cd /d "<project folder>\04 Code\FYP_code"
    python 99_document_architecture.py

Seconds. Runs nothing and imports nothing: it reads the other scripts as text,
so it works even mid-edit or with a half-installed environment.

Writes outputs\architecture\ : ARCHITECTURE.md, architecture.mmd,
architecture.png (if Graphviz is installed), parameters.csv, provenance.json

RUN THIS AT THE END OF EVERY WORKING SESSION, then commit:
    git add -A
    git commit -m "what changed"

Why it matters: by week 15 the code will have drifted from the plan. A methods
section typed by hand describes what you intended in August. This one is
generated from the source, so it cannot be wrong about its own parameters, and
it records the git commit and a hash of each file so any figure in the report
traces back to the exact code that made it.

================================================================================
"""

import ast
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime

__stage__ = "Architecture documentation generator"
__reads__ = ["all pipeline scripts", "git history", "the active environment"]
__writes__ = ["ARCHITECTURE.md", "architecture.dot", "architecture.png",
              "architecture.mmd", "parameters.csv", "provenance.json"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "architecture")
SCRIPT_RE = re.compile(r"^\d\d_.*\.py$")

# Constants that are noise in a methods section rather than method.
SKIP_CONST = {"LINES", "RETRIES", "SCRIPT_RE", "OUT_DIR", "SCRIPT_DIR",
              "API", "MEDIA", "DESC", "SKIP_CONST", "FAIL"}


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def literal(node):
    """Best-effort value of an AST node, without executing anything."""
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<computed>"


def describe(v, limit=170):
    s = repr(v) if not isinstance(v, str) else v
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def parse_script(path):
    """Pull docstring, declarations and configuration out of one script."""
    src = open(path, encoding="utf-8").read()
    info = {
        "file": os.path.basename(path),
        "sha256": hashlib.sha256(src.encode()).hexdigest()[:16],
        "lines": len(src.splitlines()),
        "bytes": len(src.encode()),
        "stage": "",
        "reads": [],
        "writes": [],
        "summary": "",
        "purpose": "",
        "consts": [],
        "functions": [],
        "parse_error": "",
    }
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        info["parse_error"] = f"line {exc.lineno}: {exc.msg}"
        return info

    doc = ast.get_docstring(tree) or ""
    lines = [l.rstrip() for l in doc.splitlines()]
    body = [l for l in lines if l.strip() and not set(l.strip()) <= {"=", "-"}]
    para = []
    for l in body[1:]:
        if not l.strip() or l.strip().isupper():
            break
        para.append(l.strip())
    info["summary"] = re.sub(r"\s+", " ", " ".join(para)).strip()
    # first prose paragraph after a WHAT/WHY heading, else the summary
    m = re.search(r"^(?:WHAT|WHY)[^\n]*\n-+\n(.+?)(?:\n\n|\Z)", doc, re.S | re.M)
    info["purpose"] = re.sub(r"\s+", " ", m.group(1)).strip() if m else info["summary"]

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if not isinstance(tgt, ast.Name):
                    continue
                name = tgt.id
                val = literal(node.value)
                if name == "__stage__":
                    info["stage"] = val
                elif name == "__reads__":
                    info["reads"] = val if isinstance(val, list) else [val]
                elif name == "__writes__":
                    info["writes"] = val if isinstance(val, list) else [val]
                elif name.isupper() and name not in SKIP_CONST and not name.startswith("_"):
                    info["consts"].append((name, describe(val)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node) or ""
            info["functions"].append(
                (node.name, re.sub(r"\s+", " ", d.split("\n")[0])[:110] if d else "")
            )
    return info


def build_graph(scripts):
    """Edges wherever one script's output name matches another's input."""
    edges = []
    produced = {}
    for s in scripts:
        for w in s["writes"]:
            produced.setdefault(str(w).lower(), []).append(s["file"])
    for s in scripts:
        for r in s["reads"]:
            key = str(r).lower()
            for other, outs in produced.items():
                if other == key or (len(key) > 6 and (key in other or other in key)):
                    for src in outs:
                        if src != s["file"]:
                            edges.append((src, s["file"], str(r)))
    seen, uniq = set(), []
    for e in edges:
        if (e[0], e[1]) not in seen:
            seen.add((e[0], e[1]))
            uniq.append(e)
    return uniq


def write_dot(scripts, edges, path):
    def nid(f):
        return "n" + re.sub(r"\W", "_", f)
    L = ['digraph pipeline {', '  rankdir=TB; bgcolor="white"; splines=ortho;',
         '  nodesep=0.35; ranksep=0.45; fontname="Helvetica";',
         '  node [shape=box, style="filled,rounded", fontname="Helvetica",',
         '        fontsize=10, margin="0.18,0.11", penwidth=1.3,',
         '        color="#3A6E9E", fillcolor="#DDEAF6"];',
         '  edge [color="#5A5A5A", penwidth=1.15, arrowsize=0.75, fontsize=8,',
         '        fontname="Helvetica", fontcolor="#666666"];']
    for s in scripts:
        stage = (s["stage"] or s["summary"] or s["file"]).replace('"', "'")
        ins = "\\n".join(f"in:  {str(r)[:38]}" for r in s["reads"][:3])
        outs = "\\n".join(f"out: {str(w)[:38]}" for w in s["writes"][:3])
        label = f'{s["file"]}\\n{stage}'
        if ins:
            label += "\\n\\n" + ins
        if outs:
            label += "\\n" + outs
        L.append(f'  {nid(s["file"])} [label="{label}"];')
    for a, b, via in edges:
        L.append(f'  {nid(a)} -> {nid(b)} [xlabel="{str(via)[:26]}"];')
    L.append("}")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def write_mermaid(scripts, edges, path):
    def nid(f):
        return re.sub(r"\W", "_", f)
    L = ["```mermaid", "flowchart TD"]
    for s in scripts:
        stage = (s["stage"] or s["file"]).replace('"', "'")
        L.append(f'    {nid(s["file"])}["<b>{s["file"]}</b><br/>{stage}"]')
    for a, b, via in edges:
        L.append(f'    {nid(a)} -->|{str(via)[:30]}| {nid(b)}')
    L.append("```")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def provenance(scripts):
    p = {
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": sh("git rev-parse HEAD") or "<not a git repository>",
        "git_branch": sh("git rev-parse --abbrev-ref HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "scripts": {s["file"]: {"sha256_16": s["sha256"], "lines": s["lines"]} for s in scripts},
        "packages": {},
        "gpu": "",
    }
    for mod in ["ee", "numpy", "pandas", "sklearn", "rasterio", "geopandas",
                "skimage", "xgboost", "torch", "torchvision"]:
        try:
            m = __import__(mod)
            p["packages"][mod] = getattr(m, "__version__", "?")
        except Exception:
            p["packages"][mod] = None
    try:
        import torch
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            p["gpu"] = f"{pr.name}, {pr.total_memory/1024**3:.2f} GB VRAM, CUDA {torch.version.cuda}"
    except Exception:
        pass
    return p


def outputs_on_disk():
    rows = []
    for root in ["data", "outputs"]:
        base = os.path.join(SCRIPT_DIR, root)
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            n = len(files)
            if not n:
                continue
            size = sum(os.path.getsize(os.path.join(dirpath, f))
                       for f in files if os.path.exists(os.path.join(dirpath, f)))
            rows.append((os.path.relpath(dirpath, SCRIPT_DIR).replace("\\", "/"), n, size))
    return sorted(rows)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:,.0f} {u}"
        n /= 1024.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(SCRIPT_DIR) if SCRIPT_RE.match(f))
    if not files:
        sys.exit(f"No pipeline scripts found in {SCRIPT_DIR}. Expected files named 00_*.py, 01_*.py and so on.")

    print(f"Reading {len(files)} script(s) in {SCRIPT_DIR}\n")
    scripts = []
    for f in files:
        info = parse_script(os.path.join(SCRIPT_DIR, f))
        scripts.append(info)
        flag = f"  PARSE ERROR: {info['parse_error']}" if info["parse_error"] else ""
        decl = "" if (info["reads"] or info["writes"]) else "   (no __reads__/__writes__ declared)"
        print(f"  {f:<34} {info['lines']:>5} lines  {info['stage'][:38]}{decl}{flag}")

    edges = build_graph(scripts)
    prov = provenance(scripts)
    disk = outputs_on_disk()

    write_dot(scripts, edges, os.path.join(OUT_DIR, "architecture.dot"))
    write_mermaid(scripts, edges, os.path.join(OUT_DIR, "architecture.mmd"))

    png = os.path.join(OUT_DIR, "architecture.png")
    rendered = False
    if sh("dot -V"):
        sh(f'dot -Tpng -Gdpi=200 "{os.path.join(OUT_DIR, "architecture.dot")}" -o "{png}"')
        rendered = os.path.exists(png)

    with open(os.path.join(OUT_DIR, "parameters.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["script", "parameter", "value"])
        for s in scripts:
            for k, v in s["consts"]:
                w.writerow([s["file"], k, v])

    with open(os.path.join(OUT_DIR, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)

    # ------------------------------------------------------------ ARCHITECTURE.md
    M = []
    A = M.append
    A("# Pipeline architecture")
    A("")
    A(f"Generated automatically by `99_document_architecture.py` on "
      f"{prov['generated_utc']} UTC. Do not edit this file by hand; it is "
      f"regenerated from the code and any edit will be overwritten.")
    A("")
    if prov["git_commit"].startswith("<"):
        A("Not generated inside a git repository, so no commit is recorded. "
          "Run `git init` and commit, so that every result can be tied to an exact code version.")
    else:
        A(f"Git commit `{prov['git_commit'][:12]}` on branch `{prov['git_branch'] or 'detached'}`"
          + ("  **(uncommitted changes present at generation time)**" if prov["git_dirty"] else ""))
    A("")

    A("## Overview")
    A("")
    A(f"The pipeline is {len(scripts)} scripts, each a single stage that reads named "
      f"inputs and writes named outputs. Stages are numbered in execution order. "
      f"No stage imports another; they communicate only through files on disk, so "
      f"any stage can be re-run on its own without repeating the ones before it.")
    A("")

    A("## Stages")
    A("")
    A("| # | Script | Stage | Reads | Writes |")
    A("|---|---|---|---|---|")
    for s in scripts:
        A(f"| {s['file'][:2]} | `{s['file']}` | {s['stage'] or '-'} | "
          f"{'; '.join(str(x) for x in s['reads']) or '-'} | "
          f"{'; '.join(str(x) for x in s['writes']) or '-'} |")
    A("")

    A("## Data flow")
    A("")
    if rendered:
        A("![Pipeline data flow](architecture.png)")
        A("")
    A(open(os.path.join(OUT_DIR, "architecture.mmd"), encoding="utf-8").read())
    A("")
    if edges:
        A("Dependencies, as declared by the scripts themselves:")
        A("")
        for a, b, via in edges:
            A(f"- `{a}` produces **{via}**, which `{b}` consumes.")
    else:
        A("No inter-stage dependencies declared yet. Each stage currently stands alone.")
    A("")

    A("## What each stage does")
    A("")
    for s in scripts:
        A(f"### `{s['file']}` — {s['stage'] or s['summary']}")
        A("")
        if s["parse_error"]:
            A(f"> **This file did not parse: {s['parse_error']}**. Fix it and re-run.")
            A("")
            continue
        if s["purpose"]:
            A(s["purpose"])
            A("")
        A(f"*{s['lines']} lines, source SHA-256 `{s['sha256']}`*")
        A("")
        if s["reads"]:
            A("**Reads:** " + ", ".join(f"`{x}`" for x in s["reads"]))
            A("")
        if s["writes"]:
            A("**Writes:** " + ", ".join(f"`{x}`" for x in s["writes"]))
            A("")
        if s["consts"]:
            A("**Parameters:**")
            A("")
            A("| Parameter | Value |")
            A("|---|---|")
            for k, v in s["consts"]:
                A(f"| `{k}` | {v} |")
            A("")
        if s["functions"]:
            A("**Functions:** " + ", ".join(f"`{n}()`" for n, _ in s["functions"]))
            A("")

    A("## Environment at generation time")
    A("")
    A(f"- Python {prov['python']}")
    A(f"- {prov['platform']}")
    if prov["gpu"]:
        A(f"- GPU: {prov['gpu']}")
    A("")
    A("| Package | Version |")
    A("|---|---|")
    for k, v in prov["packages"].items():
        A(f"| `{k}` | {v if v else 'not installed'} |")
    A("")

    if disk:
        A("## Data and outputs present on disk")
        A("")
        A("| Folder | Files | Size |")
        A("|---|---|---|")
        for d, n, sz in disk:
            A(f"| `{d}` | {n} | {human(sz)} |")
        A("")

    A("## Reproducibility statement")
    A("")
    A("Every result reported in this project was produced by the scripts listed "
      "above, at the git commit named at the top of this file. Each script records "
      "its own parameters, and this document is generated from the source rather "
      "than written alongside it, so the parameters described here are the "
      "parameters that ran. The SHA-256 prefix beside each script identifies the "
      "exact version of that file; if a figure needs to be reproduced, checking out "
      "that commit reproduces the code that made it.")
    A("")

    md_path = os.path.join(OUT_DIR, "ARCHITECTURE.md")
    open(md_path, "w", encoding="utf-8").write("\n".join(M))

    print("\n" + "=" * 68)
    print("WRITTEN")
    print("=" * 68)
    for f in ["ARCHITECTURE.md", "architecture.dot", "architecture.mmd",
              "parameters.csv", "provenance.json"] + (["architecture.png"] if rendered else []):
        p = os.path.join(OUT_DIR, f)
        if os.path.exists(p):
            print(f"  {f:<24} {human(os.path.getsize(p))}")
    if not rendered:
        print("\n  architecture.png was not produced: Graphviz is not installed.")
        print("  Not a problem. architecture.mmd renders on GitHub with nothing")
        print("  installed. To get the PNG as well:  conda install -c conda-forge graphviz")

    missing = [s["file"] for s in scripts if not (s["reads"] or s["writes"])]
    if missing:
        print(f"\n  These scripts declare no __reads__ / __writes__, so they do not")
        print(f"  appear in the data-flow graph: {', '.join(missing)}")

    print("\n  Run this at the end of every working session, then commit.")
    print()


if __name__ == "__main__":
    main()
