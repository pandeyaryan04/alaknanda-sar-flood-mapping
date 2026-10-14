# Setup runbook

Everything from a clean laptop to a working data pipeline. Follow it in order.
Commands go in **Anaconda Prompt**, not PowerShell.

Anything in `code blocks` is typed exactly. Anything in `<angle brackets>` you replace.

---

## Step 0 — Where the project lives

Pick one folder and never move it. Suggested:

```
<a folder of your choice>
```

Avoid OneDrive-synced folders. OneDrive tries to upload every intermediate file
and will fight you over a 25 GB dataset.

```
mkdir <a folder of your choice>
cd /d <a folder of your choice>
```

Unzip the code files into this folder. You should end up with:

```
FYP\
  00_check_machine.py
  01_scene_inventory.py
  02_fetch_sen1floods11.py
  99_document_architecture.py
  environment.yml
  SETUP_RUNBOOK.md
```

---

## Step 1 — Anaconda

If `conda --version` already works, skip this.

Otherwise install **Miniconda** (smaller and cleaner than full Anaconda) from
`https://docs.conda.io/en/latest/miniconda.html`. Take the Windows 64-bit
installer. During setup leave "Add to PATH" unchecked — use Anaconda Prompt from
the Start menu instead.

Check:

```
conda --version
```

---

## Step 2 — The environment

```
cd /d <a folder of your choice>
conda env create -f environment.yml
conda activate fyp
```

Takes 10–20 minutes and downloads roughly 2 GB. If solving stalls for more than
about 15 minutes, cancel with Ctrl+C and switch to the faster solver:

```
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -f environment.yml
```

Your prompt should now start with `(fyp)`. **It must say `(fyp)` before every
command from here on.** If you close the window, run `conda activate fyp` again.

---

## Step 3 — PyTorch with CUDA

Deliberately separate from `environment.yml`, because conda's Windows CUDA
packaging is unreliable.

```
conda activate fyp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

About 2.5 GB. Then check it actually sees the GPU:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want `True` and your card's name. If it says `False` you got the CPU build:

```
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Step 4 — Earth Engine

One-time, per machine.

```
conda activate fyp
earthengine authenticate
```

A browser opens. Sign in with the Google account that has Earth Engine access,
approve, and paste the code back if asked.

Find your project id at `https://code.earthengine.google.com` — top right, next
to your avatar. If it is **not** `gbpuat-fyp-alaknanda`, open `01_scene_inventory.py`
in Notepad and change the `EE_PROJECT` line near the top.

---

## Step 5 — Git (do this now, not in week 14)

```
git init
git branch -M main
```

Create a file called `.gitignore` in the FYP folder containing:

```
data/
outputs/
*.tif
*.part
__pycache__/
machine_report.txt
```

That keeps large data out of the repository while tracking the code. Then:

```
git add .
git commit -m "Project setup: environment and data scripts"
```

The public repository at the end of Semester 1 is far easier if the history
starts now rather than being assembled retrospectively.

---

## Step 6 — Check the machine

```
python 00_check_machine.py
```

Prints a report and writes `machine_report.txt`. **Keep that file.** It
answers the GPU, disk, Windows edition and Earth Engine questions in one go, and
it measures your actual download speed so the estimates below are real rather
than guessed.

It installs nothing and changes nothing.

---

## Step 7 — Scene inventory (run this early)

The most important thing you can run this week. It decides whether a usable
Sentinel-1 pair exists for either candidate flood, and everything downstream
depends on the answer.

```
python 01_scene_inventory.py
```

A few minutes. Downloads no imagery — it queries metadata only.

Writes into `data\`:

| File | What it is |
|---|---|
| `corridor.geojson` | The corridor polygon. Open in QGIS to sanity-check the reach. |
| `scene_inventory.csv` | Every scene found, with date, orbit, pass and corridor coverage |
| `screening_verdict.csv` | PASS or FAIL per event, per orbit, per pass direction |

Read the "WHAT THIS MEANS" block it prints at the end. If nothing passes, that is
a known branch with a decided response — the two CSVs and the fallback list the
script prints show which applies.

---

## Step 8 — Training data

### 8a. Survey first. Always.

```
python 02_fetch_sen1floods11.py --list
```

Downloads nothing. Reports the exact file count and byte size of every folder, so
you know what you are committing to before you start.

### 8b. Get the training minimum

```
python 02_fetch_sen1floods11.py --get core
```

Roughly **1.2 GB** — the 446 hand-labelled radar chips, their masks, and the
train/validation/test split lists. This is everything you need to train.

Safe to interrupt with Ctrl+C. Safe after a power cut. Re-run the same command
and it skips whatever is already on disk at the right size.

### 8c. The weak-label set — only if you decide you need it

Roughly **25 GB**. It is for pre-training: learn the rough shape of the problem on
a lot of noisy data, then fine-tune on the good data. Useful, not required.

**Do not download it yet.** Get `core` working end to end first.

---

## Step 9 — Optional: split the weak download across two machines

Worth it only for the 25 GB weak set, and only if the two machines are on
**genuinely separate connections**. Two laptops on the same hostel Wi-Fi share one
uplink and you gain nothing — one on Wi-Fi and one on a 5G hotspot is a real split.

**Laptop A:**
```
python 02_fetch_sen1floods11.py --get weak --shard 1/2
```

**Laptop B, at the same time:**
```
python 02_fetch_sen1floods11.py --get weak --shard 2/2
```

The split is by a hash of each filename, so the two machines never fetch the same
file and never leave a gap. No coordination and no communication between them.

When both finish, copy laptop B's `data\sen1floods11` folder **into** laptop A's —
merge, do not replace. Then on laptop A:

```
python 02_fetch_sen1floods11.py --get weak --verify
```

This checks every expected file against the bucket's own byte count and prints
exactly what is missing. If anything is, re-run without `--verify` and only the
gaps get fetched.

**Honest note on the maths.** The saving is smaller than it looks. Halving a
3-hour download saves 90 minutes, but then you move ~12 GB between machines, which
is 10–20 minutes over USB 3 and considerably longer over Wi-Fi. The real benefit is
not speed — it is that a failure on one connection does not stop the other, and
that everything ends up verified rather than assumed.

---

## Step 9b — Document the architecture (end of every session)

```
python 99_document_architecture.py
git add -A
git commit -m "<what changed>"
```

Reads every pipeline script and writes `outputs\architecture\`:

| File | What it is |
|---|---|
| `ARCHITECTURE.md` | The readable document. Stages, data flow, every parameter, the environment. |
| `architecture.mmd` | The data-flow diagram as Mermaid. GitHub renders it with nothing installed. |
| `architecture.png` | The same diagram rendered, if Graphviz is present |
| `parameters.csv` | Every configuration constant, per script |
| `provenance.json` | Package versions, git commit, per-file hashes |

**Why bother.** By week 15 the code will have drifted from the plan, and a
methodology section typed by hand describes what was intended in August rather
than what actually ran. This generates the section from the source, so it cannot
be wrong about its own methods. It also records the git commit and a hash of
each script, so any figure in the report can be traced to the exact code that
produced it.

It runs nothing and imports nothing — it parses the scripts as text, so it works
even mid-edit or with a half-installed environment.

**The one rule.** Each script declares three lines near the top:

```python
__stage__ = "Short name of what this stage does"
__reads__ = ["what it consumes"]
__writes__ = ["what it produces"]
```

Keep those honest and the diagram, the tables and the report text all follow.
That is the only maintenance this needs.

Re-run it whenever the pipeline changes.

---

## Step 10 — Before you walk away from a long run

1. **Power Options** → plugged in: sleep **Never**; "when I close the lid" → **Do nothing**
2. **Keep it plugged in.** Windows throttles hard on battery.
3. **Hard surface, elevated.** Not a bed. A laptop under sustained load needs airflow.
4. **Start the job before you leave**, writing to a log, then read the log remotely.
   Launching over SSH and having it survive disconnection is fiddlier on Windows
   than it is worth.

---

## Quick reference

| Command | What it does |
|---|---|
| `conda activate fyp` | **Run this first, every time** |
| `python 00_check_machine.py` | Machine report |
| `python 01_scene_inventory.py` | Scene inventory and event screening |
| `python 02_fetch_sen1floods11.py --list` | Survey, download nothing |
| `python 02_fetch_sen1floods11.py --get core` | Training minimum (~1.2 GB) |
| `python 02_fetch_sen1floods11.py --get weak --shard 1/2` | Half the weak set |
| `python 02_fetch_sen1floods11.py --get weak --verify` | Check for gaps |
| `python 99_document_architecture.py` | Regenerate the architecture doc — end of every session |

---

## When something breaks

**`'conda' is not recognized`** — you are in Command Prompt, not Anaconda Prompt.
Open Anaconda Prompt from the Start menu.

**`ModuleNotFoundError`** — the environment is not active. The prompt must show `(fyp)`.

**Earth Engine says not authenticated** — run `earthengine authenticate` again, and
check `EE_PROJECT` in `01_scene_inventory.py` matches the project id shown at
`code.earthengine.google.com`.

**`torch.cuda.is_available()` is False** — CPU build installed. Redo step 3.

**Download stalls or fails partway** — press Ctrl+C and re-run the same command.
Nothing is corrupted; partial files are discarded rather than kept.

**Anything else** — copy the full error text, not a screenshot of part of it.
