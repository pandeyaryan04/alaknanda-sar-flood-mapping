"""Jal-SAR extensions, step M0: environment checks for M0 INSTALL PACKAGES.bat.

Usage:
  python m0_env_check.py snapshot <out.json>      record every installed package and its version
  python m0_env_check.py constraints <out.txt>    write name==version for every installed package,
                                                  so pip can add new packages but never change these
  python m0_env_check.py compare <before.json> <after.json>
                                                  exit 1 if any package that was installed before changed
  python m0_env_check.py imports                  import shap and cdsapi and print their versions
"""
import json
import re
import sys
from importlib import metadata


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed():
    try:
        from packaging.version import Version, InvalidVersion
    except ImportError:  # pip always carries a copy
        from pip._vendor.packaging.version import Version, InvalidVersion
    out = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        ver = dist.version
        if not name or not ver:
            continue
        key = norm(name)
        if key in out:
            continue
        try:
            Version(ver)
        except InvalidVersion:
            continue
        out[key] = ver
    return dict(sorted(out.items()))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    mode = sys.argv[1]
    if mode == "snapshot":
        pk = installed()
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(pk, f, indent=1)
        print(f"  saved {len(pk)} packages to {sys.argv[2]}")
        for k in ("numpy", "scikit-learn", "torch", "rasterio", "scipy", "pandas"):
            print(f"  {k:<14}{pk.get(k, 'not installed')}")
    elif mode == "constraints":
        pk = installed()
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            for k, v in pk.items():
                f.write(f"{k}=={v}\n")
        print(f"  holding {len(pk)} installed packages at their current versions")
    elif mode == "compare":
        a = json.load(open(sys.argv[2], encoding="utf-8"))
        b = json.load(open(sys.argv[3], encoding="utf-8"))
        changed = {k: (a[k], b.get(k)) for k in a if b.get(k) != a[k]}
        added = {k: b[k] for k in b if k not in a}
        print("  new packages added:")
        for k, v in added.items():
            print(f"    + {k} {v}")
        if not added:
            print("    (none)")
        if changed:
            print("  PROBLEM: packages that were already installed have changed:")
            for k, (x, y) in changed.items():
                print(f"    ! {k}: {x} -> {y}")
            sys.exit(1)
        print("  every package that was already installed is unchanged")
    elif mode == "imports":
        import shap  # noqa: F401
        import cdsapi  # noqa: F401
        sv, cv = metadata.version("shap"), metadata.version("cdsapi")
        print(f"  shap {sv}")
        print(f"  cdsapi {cv}")
        parts = [int(p) for p in re.findall(r"\d+", cv)[:3]]
        if parts < [0, 7, 7]:
            sys.exit("  PROBLEM: cdsapi is older than 0.7.7, which the GloFAS site needs")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
