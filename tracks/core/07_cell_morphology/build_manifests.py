#!/usr/bin/env python
"""Populate stage_{a,b}_manifest.txt by querying NeuroMorpho + Allen CTDB.

Run once before fetch.sh; freeze the resulting manifest into git so the
benchmark is reproducible. Re-running with the same query parameters and
the same remote release should give the same IDs, but pin once and stop.

Stage A: NeuroMorpho.org — 100 cells from each of 3 morphologically
distinct classes. Defaults pick {cortical pyramidal, cortical basket,
cerebellar Purkinje} from species=mouse to keep it homogeneous; override
via --neuromorpho-classes.

Stage B: Allen Brain Atlas Cell Types Database — ~1000 cells with the
public dendrite_type label (spiny / aspiny / sparsely spiny). Mouse only.
"""
from __future__ import annotations
import argparse
import pathlib
import requests

TRACK = pathlib.Path(__file__).resolve().parent
NM_API = "https://neuromorpho.org/api/neuron/select"
ABA_API = ("https://api.brain-map.org/api/v2/data/Specimen/query.json"
           "?criteria=[is_cell_specimen$eqtrue],"
           "structure[acronym$eqVISp]"
           "&include=neuron_reconstructions,donor(transgenic_lines)"
           "&num_rows=2000")


def fetch_neuromorpho(class_name: str, n: int, species: str = "mouse") -> list[str]:
    """Return up to `n` neuron names matching `class_name` and `species`."""
    out: list[str] = []
    page = 0
    page_size = 100
    while len(out) < n and page < 50:
        params = {
            "q": f"cell_type:{class_name} AND species:{species}",
            "size": page_size,
            "page": page,
        }
        r = requests.get(NM_API, params=params, timeout=60)
        r.raise_for_status()
        items = r.json().get("_embedded", {}).get("neuronResources", [])
        if not items:
            break
        for it in items:
            out.append(it["neuron_name"])
            if len(out) >= n:
                break
        page += 1
    return out[:n]


def write_stage_a_manifest(per_class: int, classes: list[str], species: str):
    manifest = TRACK / "stage_a_manifest.txt"
    with open(manifest, "w") as fh:
        fh.write("neuron_name\tclass\n")
        for cls in classes:
            print(f"[stage_a] querying NeuroMorpho cell_type={cls!r} ({species}) ...")
            names = fetch_neuromorpho(cls, per_class, species=species)
            print(f"[stage_a]   got {len(names)} ids")
            for name in names:
                fh.write(f"{name}\t{cls}\n")
    print(f"[stage_a] wrote {manifest}")


def fetch_allen_morphology_cells(n: int) -> list[tuple[str, str]]:
    """Return (specimen_id, dendrite_type) tuples for cells with public SWC."""
    r = requests.get(ABA_API, timeout=120)
    r.raise_for_status()
    rows = r.json().get("msg", [])
    out: list[tuple[str, str]] = []
    for row in rows:
        recons = row.get("neuron_reconstructions") or []
        if not recons:
            continue
        sid = str(row.get("id"))
        # Allen's public label is on row['donor']['cell_reporter_status'] for
        # transgenic info; dendrite_type lives in cell_soma_locations or
        # the 'name'. The CTDB browser exposes spiny/aspiny via specimen
        # tags — fall back to "unknown" if not present in this minimal API.
        dtype = (row.get("dendrite_type")
                 or row.get("name", "").lower()
                 or "unknown")
        # Heuristic: many specimen names start with "cell_<class>_..."; the
        # canonical way to get dendrite_type is the cells.csv release. Until
        # this is wired up, leave "unknown" and fix by hand-editing the
        # manifest before bench. (TODO: switch to cells.csv if you need
        # the dendrite_type column reliably.)
        out.append((sid, dtype if dtype else "unknown"))
        if len(out) >= n:
            break
    return out


def write_stage_b_manifest(n: int):
    manifest = TRACK / "stage_b_manifest.txt"
    print(f"[stage_b] querying Allen CTDB for ~{n} morphology specimens ...")
    rows = fetch_allen_morphology_cells(n)
    print(f"[stage_b]   got {len(rows)} ids")
    with open(manifest, "w") as fh:
        fh.write("specimen_id\tclass\n")
        for sid, dtype in rows:
            fh.write(f"{sid}\t{dtype}\n")
    print(f"[stage_b] wrote {manifest}")
    print("[stage_b] NOTE: dendrite_type may be 'unknown' for many rows; "
          "for production use, join against the Allen CTDB cells.csv release "
          "to get the canonical spiny/aspiny/sparsely-spiny label and "
          "edit the manifest before running fetch.sh.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-per-class", type=int, default=100)
    ap.add_argument("--stage-a-classes", nargs="+",
                    default=["pyramidal", "basket", "Purkinje"])
    ap.add_argument("--stage-a-species", default="mouse")
    ap.add_argument("--stage-b-n", type=int, default=1000)
    ap.add_argument("--stage", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    if args.stage in ("A", "both"):
        write_stage_a_manifest(args.stage_a_per_class,
                               args.stage_a_classes,
                               args.stage_a_species)
    if args.stage in ("B", "both"):
        write_stage_b_manifest(args.stage_b_n)


if __name__ == "__main__":
    main()
