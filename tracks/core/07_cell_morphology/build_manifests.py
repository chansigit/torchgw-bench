#!/usr/bin/env python
"""Populate stage_{a,b}_manifest.txt by querying NeuroMorpho + Allen CTDB.

NeuroMorpho's REST API takes POST with a JSON body whose values are lists
(`{"cell_type":["pyramidal"], "species":["mouse"]}`). The downloadable
SWC lives at `dableFiles/{archive}/CNG version/{name}.CNG.swc` — note the
URL path uses the *archive* name, not the neuron name — so we record
both columns in the manifest.

Allen Brain Atlas Cell Types Database exposes morphology specimens via
the public Specimen API; the canonical dendrite_type label lives in the
cells.csv release and may need a manual join — see TODO in stage B.
"""
from __future__ import annotations
import argparse
import pathlib
import requests

TRACK = pathlib.Path(__file__).resolve().parent
NM_API = "https://neuromorpho.org/api/neuron/select"
ABA_SPECIMEN_API = (
    "https://api.brain-map.org/api/v2/data/Specimen/query.json"
    "?criteria=[is_cell_specimen$eqtrue]"
    "&include=neuron_reconstructions"
    "&num_rows={n}&start_row=0"
)


def fetch_neuromorpho(class_name: str, n: int, species: str = "mouse"
                      ) -> list[tuple[str, str]]:
    """Return up to `n` (neuron_name, archive) tuples for class+species."""
    out: list[tuple[str, str]] = []
    page = 0
    while len(out) < n and page < 200:
        body = {"cell_type": [class_name], "species": [species]}
        r = requests.post(f"{NM_API}?size=100&page={page}",
                          json=body, timeout=60)
        r.raise_for_status()
        items = r.json().get("_embedded", {}).get("neuronResources", [])
        if not items:
            break
        for it in items:
            out.append((it["neuron_name"], it["archive"]))
            if len(out) >= n:
                break
        page += 1
    return out[:n]


def write_stage_a_manifest(per_class: int, classes: list[str], species: str):
    manifest = TRACK / "stage_a_manifest.txt"
    with open(manifest, "w") as fh:
        fh.write("neuron_name\tclass\tarchive\n")
        for cls in classes:
            print(f"[stage_a] querying NeuroMorpho cell_type={cls!r} ({species}) ...")
            rows = fetch_neuromorpho(cls, per_class, species=species)
            print(f"[stage_a]   got {len(rows)} ids")
            for name, archive in rows:
                fh.write(f"{name}\t{cls}\t{archive}\n")
    print(f"[stage_a] wrote {manifest}")


def fetch_allen_morphology_cells(n: int) -> list[tuple[str, str]]:
    """Return (specimen_id, label_placeholder) tuples for cells with a
    public neuron_reconstruction. Labels need manual join — see TODO."""
    r = requests.get(ABA_SPECIMEN_API.format(n=n * 4), timeout=180)
    r.raise_for_status()
    rows = r.json().get("msg", [])
    out: list[tuple[str, str]] = []
    for row in rows:
        if not row.get("neuron_reconstructions"):
            continue
        sid = str(row.get("id"))
        # CTDB browser exposes spiny/aspiny via cells.csv; the Specimen API
        # alone does not return it cleanly. Default to 'unknown' and require
        # a manual join before bench (see TODO).
        out.append((sid, "unknown"))
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
    print("[stage_b] NOTE: 'class' column is 'unknown'; for production "
          "use, join against the Allen CTDB cells.csv to get the canonical "
          "spiny/aspiny/sparsely-spiny dendrite_type and edit the manifest "
          "before running fetch.sh.")


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
