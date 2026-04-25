#!/usr/bin/env python
"""Resolve each specimen_id in stage_b_manifest.txt → 3DNeuronReconstruction
download URL via Allen's well_known_files include, then download the SWC.

The simpler `well_known_file_download/specimen/{id}/recon.swc` endpoint
(what fetch.sh's bash version originally used) returns 404; the canonical
path is `/api/v2/well_known_file_download/{wkf_id}` where `wkf_id` comes
from the Specimen JSON's `neuron_reconstructions[].well_known_files`.
"""
from __future__ import annotations
import pathlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tracks/core/07_cell_morphology/stage_b_manifest.txt"
OUT = ROOT / "data/core_07_cell_morphology/swc/stage_b"
BASE = "http://api.brain-map.org"


def fetch_one(sid: str) -> tuple[str, str]:
    out = OUT / f"{sid}.swc"
    if out.exists() and out.stat().st_size > 100:
        return sid, "cached"
    try:
        r = requests.get(
            f"{BASE}/api/v2/data/Specimen/{sid}.json"
            f"?include=neuron_reconstructions(well_known_files(well_known_file_type))",
            timeout=30,
        )
        r.raise_for_status()
        nrs = r.json().get("msg", [{}])[0].get("neuron_reconstructions", [])
        wkf_id = None
        for nr in nrs:
            for wkf in nr.get("well_known_files", []):
                if wkf.get("well_known_file_type", {}).get("name") == "3DNeuronReconstruction":
                    wkf_id = wkf["id"]; break
            if wkf_id: break
        if not wkf_id:
            return sid, "no-swc-wkf"
        r2 = requests.get(f"{BASE}/api/v2/well_known_file_download/{wkf_id}",
                          timeout=120)
        r2.raise_for_status()
        out.write_bytes(r2.content)
        return sid, "ok"
    except Exception as e:
        if out.exists(): out.unlink()
        return sid, f"err:{type(e).__name__}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sids = []
    with open(MANIFEST) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("specimen_id") or line.startswith("#"):
                continue
            sids.append(line.split("\t")[0])
    print(f"resolving {len(sids)} Allen specimens with 8 worker threads")

    ok = cached = fail = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_one, s) for s in sids]
        for i, f in enumerate(as_completed(futures), 1):
            _sid, st = f.result()
            if st == "ok": ok += 1
            elif st == "cached": cached += 1
            else: fail += 1
            if i % 50 == 0:
                print(f"  [{i}/{len(sids)}] ok={ok} cached={cached} fail={fail}")
    print(f"DONE: ok={ok} cached={cached} fail={fail}")


if __name__ == "__main__":
    main()
