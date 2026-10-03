"""Route C: batch-download LIDC-IDRI series, cut 1 mm nodule patches, delete raw DICOMs.

Example (3-patient dry run):
    python -m src.data.extract_patches --start 1 --end 3 \
        --raw /tmp/lidc_raw --out /kaggle/working/lidc_patches

Outputs in --out:
    patches/<patient>_n<idx>.npz   (ct: int16 HU, mask: uint8 majority-vote mask)
    meta/<patient>.json            (one record per physical nodule; also acts as a resume marker)
Run `python -m src.data.build_index --out ...` afterwards to merge into nodules.csv.
"""
import argparse, json, shutil, sys, time
from pathlib import Path

import numpy as np

from .geometry import crop_patch

ATTRS = ["subtlety", "internalStructure", "calcification", "sphericity",
         "margin", "lobulation", "spiculation", "texture", "malignancy"]



def numpy_compat_shim():
    """pylidc is unmaintained and uses APIs removed from modern NumPy / Python.
    Restore them BEFORE importing pylidc:
      - np.int / np.bool / np.float / np.object  (removed in NumPy 1.24)
      - configparser.SafeConfigParser            (removed in Python 3.12)
    """
    for name, typ in [("int", int), ("bool", bool), ("float", float), ("object", object)]:
        if name not in np.__dict__:
            setattr(np, name, typ)
    import configparser
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser


def configure_pylidc(raw_root: Path):
    """pylidc reads ~/.pylidcrc at import time, so write it BEFORE importing pylidc."""
    rc = Path.home() / ".pylidcrc"
    rc.write_text(f"[dicom]\npath = {raw_root}\nwarn = False\n")


def pid(i: int) -> str:
    return f"LIDC-IDRI-{i:04d}"


def download_batch(scans, raw_root: Path, tmp_dl: Path):
    """Download the exact series pylidc refers to, then arrange as raw_root/<patient>/<series_uid>/."""
    from tcia_utils import nbia
    uids = [s.series_instance_uid for s in scans]
    tmp_dl.mkdir(parents=True, exist_ok=True)
    nbia.downloadSeries(uids, input_type="list", path=str(tmp_dl))
    for s in scans:
        src = tmp_dl / s.series_instance_uid
        dst = raw_root / s.patient_id / s.series_instance_uid
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {src}. Check how tcia_utils laid out the download "
                f"(run help(nbia.downloadSeries) and ls {tmp_dl}).")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))


def process_scan(scan, out_dir: Path, size: int):
    vol = scan.to_volume(verbose=False)
    lo, hi = int(vol.min()), int(vol.max())
    # Sanity: HU volumes normally have min near -1000..-3000 and max in the hundreds/thousands.
    if lo >= 0:
        print(f"  WARNING {scan.patient_id}: volume min={lo} (>=0). Not HU? Check rescale.")

    clusters = scan.cluster_annotations(verbose=False)
    zvals = np.asarray(scan.slice_zvals)
    records = []
    for n_idx, anns in enumerate(clusters):
        # count of readers marking each voxel -> majority-vote mask
        cnt = np.zeros(vol.shape, dtype=np.uint8)
        for a in anns:
            cnt[a.bbox()] += a.boolean_mask().astype(np.uint8)
        center = np.mean([np.asarray(a.centroid, dtype=float) for a in anns], axis=0)

        ct = crop_patch(vol, scan.pixel_spacing, zvals, center, size, order=1, cval=-1024)
        cp = crop_patch(cnt, scan.pixel_spacing, zvals, center, size, order=0, cval=0)
        need = int(np.ceil(len(anns) / 2))
        mask = (cp >= need).astype(np.uint8)

        fname = f"{scan.patient_id}_n{n_idx:02d}.npz"
        np.savez_compressed(out_dir / "patches" / fname,
                            ct=np.clip(ct, -1024, 3071).astype(np.int16), mask=mask)

        rec = dict(
            patient_id=scan.patient_id, series_uid=scan.series_instance_uid,
            nodule_idx=n_idx, n_readers=len(anns), cluster_too_big=len(anns) > 4,
            center_ijk=[float(x) for x in center],
            diameter_mm=float(np.mean([a.diameter for a in anns])),
            slice_thickness=float(scan.slice_thickness),
            pixel_spacing=float(scan.pixel_spacing),
            contrast_used=bool(scan.contrast_used),
            patch_file=f"patches/{fname}", patch_size=size,
        )
        for name in ATTRS:  # keep EVERY reader's rating, not just the mean
            rec[name] = [int(getattr(a, name)) for a in anns]
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=3, help="inclusive patient index (max 1012)")
    ap.add_argument("--raw", type=Path, required=True, help="temp dir for raw DICOMs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=10, help="patients per download call")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()

    args.raw.mkdir(parents=True, exist_ok=True)
    (args.out / "patches").mkdir(parents=True, exist_ok=True)
    (args.out / "meta").mkdir(parents=True, exist_ok=True)

    numpy_compat_shim()
    configure_pylidc(args.raw)
    import pylidc as pl  # must come after configure_pylidc

    ids = [pid(i) for i in range(args.start, args.end + 1)]
    todo = [p for p in ids if not (args.out / "meta" / f"{p}.json").exists()]
    print(f"{len(ids)} requested, {len(todo)} to do (others already done)")

    for b in range(0, len(todo), args.batch):
        batch_ids = todo[b:b + args.batch]
        scans = []
        for p in batch_ids:
            scans += pl.query(pl.Scan).filter(pl.Scan.patient_id == p).all()
        if not scans:
            print("no annotated scans for", batch_ids); continue
        t0 = time.time()
        try:
            download_batch(scans, args.raw, args.raw / "_dl")
        except Exception as e:
            print("download failed for batch", batch_ids, "->", repr(e)); continue
        print(f"downloaded {len(scans)} series in {time.time()-t0:.0f}s")

        for p in batch_ids:
            recs = []
            ok = True
            for scan in [s for s in scans if s.patient_id == p]:
                try:
                    recs += process_scan(scan, args.out, args.size)
                except Exception as e:
                    ok = False
                    print(f"  FAILED {p}: {e!r}")
            if ok:
                (args.out / "meta" / f"{p}.json").write_text(json.dumps(recs))
                print(f"  {p}: {len(recs)} nodules")
                if not args.keep_raw:
                    shutil.rmtree(args.raw / p, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
