"""Resumable extraction: process patients in chunks, zip each finished chunk to Google Drive.

If the Colab/Kaggle session resets, just run the same command again: chunks whose zip already
exists on Drive are skipped, so you lose at most the chunk that was in progress.

  Extract (resumable):
    python -m src.data.run_chunks --drive /content/drive/MyDrive/lung_nodule \
        --work /content/work --raw /content/lidc_raw --chunk 100
  Assemble all chunk zips into one folder + index (do this at the start of every later session):
    python -m src.data.run_chunks --drive /content/drive/MyDrive/lung_nodule \
        --assemble /content/lidc_patches
"""
import argparse, shutil, subprocess, sys, zipfile
from pathlib import Path


def expected_patients(first, last):
    """Patients in [first,last] that have an annotated scan in pylidc (2 IDs in 1..1012 have none)."""
    from .extract_patches import numpy_compat_shim
    numpy_compat_shim()
    import pylidc as pl
    have = {s.patient_id for s in pl.query(pl.Scan).all()}
    return {f"LIDC-IDRI-{i:04d}" for i in range(first, last + 1)} & have


def run_extract(start, end, raw, out, stub=False):
    cmd = [sys.executable, "-m", "src.data.extract_patches", "--start", str(start), "--end", str(end),
           "--raw", str(raw), "--out", str(out)]
    if stub:  # test-only: pretend extraction worked
        import json
        (out / "meta").mkdir(parents=True, exist_ok=True); (out / "patches").mkdir(exist_ok=True)
        for i in range(start, end + 1):
            p = f"LIDC-IDRI-{i:04d}"
            (out / "meta" / f"{p}.json").write_text("[]")
            (out / "patches" / f"{p}_n00.npz").write_bytes(b"x")
        return
    subprocess.run(cmd, check=False)


def missing(out, expect):
    done = {p.stem for p in (out / "meta").glob("*.json")} if (out / "meta").exists() else set()
    return sorted(expect - done)


def zip_dir(src: Path, dst_zip: Path):
    tmp = dst_zip.with_suffix(".partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # npz files are already compressed
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(src))
    tmp.replace(dst_zip)  # rename only after the whole zip is written -> no half-finished "done" files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/content/work"))
    ap.add_argument("--raw", type=Path, default=Path("/content/lidc_raw"))
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=1012)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--assemble", type=Path, help="unzip every chunk zip into this folder and exit")
    ap.add_argument("--stub", action="store_true", help="test only")
    a = ap.parse_args()
    a.drive.mkdir(parents=True, exist_ok=True)

    if a.assemble:
        a.assemble.mkdir(parents=True, exist_ok=True)
        zips = sorted(a.drive.glob("chunk_*.zip"))
        print(f"assembling {len(zips)} chunk zips -> {a.assemble}")
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(a.assemble)
        n = len(list((a.assemble / "meta").glob("*.json")))
        print(f"done: {n} patient records, {len(list((a.assemble / 'patches').glob('*.npz')))} patches")
        return

    failed = []
    for start in range(a.first, a.last + 1, a.chunk):
        end = min(start + a.chunk - 1, a.last)
        zname = a.drive / f"chunk_{start:04d}_{end:04d}.zip"
        if zname.exists():
            print(f"[skip] {zname.name} already on Drive"); continue
        out = a.work / f"chunk_{start:04d}_{end:04d}"
        expect = expected_patients(start, end)
        print(f"[run ] patients {start}-{end}  ({len(expect)} with scans)")
        for attempt in (1, 2):                       # one retry fills transient download failures
            run_extract(start, end, a.raw, out, a.stub)
            miss = missing(out, expect)
            if not miss:
                break
            print(f"  attempt {attempt}: {len(miss)} patients missing, e.g. {miss[:3]}")
        if miss:
            print(f"  NOT zipping chunk {start}-{end}; missing {miss}"); failed.append((start, end, miss)); continue
        zip_dir(out, zname)
        print(f"[done] saved {zname.name} ({zname.stat().st_size/1e6:.0f} MB)")
        shutil.rmtree(out, ignore_errors=True)       # free local disk; the zip on Drive is the master copy
    print("\nFINISHED. Chunks that failed:", failed if failed else "none")


if __name__ == "__main__":
    main()
