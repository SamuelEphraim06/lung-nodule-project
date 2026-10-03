# Lung nodule malignancy project

## Week 1: patch extraction (route C)
```bash
pip install -r requirements.txt
python tests_geometry.py                       # must print OK
python -m src.data.extract_patches --start 1 --end 3 --raw /tmp/lidc_raw --out /kaggle/working/lidc_patches
python -m src.data.build_index  --out /kaggle/working/lidc_patches
python -m src.data.sanity_check --out /kaggle/working/lidc_patches
```
Open `sanity.png`: the red outline must sit on the nodule in all three views.
Only then scale up (`--start 1 --end 1012`), in chunks, re-running is safe (finished patients are skipped).

## Notes
- Patches are 96^3 at 1 mm, stored as int16 HU (clip to [-1000, 400] at load time).
- Every reader's ratings are kept as lists in nodules.csv (malignancy, spiculation, ...).
- `numpy_compat_shim()` is needed because pylidc uses removed NumPy aliases.
- Split by PATIENT (patient_id), never by nodule.
