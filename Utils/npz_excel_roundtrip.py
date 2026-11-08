#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npz_excel_roundtrip.py

Purpose
-------
Utility script to round-trip NumPy `.npz` telemetry/diagnostic files through
Excel for manual inspection and light-weight editing, without losing array
shapes or dtypes.

It supports two symmetric operations:

1. EXPORT (NPZ → Excel)
   - Each array in the `.npz` is written to its own Excel sheet.
   - Sheet names are sanitized and de-duplicated to satisfy Excel’s 31-character
     and character-set constraints.
   - A special `_meta` sheet records, for every key:
       * the original NPZ key,
       * the associated sheet name,
       * the array dtype,
       * the original N-D shape,
       * the 2-D “saved” shape written to Excel,
       * the original number of dimensions.
   - High-dimensional arrays (ND > 2) are flattened along trailing dimensions
     so the leading axis maps to rows and all remaining elements become columns.

2. IMPORT (Excel → NPZ)
   - Reads the `_meta` sheet to recover keys, dtypes, and shapes.
   - Loads each sheet, coerces to the recorded dtype, trims fully empty trailing
     rows/columns the user may have left, and pads/truncates back to the saved
     2-D shape if needed.
   - Reshapes each 2-D sheet back to the original N-D array shape and rebuilds
     a `.npz` with the same keys and dtypes as the original export.

Usage
-----
From the command line:

  # Export one or more NPZ files to side-by-side Excel workbooks
  python npz_excel_roundtrip.py export deputy1_PH_SE3.npz deputy1_MRP_FB.npz deputy1_PID.npz

  # Import back from edited Excel files (writes/overwrites .npz next to them)
  python npz_excel_roundtrip.py import deputy1_PH_SE3.xlsx deputy1_MRP_FB.xlsx deputy1_PID.xlsx

Notes
-----
- Numeric arrays are passed through `pandas` and coerced to numeric types where
  possible, preserving NaNs to represent empty cells.
- Non-numeric arrays (e.g., strings, object) are round-tripped as raw values
  without numeric coercion.
- The `_meta` sheet is required for the import step; deleting or editing it
  will prevent reconstruction of the original NPZ structure.
"""


import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

MAX_SHEET_LEN = 31

def _sanitize_sheet(name: str) -> str:
    """Excel sheet name constraints."""
    s = re.sub(r'[:\\/?*\[\]]', '_', name)
    return s[:MAX_SHEET_LEN] if len(s) > MAX_SHEET_LEN else s

def _make_unique(names):
    """Ensure unique sheet names while preserving order."""
    seen = {}
    out = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 0
            out.append(base)
            continue
        i = seen[base] + 1
        while True:
            cand = f"{base[:MAX_SHEET_LEN-4]}_{i}"
            if cand not in seen:
                seen[base] = i
                seen[cand] = 0
                out.append(cand)
                break
            i += 1
    return out

def _to_2d(arr: np.ndarray):
    """Return (arr2d, saved_shape) where arr2d is 2D representation for Excel."""
    if arr.ndim == 0:
        a2 = arr.reshape(1, 1)
        saved = a2.shape
        return a2, saved
    if arr.ndim == 1:
        a2 = arr.reshape(-1, 1)
        return a2, a2.shape
    if arr.ndim == 2:
        return arr, arr.shape
    # ND > 2: flatten trailing dims; preserve leading axis as rows
    rows = arr.shape[0]
    cols = int(np.prod(arr.shape[1:]))
    a2 = arr.reshape(rows, cols)
    return a2, a2.shape

def export_npz_to_excel(npz_path: Path, xlsx_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    keys = list(data.files)
    # Map keys → unique, sanitized sheet names
    raw_sheet_names = [_sanitize_sheet(k) for k in keys]
    sheet_names = _make_unique(raw_sheet_names)

    meta_rows = []
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        for k, sheet in zip(keys, sheet_names):
            arr = data[k]
            arr2d, saved_shape = _to_2d(arr)
            df = pd.DataFrame(arr2d)
            df.to_excel(writer, sheet_name=sheet, index=False)
            meta_rows.append({
                "key": k,
                "sheet": sheet,
                "dtype": str(arr.dtype),
                "orig_shape": json.dumps(list(arr.shape)),
                "saved_shape": json.dumps(list(saved_shape)),
                "ndim": arr.ndim,
            })
        meta_df = pd.DataFrame(meta_rows, columns=["key","sheet","dtype","orig_shape","saved_shape","ndim"])
        meta_df.to_excel(writer, sheet_name="_meta", index=False)

    print(f"[EXPORT] {npz_path.name} → {xlsx_path.name}  (arrays: {len(keys)})")

def _coerce_numeric(df: pd.DataFrame, dtype: np.dtype):
    """Convert DataFrame to numpy with target dtype, keeping NaNs if present."""
    # Try numeric where possible; keep strings as-is if dtype not numeric
    if np.issubdtype(dtype, np.number):
        df2 = df.apply(pd.to_numeric, errors="coerce")
        return df2.to_numpy().astype(dtype, copy=False)
    else:
        # object or string-like; do not coerce
        return df.to_numpy()

def import_excel_to_npz(xlsx_path: Path, npz_path: Path):
    xl = pd.ExcelFile(xlsx_path)
    if "_meta" not in xl.sheet_names:
        raise RuntimeError(f"Missing _meta sheet in {xlsx_path.name}; cannot reconstruct shapes/dtypes.")

    meta = pd.read_excel(xlsx_path, sheet_name="_meta")
    required_cols = {"key","sheet","dtype","orig_shape","saved_shape","ndim"}
    if not required_cols.issubset(set(meta.columns)):
        raise RuntimeError("Meta sheet missing required columns. Aborting.")

    out = {}
    for _, row in meta.iterrows():
        key = str(row["key"])
        sheet = str(row["sheet"])
        dtype = np.dtype(str(row["dtype"]))
        orig_shape = tuple(json.loads(row["orig_shape"]))
        saved_shape = tuple(json.loads(row["saved_shape"]))

        if sheet not in xl.sheet_names:
            raise RuntimeError(f"Sheet '{sheet}' for key '{key}' not found in {xlsx_path.name}.")

        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        arr2d = _coerce_numeric(df, dtype)

        # Trim any fully-empty trailing rows/cols the user might have left
        # (only for numeric types)
        if np.issubdtype(dtype, np.number):
            # drop rows that are all NaN
            mask_rows = ~np.all(pd.isna(arr2d), axis=1)
            arr2d = arr2d[mask_rows]
            # drop columns that are all NaN
            if arr2d.size > 0:
                mask_cols = ~np.all(pd.isna(arr2d), axis=0)
                arr2d = arr2d[:, mask_cols]

        # Verify shape (pad/truncate if needed to match saved_shape)
        rows, cols = arr2d.shape if arr2d.ndim == 2 else (arr2d.size, 1)
        target_rows, target_cols = (saved_shape + (1,))[:2] if len(saved_shape) == 1 else saved_shape

        # Pad with NaNs/zeros if user deleted rows/cols; truncate if added too many
        def _pad(arr, r, c, dtype):
            rr, cc = arr.shape
            R = min(rr, r); C = min(cc, c)
            out = np.empty((r, c), dtype=float if np.issubdtype(dtype, np.number) else object)
            out[:] = np.nan if np.issubdtype(dtype, np.number) else ""
            out[:R, :C] = arr[:R, :C]
            return out

        if (rows, cols) != (target_rows, target_cols):
            arr2d = _pad(arr2d, target_rows, target_cols, dtype)

        # Cast back to requested dtype
        if np.issubdtype(dtype, np.number):
            arr2d = np.asarray(arr2d, dtype=dtype)
        else:
            arr2d = arr2d.astype(dtype)

        # Reshape to original shape
        arrN = arr2d
        if len(orig_shape) == 0:
            arrN = arr2d.reshape(())
        elif len(orig_shape) == 1:
            arrN = arr2d.reshape(orig_shape[0])
        elif len(orig_shape) == 2:
            arrN = arr2d.reshape(orig_shape)
        else:
            # ND>2 case: we flattened trailing dims, so invert that
            trailing = int(np.prod(orig_shape[1:]))
            arrN = arr2d.reshape(orig_shape[0], trailing).reshape(orig_shape)

        out[key] = arrN

    # Save NPZ
    np.savez(npz_path, **out)
    print(f"[IMPORT] {xlsx_path.name} → {npz_path.name}  (arrays: {len(out)})")

def main():
    ap = argparse.ArgumentParser(description="Round-trip NPZ ⇄ Excel (.xlsx)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("export", help="Export NPZ to Excel")
    ap_exp.add_argument("npz_files", nargs="+", help="One or more .npz paths")

    ap_imp = sub.add_parser("import", help="Import Excel back to NPZ")
    ap_imp.add_argument("xlsx_files", nargs="+", help="One or more .xlsx paths")

    args = ap.parse_args()

    if args.cmd == "export":
        for p in args.npz_files:
            npz = Path(p)
            if not npz.exists():
                print(f"[SKIP] {npz} not found", file=sys.stderr)
                continue
            xlsx = npz.with_suffix(".xlsx")
            export_npz_to_excel(npz, xlsx)

    elif args.cmd == "import":
        for p in args.xlsx_files:
            xls = Path(p)
            if not xls.exists():
                print(f"[SKIP] {xls} not found", file=sys.stderr)
                continue
            npz = xls.with_suffix(".npz")
            import_excel_to_npz(xls, npz)

if __name__ == "__main__":
    main()

