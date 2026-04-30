# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║   MULTISPECTRAL DRONE IMAGE CO-REGISTRATION                  ║
║   For multi-temporal forest monitoring                       ║
╚══════════════════════════════════════════════════════════════╝

WHAT THIS DOES
--------------
Co-registers multispectral drone GeoTIFFs from different dates
against a single reference image, preserving the original CRS,
geotransform, dtype (uint16, float32, etc.) and nodata values.

Designed for forest monitoring where:
  - Images have slightly different GSD (drone flew at different altitudes)
  - Spectral content changes significantly between dates (vegetation change)
  - A GPS drift of a few meters exists between flight campaigns

PIPELINE
--------
  1. Reproject source to exact target grid (same CRS, GSD, pixel origin)
     -> handles GSD mismatch and CRS differences
  2. Normalized template matching -> coarse pixel offset
     -> robust to spectral change (matches spatial structure, not values)
  3. ECC sub-pixel refinement -> sub-pixel precision
     -> fine-tunes the alignment

WHY NOT SIFT/AKAZE?
-------------------
Classical feature detectors (SIFT, AKAZE) work well for scenes with
stable spectral content. In forest monitoring, vegetation changes
dramatically between dates (growth, mortality, phenology), making
intensity-based descriptors unreliable. Template matching on a
structurally stable channel (NDVI or NIR) is more robust because
it looks for spatial patterns (dead tree clusters, canopy gaps)
rather than absolute reflectance values.

WHY IS CC ~0.5?
---------------
A Cross-Correlation of ~0.5 does NOT mean poor registration.
With one year of forest change, spectral content differs even in
perfectly aligned pixels. CC is used here only as a relative
improvement indicator, not as an absolute quality metric.
Visually verify registration using stable landmarks (dead trees,
roads, clearings) in QGIS or ArcGIS.

REQUIREMENTS
------------
    pip install opencv-python rasterio numpy matplotlib

Tested with:
    opencv-python == 4.11.0
    rasterio      == 1.4.3
    numpy         == 2.2.3
    matplotlib    == 3.10.1
    Python        == 3.11

USAGE
-----
    # Single image (for testing):
    python drone_multispec_coregistration.py

    # Batch (all .tif in a folder):
    Set MODE = 'batch' in the CONFIGURATION section below.

AUTHORS
-------
    Developed collaboratively by:
      - Lelian15 (domain expertise, field validation, testing)
      - Claude (Anthropic) - claude-sonnet-4-6
        https://www.anthropic.com

    Multi-temporal forest monitoring project, 2025.

LICENSE
-------
    MIT License - free to use, modify and distribute.
"""

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION  <-- edit only this section
# ══════════════════════════════════════════════════════════════

# 'single' -> register one image (good for testing before batch)
# 'batch'  -> register all .tif files in INPUT_DIR
MODE = 'single'

# --- Paths ---
INPUT_DIR       = r"path/to/your/images"         # folder with source images
OUTPUT_DIR      = r"path/to/your/output"         # results go here
TARGET_IMAGE    = "reference_image.tif"          # filename of the reference image
SOURCE_IMAGE    = "image_to_align.tif"           # only used when MODE='single'

# --- Band indices (0-based) ---
# Typical 5-band multispectral camera: B=0 G=1 R=2 RE=3 NIR=4
# Typical 6-band multispectral camera: B=0 G=1 R=2 RE=3 NIR=4 ...
BAND_RED = 2
BAND_NIR = 4

# --- Alignment channel ---
# 'ndvi'  -> (NIR-R)/(NIR+R): dead trees stand out strongly (recommended)
# 'nir'   -> raw NIR band: good texture, less contrast on dead trees
# 'mean'  -> average of all bands: use if NIR band index is unknown
ALIGN_CHANNEL = 'ndvi'

# --- Template matching search window ---
# Pixels to search in each direction for the coarse offset.
# Rule of thumb: GPS drift in meters / GSD (m/px) * 1.5 safety margin
# Example: 2m drift / 0.11 m/px * 1.5 = ~27 px  -> use 50-200 to be safe
SEARCH_RADIUS = 200

# --- ECC sub-pixel iterations (100-300) ---
ECC_ITERATIONS = 150

# --- Preview bands for RGB visualization (0-based indices) ---
RGB_PREVIEW = (2, 1, 0)   # R, G, B

# ══════════════════════════════════════════════════════════════
#  CODE  (no need to edit below this line)
# ══════════════════════════════════════════════════════════════

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path


# --------------------------------------------------------------
# I/O
# --------------------------------------------------------------

def read_geotiff(path):
    """Read a GeoTIFF preserving dtype, nodata, CRS and geotransform."""
    with rasterio.open(path) as src:
        data      = src.read()
        profile   = src.profile.copy()
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata
        dtype_str = src.dtypes[0]
    data = np.transpose(data, (1, 2, 0))   # (C,H,W) -> (H,W,C)
    return data, profile, transform, crs, nodata, dtype_str


def write_geotiff(path, array, profile, transform, crs, nodata, dtype_str):
    """
    Write a GeoTIFF preserving the original dtype.
    NaN values are converted to nodata before writing.
    """
    h, w, c = array.shape
    out = profile.copy()
    out.update(
        driver='GTiff', height=h, width=w, count=c,
        dtype=dtype_str, crs=crs, transform=transform,
        compress='lzw', tiled=True, blockxsize=256, blockysize=256,
    )
    if nodata is not None:
        out['nodata'] = nodata

    fill = float(nodata) if nodata is not None else 0.0
    arr  = array.copy()
    arr[np.isnan(arr)] = fill

    if np.issubdtype(np.dtype(dtype_str), np.integer):
        info = np.iinfo(np.dtype(dtype_str))
        arr  = np.clip(np.round(arr), info.min, info.max)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', **out) as dst:
        for b in range(c):
            dst.write(arr[:, :, b].astype(dtype_str), b + 1)
    print(f"    Saved : {path}")


# --------------------------------------------------------------
# Step 1: Reproject source to target grid
# --------------------------------------------------------------

def snap_to_grid(source, s_transform, s_crs,
                 t_profile, t_transform, t_crs, nodata_fill):
    """
    Reproject source so it shares exactly the same grid as target:
    same CRS, GSD, pixel origin and array dimensions.

    This aligns images using their GPS georeference. The residual
    (GPS drift between flights) is corrected in the next step.
    Uses bilinear resampling (correct for continuous reflectance data).
    """
    t_h = t_profile['height']
    t_w = t_profile['width']
    n   = source.shape[2]
    dst = np.zeros((t_h, t_w, n), dtype=np.float64)

    for b in range(n):
        band = np.where(np.isnan(source[:, :, b]),
                        nodata_fill,
                        source[:, :, b]).astype(np.float64)
        reproject(
            source        = band,
            destination   = dst[:, :, b],
            src_transform = s_transform,
            src_crs       = s_crs,
            dst_transform = t_transform,
            dst_crs       = t_crs,
            src_nodata    = nodata_fill,
            dst_nodata    = nodata_fill,
            resampling    = Resampling.bilinear,
        )

    dst[dst == nodata_fill] = np.nan
    return dst


# --------------------------------------------------------------
# Alignment channel
# --------------------------------------------------------------

def make_align_channel(ms, mode):
    """
    Build a normalized float32 [0,1] single-band image for offset search.
    NaN pixels (no-data borders) are replaced by the median so they
    do not bias the template matching correlation.

    NDVI is recommended for forests: dead trees have very low NDVI
    while live vegetation is high -> high spatial contrast -> reliable matches.
    """
    eps = 1e-6
    if mode == 'ndvi':
        nir = ms[:, :, BAND_NIR].astype(np.float32)
        red = ms[:, :, BAND_RED].astype(np.float32)
        ch  = (nir - red) / (nir + red + eps)
    elif mode == 'nir':
        ch = ms[:, :, BAND_NIR].astype(np.float32)
    else:
        ch = np.nanmean(ms, axis=2).astype(np.float32)

    median = float(np.nanmedian(ch))
    ch = np.where(np.isnan(ch), median, ch)
    lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
    hi = hi if hi > lo else lo + eps
    return np.clip((ch - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------
# Step 2: Template matching -> coarse offset
# --------------------------------------------------------------

def find_offset_template(target_ms, source_ms,
                         channel=ALIGN_CHANNEL,
                         search_radius=SEARCH_RADIUS):
    """
    Use cv2.matchTemplate (TM_CCOEFF_NORMED) to find the pixel offset
    between source and target after reprojection.

    Strategy:
      - Take the central 60% of the target as the template (avoids NaN borders)
      - Search for that patch in source within a +/- search_radius window
      - The correlation peak gives the offset

    Returns (dx, dy, match_score):
      dx, dy      : offset in pixels (how much source is shifted vs target)
      match_score : peak correlation value in [0,1]
                    >0.3 = reliable, >0.6 = very good
    """
    t_ch = make_align_channel(target_ms, channel)
    s_ch = make_align_channel(source_ms, channel)
    h, w = t_ch.shape

    # Template: central 60% of target
    margin_y = h // 5
    margin_x = w // 5
    ty1, ty2 = margin_y, h - margin_y
    tx1, tx2 = margin_x, w - margin_x
    template  = t_ch[ty1:ty2, tx1:tx2]

    # Search window in source
    sy1 = max(0, ty1 - search_radius)
    sy2 = min(h, ty2 + search_radius)
    sx1 = max(0, tx1 - search_radius)
    sx2 = min(w, tx2 + search_radius)
    search_region = s_ch[sy1:sy2, sx1:sx2]

    if (search_region.shape[0] < template.shape[0] or
            search_region.shape[1] < template.shape[1]):
        print("    [WARNING] Search window too small. Increase SEARCH_RADIUS.")
        return 0.0, 0.0, 0.0

    result      = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    match_x = max_loc[0] + sx1
    match_y = max_loc[1] + sy1
    dx = float(match_x - tx1)
    dy = float(match_y - ty1)

    return dx, dy, float(max_val)


# --------------------------------------------------------------
# Apply shift to all bands
# --------------------------------------------------------------

def shift_image(ms, dx, dy):
    """
    Shift ms by (dx, dy) pixels using bilinear interpolation.

    warpAffine convention (no WARP_INVERSE_MAP):
      dst(x,y) = src(x+dx, y+dy)  =>  M = [[1,0,-dx],[0,1,-dy]]
    """
    h, w, n = ms.shape
    M   = np.float32([[1, 0, -dx], [0, 1, -dy]])
    out = np.zeros_like(ms, dtype=np.float64)

    for b in range(n):
        band     = ms[:, :, b].copy()
        nan_mask = np.isnan(band)
        band[nan_mask] = 0.0
        warped = cv2.warpAffine(
            band.astype(np.float32), M, (w, h),
            flags      = cv2.INTER_LINEAR,
            borderMode = cv2.BORDER_CONSTANT,
            borderValue= 0.0,
        ).astype(np.float64)
        warped[nan_mask] = np.nan
        out[:, :, b] = warped

    return out


# --------------------------------------------------------------
# Step 3: ECC sub-pixel refinement
# --------------------------------------------------------------

def refine_ecc(target_ms, source_ms,
               channel=ALIGN_CHANNEL,
               max_iter=ECC_ITERATIONS):
    """
    Sub-pixel refinement using ECC (Enhanced Correlation Coefficient).
    Only called after template matching has brought images to within ~5 px,
    which is the convergence range of ECC.

    Uses MOTION_TRANSLATION (dx, dy only). Change to MOTION_EUCLIDEAN
    if there is a rotation between flights.
    """
    t_ch = make_align_channel(target_ms, channel)
    s_ch = make_align_channel(source_ms, channel)

    warp_mat = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iter, 1e-6)
    try:
        cc, warp_mat = cv2.findTransformECC(
            t_ch, s_ch, warp_mat,
            cv2.MOTION_TRANSLATION,
            criteria    = criteria,
            inputMask   = None,
            gaussFiltSize = 3,
        )
        print(f"    ECC sub-pixel : dx={warp_mat[0,2]:.3f} px  "
              f"dy={warp_mat[1,2]:.3f} px  CC={cc:.4f}")
    except cv2.error as e:
        print(f"    ECC did not converge ({e}). Using template matching result only.")
        warp_mat = np.eye(2, 3, dtype=np.float32)

    return shift_image(source_ms, warp_mat[0, 2], warp_mat[1, 2])


# --------------------------------------------------------------
# Metrics and visualization
# --------------------------------------------------------------

def cross_corr(t_ms, s_ms, band=BAND_NIR):
    """Normalized cross-correlation on one band. Used as relative indicator only."""
    t = t_ms[:, :, band].astype(np.float64)
    s = s_ms[:, :, band].astype(np.float64)
    mask = ~np.isnan(t) & ~np.isnan(s)
    if mask.sum() < 100:
        return 0.0
    t, s = t[mask] - t[mask].mean(), s[mask] - s[mask].mean()
    denom = np.sqrt(np.sum(t**2) * np.sum(s**2))
    return float(np.sum(t * s) / (denom + 1e-10))


def make_preview(ms, rgb=RGB_PREVIEW):
    """Build a uint8 RGB preview image from a multispectral array."""
    n = ms.shape[2]
    channels = []
    for b in rgb:
        ch  = ms[:, :, min(b, n-1)].astype(np.float64)
        med = float(np.nanmedian(ch))
        ch  = np.where(np.isnan(ch), med, ch)
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        hi = hi if hi > lo else lo + 1e-6
        channels.append(
            np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        )
    return np.stack(channels, axis=2)


def save_comparison(target_ms, warped_ms, output_dir, cc_before, cc_after):
    """Save a 3-panel comparison figure: target | registered | NIR difference."""
    t_nir = np.where(np.isnan(target_ms[:,:,BAND_NIR]), 0,
                     target_ms[:,:,BAND_NIR]).astype(np.float64)
    s_nir = np.where(np.isnan(warped_ms[:,:,BAND_NIR]), 0,
                     warped_ms[:,:,BAND_NIR]).astype(np.float64)
    valid = (t_nir > 0) & (s_nir > 0)
    if valid.sum() > 100:
        lo  = min(np.percentile(t_nir[valid], 2), np.percentile(s_nir[valid], 2))
        hi  = max(np.percentile(t_nir[valid], 98), np.percentile(s_nir[valid], 98))
        hi  = hi if hi > lo else lo + 1
        t_n = np.clip((t_nir - lo) / (hi - lo), 0, 1)
        s_n = np.clip((s_nir - lo) / (hi - lo), 0, 1)
        diff = np.abs(t_n - s_n)
    else:
        diff = np.zeros_like(t_nir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(make_preview(target_ms))
    axes[0].set_title("Target (reference)")
    axes[0].axis("off")

    axes[1].imshow(make_preview(warped_ms))
    axes[1].set_title("Source (registered)")
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title("NIR difference\n(dark = well aligned)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.suptitle(
        f"CC before registration: {cc_before:.4f}  ->  after: {cc_after:.4f}\n"
        f"Note: low CC is expected with high inter-annual vegetation change",
        fontsize=11
    )
    plt.tight_layout()
    out = os.path.join(output_dir, "registration_result.jpg")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    Preview : {out}")
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------
# Core registration function (single image)
# --------------------------------------------------------------

def register_one(target_f, t_prof, t_tr, t_crs, t_nd, t_dtype,
                 source_path, output_path, verbose=True):
    """
    Full registration pipeline for one source image.
    target_f must already be loaded as float64 with NaN for nodata.
    Returns cc_final, or None on error.
    """
    source_ms, s_prof, s_tr, s_crs, s_nd, s_dtype = read_geotiff(str(source_path))
    s_fill = float(s_nd) if s_nd is not None else 0.0
    source_f = source_ms.astype(np.float64)
    source_f[source_f == s_fill] = np.nan

    if verbose:
        s_gsd = abs(s_tr.a)
        t_gsd = abs(t_tr.a)
        print(f"    GSD  : {s_gsd:.4f} -> {t_gsd:.4f} m/px")
        print(f"    Shape: {source_ms.shape[:2]} -> {target_f.shape[:2]}")

    # Step 1: reproject
    source_snapped = snap_to_grid(
        source_f, s_tr, s_crs, t_prof, t_tr, t_crs, s_fill)
    cc_rep = cross_corr(target_f, source_snapped)
    if verbose:
        print(f"    CC after reprojection : {cc_rep:.4f}")

    # Step 2: template matching
    dx, dy, score = find_offset_template(
        target_f, source_snapped,
        channel=ALIGN_CHANNEL,
        search_radius=SEARCH_RADIUS)
    source_shifted = shift_image(source_snapped, dx, dy)
    if verbose:
        t_gsd = abs(t_tr.a)
        print(f"    Template match score  : {score:.4f}  "
              f"dx={dx:.0f}px ({dx*t_gsd:.2f}m)  "
              f"dy={dy:.0f}px ({dy*t_gsd:.2f}m)")
    if score < 0.2 and verbose:
        print("    [WARNING] Low match score. Check ALIGN_CHANNEL or SEARCH_RADIUS.")

    # Step 3: ECC sub-pixel
    source_final = refine_ecc(target_f, source_shifted,
                              channel=ALIGN_CHANNEL,
                              max_iter=ECC_ITERATIONS)
    cc_final = cross_corr(target_f, source_final)
    if verbose:
        print(f"    CC final              : {cc_final:.4f}")

    write_geotiff(str(output_path), source_final,
                  t_prof, t_tr, t_crs, s_nd, s_dtype)

    return cc_final, cc_rep, dx, dy, score


# --------------------------------------------------------------
# Single image mode
# --------------------------------------------------------------

def main():
    print("=" * 60)
    print("  MULTISPECTRAL CO-REGISTRATION  (single image)")
    print("=" * 60)

    target_path = os.path.join(INPUT_DIR, TARGET_IMAGE)
    source_path = os.path.join(INPUT_DIR, SOURCE_IMAGE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for p in [target_path, source_path]:
        if not os.path.exists(p):
            print(f"\n[ERROR] File not found: {p}")
            sys.exit(1)

    print(f"\n  Target : {TARGET_IMAGE}")
    print(f"  Source : {SOURCE_IMAGE}")

    target_ms, t_prof, t_tr, t_crs, t_nd, t_dtype = read_geotiff(target_path)
    t_fill   = float(t_nd) if t_nd is not None else 0.0
    target_f = target_ms.astype(np.float64)
    target_f[target_f == t_fill] = np.nan

    print(f"  Target : {target_ms.shape}  GSD={abs(t_tr.a):.4f} m  dtype={t_dtype}")

    out_path = os.path.join(OUTPUT_DIR, SOURCE_IMAGE)
    print()
    cc_final, cc_rep, dx, dy, score = register_one(
        target_f, t_prof, t_tr, t_crs, t_nd, t_dtype,
        source_path, out_path, verbose=True)

    # Save target as reference
    target_out_path = os.path.join(OUTPUT_DIR, "target_reference.tif")
    write_geotiff(target_out_path, target_f,
                  t_prof, t_tr, t_crs, t_nd, t_dtype)

    # Reload warped for comparison figure
    warped_ms, *_ = read_geotiff(out_path)
    warped_f = warped_ms.astype(np.float64)
    save_comparison(target_f, warped_f, OUTPUT_DIR, cc_rep, cc_final)

    print(f"\n{'='*60}")
    if cc_final > 0.85:
        verdict = "EXCELLENT"
    elif cc_final > 0.70:
        verdict = "GOOD"
    elif cc_final > 0.45:
        verdict = "ACCEPTABLE - verify visually in QGIS"
    else:
        verdict = "REVIEW - check stable landmarks in QGIS"
    print(f"  Result : {verdict}  (CC={cc_final:.4f})")
    print(f"  Note   : CC < 0.5 is normal with high inter-annual vegetation change")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"{'='*60}")


# --------------------------------------------------------------
# Batch mode
# --------------------------------------------------------------

def batch_register():
    print("=" * 60)
    print("  MULTISPECTRAL CO-REGISTRATION  (batch)")
    print("=" * 60)

    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only .tif/.tiff files, excluding the target and ArcGIS auxiliary files
    all_tifs = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ('.tif', '.tiff')
        and f.name != TARGET_IMAGE
    ])

    if not all_tifs:
        print(f"[ERROR] No .tif files found in {input_dir}")
        return

    target_path = input_dir / TARGET_IMAGE
    if not target_path.exists():
        print(f"[ERROR] Target not found: {target_path}")
        return

    print(f"\n  Target    : {TARGET_IMAGE}")
    print(f"  Images    : {len(all_tifs)}")
    print(f"  Input dir : {input_dir}")
    print(f"  Output dir: {output_dir}\n")

    # Load target once
    target_ms, t_prof, t_tr, t_crs, t_nd, t_dtype = read_geotiff(str(target_path))
    t_fill   = float(t_nd) if t_nd is not None else 0.0
    target_f = target_ms.astype(np.float64)
    target_f[target_f == t_fill] = np.nan
    print(f"  Target shape : {target_ms.shape}  GSD={abs(t_tr.a):.4f} m\n")

    results = []
    for i, src_path in enumerate(all_tifs, 1):
        print(f"[{i}/{len(all_tifs)}] {src_path.name}")
        out_path = output_dir / src_path.name
        try:
            cc_final, cc_rep, dx, dy, score = register_one(
                target_f, t_prof, t_tr, t_crs, t_nd, t_dtype,
                src_path, out_path, verbose=True)
            status = (f"OK  CC={cc_final:.3f}  "
                      f"score={score:.3f}  "
                      f"dx={dx:.0f}px  dy={dy:.0f}px")
        except Exception as e:
            status = f"ERROR: {e}"
            print(f"    {status}")
        results.append((src_path.name, status))
        print()

    # Summary
    ok  = [r for r in results if r[1].startswith("OK")]
    err = [r for r in results if r[1].startswith("ERROR")]
    warn = [r for r in results
            if r[1].startswith("OK") and
            float(r[1].split("score=")[1].split()[0]) < 0.3]

    print(f"{'='*60}")
    print(f"  BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"  Completed   : {len(ok)}/{len(all_tifs)}")
    if warn:
        print(f"  Low score (<0.3) - verify in QGIS:")
        for name, _ in warn:
            print(f"    {name}")
    if err:
        print(f"  Errors:")
        for name, msg in err:
            print(f"    {name}: {msg}")
    print(f"  Output      : {output_dir}")
    print(f"{'='*60}")


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------

if __name__ == "__main__":
    if MODE == 'single':
        main()
    elif MODE == 'batch':
        batch_register()
    else:
        print(f"[ERROR] MODE must be 'single' or 'batch', got '{MODE}'")
        sys.exit(1)
