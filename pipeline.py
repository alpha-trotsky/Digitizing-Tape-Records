#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — Handwritten package-code extractor (steps 0–7)

Implements the row-first, separator-first pipeline we discussed:
  0) Preprocess image (grayscale, normalize, threshold, deskew, optional dewarp)
  1) Detect vertical separators FIRST and split the page into regions
  2) Build horizontal row bands via ruled lines (assign tokens to nearest band ABOVE)
  3) OCR → tokens (text, confidence, quad box) per region
  4) Within each row band: merge tokens left→right → normalize → classify (FULL / SUFFIX / OTHER)
  5) Place rows into columns (per region): by separator if present, else by largest x-gap
  6) Reconstruct codes without arrows: prefix from FULL + 2-digit SUFFIX rows below
  7) Strike-through detection, regex validation, export CSV

Notes
- OCR engine: PaddleOCR (open source). Install: `pip install paddleocr shapely` (plus system deps).
- Image pre-processing: OpenCV (cv2). Install: `pip install opencv-python`
- For CSV writing we use the stdlib `csv` module (no pandas required).
- Regex families start strict but are easy to edit in CONFIG below.
"""

from __future__ import annotations
import os
import sys
import csv
import math
import argparse
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# --- CONFIG ------------------------------------------------------------------

class CONFIG:
    # Ambiguous character mapping in the NUMERIC TAIL ONLY (we won't mutate the letter prefix)
    AMBIG_MAP = str.maketrans({'O':'0','I':'1','S':'5','Z':'2','B':'8','Q':'0'})

    # Regex families for FULL codes.
    # Edit these to match exactly what you expect.
    #  - Two letters + 7 digits     (e.g., UA0225517, DA0447850)
    #  - One letter + 9xxxxxxx      (e.g., D90108542)  -> 1 letter + '9' + 7 digits = total 9 chars
    #  - Optional 3-letter prefix family (e.g., VAD214653) seen in your samples; adjust length once fixed.
    REGEX_FULL = r'^(?:[A-Z]{2}\d{7}|[A-Z]{1}\d{8}|[A-Z]{2}\d{7})$'
    REGEX_SUFFIX2 = r'^\d{2}$'

    # Confidence thresholds (empirical, not probabilities)
    TOKEN_CONF_FLOOR = 0.45      # drop raw OCR tokens below this before merging
    MERGED_LOW_CONF = 0.55       # if merged row fails regex and conf < this, ignore row
    STRIKE_CONF_HINT = 0.60      # low-ish conf + strong horizontal stroke → crossed_out

    # Merge gap factor: max gap between adjacent tokens (in px) relative to median char height of the row
    MERGE_GAP_FACTOR = 1

    # Column detection
    MAX_COLUMNS_PER_REGION = 3
    MIN_LARGEST_GAP_PX = 20     # if the largest gap between x-centers < this, treat as 1 column

    #WHY ARE WE DOING 60 POTENTIALLY FIX???

    # Hough/line-detection params
    HOUGH_RHO = 1
    HOUGH_THETA = math.pi / 180.0
    HOUGH_THRESH = 200

    # Vertical separator min length as fraction of image height
    VSEP_MIN_HEIGHT_FRAC = 0.3

    # Strike-through horizontal line coverage threshold (fraction of box width)
    STRIKE_HLINE_FRAC = 0.5


# --- DATA TYPES ---------------------------------------------------------------

@dataclass
class Token:
    text: str
    conf: float
    box: List[Tuple[float, float]]  # quadrilateral [(x1,y1),...,(x4,y4)]
    x_center: float
    y_center: float
    row_band: int = -1
    region_id: int = 0


@dataclass
class RowCandidate:
    region_id: int
    band_idx: int
    x_center: float
    y_center: float
    text_merged: str
    conf_merged: float
    box_merged: Tuple[int,int,int,int]  # (x,y,w,h) axis-aligned bbox
    klass: str  # 'FULL' | 'SUFFIX' | 'OTHER'


@dataclass
class EmittedCode:
    region_id: int
    column_id: int
    band_idx: int
    y_center: float
    raw_merged: str
    code: str
    conf: float
    crossed_out: bool
    klass: str  # 'FULL' | 'RECONSTRUCTED'


# --- UTILITIES ---------------------------------------------------------------

def _safe_import_cv2():
    try:
        import cv2
        return cv2
    except Exception as e:
        sys.stderr.write("ERROR: OpenCV (cv2) is required. Install with `pip install opencv-python`.\n")
        raise

def _safe_import_paddle():
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR
    except Exception as e:
        sys.stderr.write("ERROR: PaddleOCR is required. Install with `pip install paddleocr shapely` and try again.\n")
        raise

def quad_to_aabb(box: List[Tuple[float,float]]) -> Tuple[int,int,int,int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return x0, y0, x1 - x0, y1 - y0

def box_center(box: List[Tuple[float,float]]) -> Tuple[float,float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs)/4.0, sum(ys)/4.0

def normalize_candidate(s: str) -> str:
    """Uppercase, strip spaces; convert ambiguous letters to digits only in numeric tail."""
    import re
    s = (s or "").replace(" ", "").upper()
    m = re.match(r'^([A-Z]+)(\d+)$', s)
    if m:
        return m.group(1) + m.group(2).translate(CONFIG.AMBIG_MAP)
    return s.translate(CONFIG.AMBIG_MAP)

def classify_candidate(s: str) -> str:
    import re
    if re.match(CONFIG.REGEX_FULL, s or ""):
        return "FULL"
    if re.match(CONFIG.REGEX_SUFFIX2, s or ""):
        return "SUFFIX"
    return "OTHER"


# --- STEP 0 & 1: PREPROCESS + VERTICAL SEPARATORS ----------------------------

def preprocess_image(path: str) -> Dict[str, Any]:
    """Step 0 pre-process: grayscale → CLAHE → adaptive threshold → deskew; optional dewarp.
       Returns dict with 'orig','gray','binary','rotated','M' (rotation matrix) and geometry.
    """
    cv2 = _safe_import_cv2()
    import numpy as np

    orig = cv2.imread(path)
    if orig is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = orig.shape[:2]

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug_01_grayscale.png", gray)

    # Contrast normalize (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    cv2.imwrite("debug_02_clahe.png", gray_eq)

    # Adaptive threshold (binary)
    binary = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    cv2.imwrite("debug_03_binary_initial.png", binary)

    # Invert so ink is 1, background 0 for morphology convenience
    bin_inv = cv2.bitwise_not(binary)
    cv2.imwrite("debug_04_binary_inverted_initial.png", bin_inv)

    # Deskew by dominant horizontal lines
    angle = estimate_skew_angle(bin_inv)
    rotated = rotate_image(gray_eq, -angle)
    cv2.imwrite("debug_05_rotated_grayscale.png", rotated)
    # Recompute binary after rotation for clean lines
    binary_rot = cv2.adaptiveThreshold(
        rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    cv2.imwrite("debug_06_binary_rotated.png", binary_rot)
    bin_inv_rot = cv2.bitwise_not(binary_rot)
    cv2.imwrite("debug_07_binary_inverted_rotated.png", bin_inv_rot)

    return {
        "orig": orig,
        "gray": gray_eq,
        "binary": binary_rot,
        "bin_inv": bin_inv_rot,
        "h": h, "w": w,
        "angle": angle,
    }

def estimate_skew_angle(bin_inv):
    """Estimate skew via Hough on horizontal lines."""
    cv2 = _safe_import_cv2()
    import numpy as np
    edges = cv2.Canny(bin_inv, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, CONFIG.HOUGH_RHO, CONFIG.HOUGH_THETA, CONFIG.HOUGH_THRESH)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines:
        rho, theta = rho_theta[0]
        # Convert to degrees; horizontal lines ~ 0° or 180°; we want small deviation from 0
        deg = (theta * 180.0 / math.pi) - 90.0
        # Keep near-horizontal angles
        if abs(deg) < 20:
            angles.append(deg)
    if not angles:
        return 0.0
    return float(np.median(angles))

def rotate_image(img, angle_deg):
    cv2 = _safe_import_cv2()
    import numpy as np
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def detect_vertical_separators(bin_inv_rot, min_height_frac=CONFIG.VSEP_MIN_HEIGHT_FRAC):
    """Step 1: detect long vertical lines; return list of x positions to split regions."""
    cv2 = _safe_import_cv2()
    import numpy as np
    h, w = bin_inv_rot.shape[:2]

    # Emphasize vertical strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 50)))
    vert = cv2.morphologyEx(bin_inv_rot, cv2.MORPH_OPEN, kernel, iterations=1)

    edges = cv2.Canny(vert, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi/180, threshold=100,
        minLineLength=int(min_height_frac * h), maxLineGap=10
    )

    xs = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            # near-vertical if dx is small
            if abs(x2 - x1) < max(5, w // 200):
                xs.append(int((x1 + x2) / 2))

    if not xs:
        return []

    # Merge close x's
    xs.sort()
    merged = []
    group = [xs[0]]
    for x in xs[1:]:
        if abs(x - group[-1]) <= max(10, w // 100):
            group.append(x)
        else:
            merged.append(int(sum(group)/len(group)))
            group = [x]
    merged.append(int(sum(group)/len(group)))
    # Filter borders (within 2% of edges)
    merged = [x for x in merged if x > 0.02*w and x < 0.98*w]
    return merged

def split_regions(w, separators: List[int]) -> List[Tuple[int,int]]:
    """Return list of (x0,x1) regions split by vertical separators."""
    xs = [0] + sorted(separators) + [w]
    regions = []
    for i in range(len(xs)-1):
        x0, x1 = xs[i], xs[i+1]
        # leave a small gutter around separators
        regions.append((int(x0), int(x1)))
    return regions


# --- STEP 2: HORIZONTAL BANDS (NEAREST-ABOVE RULE) ---------------------------

def detect_horizontal_bands(bin_inv_rot) -> List[int]:
    """Detect y positions of ruled lines; return sorted list of y's (band edges).
       If none detected, synthesize bands from horizontal projection minima.
    """
    cv2 = _safe_import_cv2()
    import numpy as np
    h, w = bin_inv_rot.shape[:2]

    # Emphasize horizontal strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 40), 1))
    horiz = cv2.morphologyEx(bin_inv_rot, cv2.MORPH_OPEN, kernel, iterations=1)

    edges = cv2.Canny(horiz, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi/180, threshold=100,
        minLineLength=int(0.6 * w), maxLineGap=10
    )
    ys = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            if abs(y2 - y1) < 5:  # near-horizontal
                ys.append(int((y1 + y2) / 2))
    if ys:
        ys = sorted(set(ys))
        # Add top/bottom virtual lines
        return [0] + ys + [h]

    # Fallback: projection profile to synthesize rows
    proj = bin_inv_rot.sum(axis=1)  # more ink → larger values at text rows
    # Find troughs as potential separators → here we just split every ~line_height
    approx_line = max(20, h // 50)
    band_edges = list(range(0, h, approx_line))
    if band_edges[-1] != h:
        band_edges.append(h)
    return band_edges

def assign_band_above(yc: float, band_edges: List[int]) -> int:
    """Return index i such that band i spans [y_i, y_{i+1}) and y_i <= yc < y_{i+1} using the 'nearest ABOVE' rule."""
    # band_edges is sorted list of y's; we pick the largest y_i <= yc
    lo, hi = 0, len(band_edges) - 2
    idx = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if band_edges[mid] <= yc:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return idx


# --- STEP 3: OCR → TOKENS ----------------------------------------------------

def run_ocr_on_region(ocr_engine, gray_rot, region: Tuple[int,int]) -> List[Token]:
    """Run OCR on a vertical region and return tokens in absolute coords."""
    import numpy as np
    x0, x1 = region
    crop = gray_rot[:, x0:x1]

    result = ocr_engine.ocr(crop, cls=True)
    tokens: List[Token] = []
    if not result:
        return tokens

    for line in result:
        for box, (text, conf) in line:
            # box is 4 points in crop coords; convert to absolute
            abs_box = [(p[0] + x0, p[1]) for p in box]
            xc, yc = box_center(abs_box)
            tokens.append(Token(
                text=text or "",
                conf=float(conf or 0.0),
                box=abs_box,
                x_center=float(xc),
                y_center=float(yc),
                region_id=x0  # use region's left x as id
            ))
    # Filter low-confidence tokens
    tokens = [t for t in tokens if t.conf >= CONFIG.TOKEN_CONF_FLOOR]
    return tokens


# --- STEP 4: MERGE TOKENS PER ROW → CLASSIFY ---------------------------------

def merge_tokens_in_band(tokens: List[Token], band_idx: int, band_edges: List[int]) -> List[RowCandidate]:
    """Within a single band, sort by x and merge small-gap neighbors to produce one candidate string."""
    import re
    if not tokens:
        return []

    # Sort by x
    ts = sorted(tokens, key=lambda t: t.x_center)

    # Estimate a median character height for gap thresholding
    heights = []
    for t in ts:
        x, y, w, h = quad_to_aabb(t.box)
        heights.append(h)
    median_h = max(1, int(sorted(heights)[len(heights)//2]))
    max_gap = CONFIG.MERGE_GAP_FACTOR * median_h

    # Greedy merge left→right based on gap between adjacent token boxes
    merged_items = []
    current_texts = []
    current_boxes = []
    current_confs = []

    def flush_current():
        nonlocal merged_items, current_texts, current_boxes, current_confs
        if not current_texts:
            return
        # Merge text
        raw = "".join(current_texts)
        s = normalize_candidate(raw)
        klass = classify_candidate(s)
        # Merge geometry
        xs = [b[0] for b in current_boxes] + [b[0]+b[2] for b in current_boxes]
        ys = [b[1] for b in current_boxes] + [b[1]+b[3] for b in current_boxes]
        x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        xc, yc = (x0+x1)/2.0, (y0+y1)/2.0
        conf = sum(current_confs)/len(current_confs)
        merged_items.append(RowCandidate(
            region_id=tokens[0].region_id,
            band_idx=band_idx,
            x_center=xc, y_center=yc,
            text_merged=s, conf_merged=conf,
            box_merged=(x0, y0, x1-x0, y1-y0),
            klass=klass
        ))
        # reset
        current_texts, current_boxes, current_confs = [], [], []

    prev_right = None
    for t in ts:
        x, y, w, h = quad_to_aabb(t.box)
        left, right = x, x + w
        if prev_right is None or (left - prev_right) <= max_gap:
            current_texts.append(t.text)
            current_boxes.append((x,y,w,h))
            current_confs.append(t.conf)
        else:
            flush_current()
            current_texts = [t.text]
            current_boxes = [(x,y,w,h)]
            current_confs = [t.conf]
        prev_right = right
    flush_current()

    # Drop very low-quality non-matching rows
    pruned = []
    for rc in merged_items:
        if rc.klass == "OTHER" and rc.conf_merged < CONFIG.MERGED_LOW_CONF:
            continue
        pruned.append(rc)
    return pruned


# --- STEP 5: COLUMNS PER REGION ---------------------------------------------

def infer_columns_from_fulls(row_items: List[RowCandidate], region: Tuple[int,int]) -> Dict[int, int]:
    """Return mapping row_index -> column_id using FULL rows' x-centers.
       Strategy: if <2 FULLs, just 1 column. Else split by largest gap up to MAX_COLUMNS_PER_REGION.
    """
    x0, x1 = region
    fulls = [(i, rc.x_center) for i, rc in enumerate(row_items) if rc.klass == "FULL"]
    if len(fulls) < 2:
        # single column
        return {i: 0 for i in range(len(row_items))}

    # Sort by x center
    fulls_sorted = sorted(fulls, key=lambda p: p[1])
    xs = [x for _, x in fulls_sorted]
    # Find largest gaps
    gaps = [(j, xs[j+1]-xs[j]) for j in range(len(xs)-1)]
    if not gaps:
        return {i: 0 for i in range(len(row_items))}

    # If the largest gap is small, stick to one column
    j_max, g_max = max(gaps, key=lambda kv: kv[1])
    if g_max < CONFIG.MIN_LARGEST_GAP_PX:
        return {i: 0 for i in range(len(row_items))}

    # Otherwise split at top K-1 gaps (K columns), but cap at MAX_COLUMNS_PER_REGION
    # For typical pages, K=2; if you truly have 3, this will handle it
    gaps_sorted = sorted(gaps, key=lambda kv: kv[1], reverse=True)
    K = min(CONFIG.MAX_COLUMNS_PER_REGION, 1 + len(gaps_sorted))
    cut_positions = sorted([xs[j] for j, _ in gaps_sorted[:K-1]])

    # Assign each row to column by its x_center relative to cuts
    def col_id(xc):
        cid = 0
        for cut in cut_positions:
            if xc > cut:
                cid += 1
        return cid

    # Use FULL rows to define columns; SUFFIX/OTHER adopt nearest FULL above later
    col_map = {}
    for i, rc in enumerate(row_items):
        col_map[i] = col_id(rc.x_center)
    return col_map


# --- STEP 6: RECONSTRUCT WITHOUT ARROWS -------------------------------------

def reconstruct_codes_in_region(rows: List[RowCandidate],
                                col_map: Dict[int,int]) -> List[EmittedCode]:
    """Scan per column top→bottom; FULL sets prefix=[:-2], SUFFIX becomes prefix+suffix. Reset at next FULL."""
    # group rows by column, ordered by band_idx (top to bottom)
    by_col: Dict[int, List[Tuple[int, RowCandidate]]] = {}
    for idx, rc in enumerate(rows):
        cid = col_map.get(idx, 0)
        by_col.setdefault(cid, []).append((idx, rc))

    emissions: List[EmittedCode] = []
    for cid, lst in by_col.items():
        lst.sort(key=lambda t: t[1].band_idx)
        prefix = None
        for _, rc in lst:
            # strike-through detection later (step 7); we pass crossed_out=False here
            if rc.klass == "FULL":
                code = rc.text_merged
                emissions.append(EmittedCode(
                    region_id=rc.region_id, column_id=cid, band_idx=rc.band_idx,
                    y_center=rc.y_center, raw_merged=rc.text_merged, code=code,
                    conf=rc.conf_merged, crossed_out=False, klass="FULL"
                ))
                if len(code) >= 2 and code[-2:].isdigit():
                    prefix = code[:-2]
                else:
                    prefix = None
            elif rc.klass == "SUFFIX" and prefix:
                code = prefix + rc.text_merged
                emissions.append(EmittedCode(
                    region_id=rc.region_id, column_id=cid, band_idx=rc.band_idx,
                    y_center=rc.y_center, raw_merged=rc.text_merged, code=code,
                    conf=rc.conf_merged, crossed_out=False, klass="RECONSTRUCTED"
                ))
            else:
                # ignore OTHER in v1
                pass
    return emissions


# --- STEP 7: STRIKE-THROUGH + VALIDATE + EXPORT ------------------------------

def detect_strike_through(bin_inv_rot, box_merged: Tuple[int,int,int,int], conf_merged: float) -> bool:
    """Heuristic: low-ish conf + a long near-horizontal stroke crossing >= 70% of the box width."""
    cv2 = _safe_import_cv2()
    import numpy as np
    x, y, w, h = box_merged
    h_img, w_img = bin_inv_rot.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return False
    roi = bin_inv_rot[y0:y1, x0:x1]

    # Emphasize horizontal strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 3), 1))
    horiz = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

    edges = cv2.Canny(horiz, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi/180, threshold=20,
        minLineLength=int(CONFIG.STRIKE_HLINE_FRAC * w), maxLineGap=5
    )
    strong_line = lines is not None and len(lines) > 0

    return (conf_merged < CONFIG.STRIKE_CONF_HINT) and strong_line

def validate_against_regex(code: str) -> bool:
    import re
    return re.match(CONFIG.REGEX_FULL, code or "") is not None

def export_csv(records: List[EmittedCode], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "column", "band", "y_px", "raw_merged", "code", "conf", "crossed_out", "klass", "valid_shape"])
        for r in records:
            w.writerow([r.region_id, r.column_id, r.band_idx, int(r.y_center),
                        r.raw_merged, r.code, f"{r.conf:.3f}", "TRUE" if r.crossed_out else "FALSE",
                        r.klass, "TRUE" if validate_against_regex(r.code) else "FALSE"])


# --- ORCHESTRATOR ------------------------------------------------------------

def process_image_to_csv(img_path: str, out_csv: str):
    """
    Full pipeline for one image → CSV.
    """
    PaddleOCR = _safe_import_paddle()
    cv2 = _safe_import_cv2()
    import numpy as np
    import re

    prep = preprocess_image(img_path)
    gray = prep["gray"]
    binary = prep["binary"]
    bin_inv = prep["bin_inv"]
    h, w = prep["h"], prep["w"]

    # Step 1: separators → regions
    seps = detect_vertical_separators(bin_inv)
    regions = split_regions(w, seps)  # list of (x0,x1)

    # Init OCR engine once
    ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)

    all_emissions: List[EmittedCode] = []

    # Process each region independently
    for region in regions:
        x0, x1 = region
        # Bands from horizontal lines
        band_edges = detect_horizontal_bands(bin_inv[:, x0:x1])
        # Offset band edges to absolute y by adding 0 since we didn't crop vertically
        # Assign tokens to nearest ABOVE band
        tokens = run_ocr_on_region(ocr, gray, region)

        # assign band indices
        for t in tokens:
            # NOTE: band_edges were computed on the region slice (same y as full image)
            bi = assign_band_above(t.y_center, band_edges)
            t.row_band = bi

        # Build row candidates per band
        rows: List[RowCandidate] = []
        bands_with_tokens = {}
        for t in tokens:
            bands_with_tokens.setdefault(t.row_band, []).append(t)
        for bi, toklist in sorted(bands_with_tokens.items(), key=lambda kv: kv[0]):
            rows.extend(merge_tokens_in_band(toklist, bi, band_edges))

        # Infer columns (per region) using FULL rows
        col_map = infer_columns_from_fulls(rows, region)
 
        # Reconstruct
        emissions = reconstruct_codes_in_region(rows, col_map)

        # Strike-through detection + regex validity tagging
        for e in emissions:
            # Find the source row candidate to get its merged box
            # (map by (band_idx, raw_merged) heuristic)
            rc_match = next((rc for rc in rows if rc.band_idx == e.band_idx and rc.text_merged == e.raw_merged), None)
            if rc_match:
                crossed = detect_strike_through(bin_inv, rc_match.box_merged, e.conf)
                e.crossed_out = crossed
            else:
                e.crossed_out = False  # fallback
        all_emissions.extend(emissions)

    # Export
    export_csv(all_emissions, out_csv)


# --- CLI ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Extract handwritten package codes from a photo → CSV.")
    p.add_argument("image", help="Path to input image (JPEG/PNG).")
    p.add_argument("-o", "--out", help="Path to output CSV (default: <image>.csv).", default=None)
    args = p.parse_args()

    img_path = args.image
    out_csv = args.out or (os.path.splitext(img_path)[0] + ".csv")

    process_image_to_csv(img_path, out_csv)
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()
