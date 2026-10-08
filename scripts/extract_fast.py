"""
Fast extraction: extract labels from small label ZIPs, then extract matching images.
Reads directly from Windows mount, extracting only needed files (not full ZIPs).

Usage:
  python extract_fast.py --samples-per-type 100
"""

import argparse
import json
import os
import random
import sys
import zipfile
from pathlib import Path

DATA_ROOT = Path("/mnt/c/projects/227.건설 현장 위험 상태 판단 데이터/01-1.정식개방데이터")
TRAIN_IMG_DIR = DATA_ROOT / "Training" / "01.원천데이터"
TRAIN_LABEL_DIR = DATA_ROOT / "Training" / "02.라벨링데이터"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "construction"
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

TYPE_MAP = {
    "추락": "fall",
    "낙하": "falling_object",
    "협착": "entrapment",
    "전도": "overturn",
    "화재": "fire",
}


def parse_zip_type(filename: str):
    for kr, en in TYPE_MAP.items():
        if kr in filename:
            is_hazardous = "비정상" in filename or "_N-" in filename
            return en, is_hazardous
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-type", type=int, default=100,
                        help="Samples per (type, hazard) pair. 10 pairs → total = 10x this value")
    args = parser.parse_args()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    label_zips = sorted([f for f in os.listdir(TRAIN_LABEL_DIR) if f.endswith('.zip')])
    print(f"Found {len(label_zips)} label ZIPs")

    # Group by type
    type_groups = {}  # (event_type, is_hazardous) → list of (label_zip, img_zip)
    for lz in label_zips:
        event_type, is_hazardous = parse_zip_type(lz)
        if event_type is None:
            continue
        img_zip = lz.replace("TL_", "TS_")
        key = (event_type, is_hazardous)
        type_groups.setdefault(key, []).append((lz, img_zip))

    print(f"\nType groups: {len(type_groups)}")
    for k, v in sorted(type_groups.items()):
        print(f"  {k[0]} ({'hazard' if k[1] else 'safe'}): {len(v)} ZIPs")

    # Phase 1: Extract labels (fast — label ZIPs are small ~5-15MB each)
    all_labels = {}  # stem → (label_data, event_type, is_hazardous)
    for (event_type, is_hazardous), zip_pairs in sorted(type_groups.items()):
        budget = args.samples_per_type
        status = "hazard" if is_hazardous else "safe"
        print(f"\n[Labels] {event_type}_{status} — budget: {budget}")

        for lz, iz in zip_pairs:
            if budget <= 0:
                break

            lz_path = os.path.join(TRAIN_LABEL_DIR, lz)
            try:
                with zipfile.ZipFile(lz_path, 'r') as zf:
                    jsons = [n for n in zf.namelist() if n.endswith('.json')]
                    random.shuffle(jsons)
                    take = min(len(jsons), budget, 50)  # max 50 per ZIP

                    for jn in jsons[:take]:
                        stem = Path(jn).stem
                        with zf.open(jn) as f:
                            data = json.loads(f.read().decode('utf-8-sig'))
                        all_labels[stem] = (data, event_type, is_hazardous)

                    budget -= take
                    print(f"  {lz}: extracted {take} labels (budget left: {budget})")
            except Exception as e:
                print(f"  ERROR {lz}: {e}")

    print(f"\nTotal labels extracted: {len(all_labels)}")

    # Phase 2: Extract matching images (slower — image ZIPs are large)
    print("\n" + "=" * 60)
    print("Phase 2: Extracting images...")
    print("=" * 60)

    needed_stems = set(all_labels.keys())
    found_images = set()

    # Check already extracted images
    for p in IMAGES_DIR.glob("*.jpg"):
        if p.stem in needed_stems:
            found_images.add(p.stem)
    print(f"Already have {len(found_images)} images")

    remaining = needed_stems - found_images
    if not remaining:
        print("All images already extracted!")
    else:
        print(f"Need to extract {len(remaining)} images")

        # Group needed stems by ZIP
        img_zips = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.zip')])
        for iz_name in img_zips:
            if not remaining:
                break

            iz_path = os.path.join(TRAIN_IMG_DIR, iz_name)
            try:
                with zipfile.ZipFile(iz_path, 'r') as zf:
                    img_names = {Path(n).stem: n for n in zf.namelist()
                                 if n.lower().endswith(('.jpg', '.jpeg', '.png'))}

                    to_extract = remaining & set(img_names.keys())
                    if not to_extract:
                        continue

                    print(f"  {iz_name}: extracting {len(to_extract)} images...", end="", flush=True)
                    for stem in to_extract:
                        img_member = img_names[stem]
                        data = zf.read(img_member)
                        out_path = IMAGES_DIR / f"{stem}.jpg"
                        with open(out_path, 'wb') as f:
                            f.write(data)
                        found_images.add(stem)
                        remaining.discard(stem)
                    print(" done")
            except Exception as e:
                print(f" ERROR: {e}")

    print(f"\nImages extracted: {len(found_images)}")
    print(f"Missing images: {len(needed_stems - found_images)}")

    # Phase 3: Generate training JSON
    print("\n" + "=" * 60)
    print("Phase 3: Generating training data...")

    # Save labels as individual JSONs (for reference)
    for stem, (data, _, _) in all_labels.items():
        label_path = LABELS_DIR / f"{stem}.json"
        if not label_path.exists():
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)

    print(f"Labels saved: {len(all_labels)}")
    print(f"Images available: {len(found_images)}")
    print(f"Output dirs:")
    print(f"  Images: {IMAGES_DIR}")
    print(f"  Labels: {LABELS_DIR}")


if __name__ == "__main__":
    main()
