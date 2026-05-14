"""
setup_mango_dataset.py
=======================
Copies 20 images per class from MangoLeafBD Dataset into MangoDataset/
for CNN training in AgriAssist+.

Mapping:
  Alphonso_Healthy    ← Healthy/ (images 1–20)
  ImamPasand_Healthy  ← Healthy/ (images 21–40)
  Alphonso_Diseased   ← Anthracnose/ (images 1–20)
  ImamPasand_Diseased ← Die Back/ (images 1–20)

Run:
    cd D:\Agri_Assist
    python setup_mango_dataset.py
"""

import os
import shutil

SRC_BASE = r"D:\MangoLeafBD Dataset"
DST_BASE = r"D:\Agri_Assist\MangoDataset"

PLAN = [
    # (dst_folder,           src_folder,    start, count, prefix)
    ("Alphonso_Healthy",    "Healthy",      0,     20,    "alphonso_healthy"),
    ("ImamPasand_Healthy",  "Healthy",      20,    20,    "imampasand_healthy"),
    ("Alphonso_Diseased",   "Anthracnose",  0,     20,    "alphonso_diseased"),
    ("ImamPasand_Diseased", "Die Back",     0,     20,    "imampasand_diseased"),
]

total_copied = 0

for dst_folder, src_folder, start, count, prefix in PLAN:
    src_dir = os.path.join(SRC_BASE, src_folder)
    dst_dir = os.path.join(DST_BASE, dst_folder)
    os.makedirs(dst_dir, exist_ok=True)

    all_files = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    selected = all_files[start:start + count]

    copied = 0
    for i, fname in enumerate(selected):
        src_path = os.path.join(src_dir, fname)
        new_name = f"{prefix}_{i+1:02d}.jpg"
        dst_path = os.path.join(dst_dir, new_name)
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"  ✅ {dst_folder:<25} — {copied} images copied  (source: {src_folder}/)")
    total_copied += copied

print(f"\n🎉 Done! Total {total_copied} images copied to D:\\Agri_Assist\\MangoDataset\\")
print("\nFolder structure:")
for dst_folder, *_ in PLAN:
    path = os.path.join(DST_BASE, dst_folder)
    n = len([f for f in os.listdir(path) if f.endswith(".jpg")])
    print(f"  {dst_folder}/  → {n} images")

print("\n✅ Ready to train! Run:  python train_mango_cnn.py")
