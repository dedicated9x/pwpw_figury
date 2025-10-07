#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# predict_batch.py — YOLO inference dla wszystkich plików bezpośrednio w _inputs (bez rekurencji)

from pathlib import Path
import argparse
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# -------- CFG (stałe, nie-ścieżkowe) --------
CFG = {
    "imgsz": 1280,
    "conf": 0.35,
    "iou": 0.55,
    "max_det": 4000,
    "shrink": 0.55,          # zmniejszenie W,H bboxa
    "thickness": 2,
    "font_scale": 0.6,
    "class_names": ["square", "circle", "triangle"],
}
# --------------------------------------------

def shrink_box(x1, y1, x2, y2, shrink, W, H):
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    w = (x2 - x1) * shrink; h = (y2 - y1) * shrink
    nx1 = int(max(0, cx - w/2)); ny1 = int(max(0, cy - h/2))
    nx2 = int(min(W-1, cx + w/2)); ny2 = int(min(H-1, cy + h/2))
    return nx1, ny1, nx2, ny2

def draw_alpha_box(img, pt1, pt2, color_bgr, alpha=0.18, thickness=2):
    overlay = img.copy()
    cv.rectangle(overlay, pt1, pt2, color_bgr, -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv.rectangle(img, pt1, pt2, color_bgr, thickness, cv.LINE_AA)

def predict_one(model: YOLO, img_path: Path, out_path: Path):
    # infer
    res = model.predict(
        source=str(img_path),
        imgsz=CFG["imgsz"],
        conf=CFG["conf"],
        iou=CFG["iou"],
        max_det=CFG["max_det"],
        save=False,
        verbose=False
    )[0]

    img = cv.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Nie wczytam obrazu: {img_path}")
    H, W = img.shape[:2]

    colors = {0:(0,200,0), 1:(230,160,0), 2:(0,140,255)}

    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)

        for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
            sx1, sy1, sx2, sy2 = shrink_box(x1, y1, x2, y2, CFG["shrink"], W, H)
            col = colors.get(k, (255,255,255))
            draw_alpha_box(img, (sx1,sy1), (sx2,sy2), col, alpha=0.18, thickness=CFG["thickness"])
            label = f"{CFG['class_names'][k]} {c:.2f}" if 0 <= k < len(CFG["class_names"]) else f"{k} {c:.2f}"
            (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, CFG["font_scale"], 2)
            tx = max(0, sx1); ty = max(th + 4, sy1)
            cv.rectangle(img, (tx, ty - th - 4), (tx + tw + 2, ty + 2), (0,0,0), -1)
            cv.putText(img, label, (tx + 1, ty - 2), cv.FONT_HERSHEY_SIMPLEX, CFG["font_scale"], (255,255,255), 2, cv.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(out_path), img):
        raise RuntimeError(f"Nie zapisano: {out_path}")

# Domyślne ścieżki
BASE_DIR = Path(__file__).parents[0]
DEFAULT_INPUTS_DIR = BASE_DIR / "_inputs"
DEFAULT_OUT_DIR    = BASE_DIR / "_outputs/_figury/step3_infer"
DEFAULT_WEIGHTS    = BASE_DIR / "_outputs/_figury/step2_training/runs/shapes_yolo_n/weights/last.pt"

def list_input_files(inputs_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in inputs_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

def parse_args():
    ap = argparse.ArgumentParser(description="Batch YOLO inference dla wszystkich plików bezpośrednio w katalogu.")
    ap.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR,
                    help="Katalog wejściowy (bez rekurencji).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Katalog wyjściowy.")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="Ścieżka do pliku wag last.pt.")
    return ap.parse_args()

def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Brak pliku wag: {args.weights}")
    if not args.inputs_dir.exists():
        raise FileNotFoundError(f"Brak katalogu wejściowego: {args.inputs_dir}")

    files = list_input_files(args.inputs_dir)
    if not files:
        print(f"Brak obrazów do przetworzenia w: {args.inputs_dir}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    ok = 0
    for img_path in tqdm(files, desc="YOLO infer", unit="img"):
        try:
            out_path = args.out_dir / f"{img_path.stem}_pred.jpg"
            predict_one(model, img_path, out_path)
            ok += 1
        except Exception as e:
            print(f"[ERR] {img_path.name}: {e}")

    print(f"\n[SUMMARY] OK: {ok}/{len(files)} | OUT: {args.out_dir}")

if __name__ == "__main__":
    main()
