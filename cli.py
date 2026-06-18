#!/usr/bin/env python3

import argparse
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retouch import RetouchEngine, retouch as _retouch_fn

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

RECIPE_CHOICES = ["natural", "portrait", "beauty", "cosplay", "cosplay_3d", "cosplay_no_eq", "anime", "xiaohongshu", "dreamy", "magazine", "korean_beauty", "idol", "wedding", "anime_cosplay"]
PRESET_CHOICES = ["natural", "magazine", "beauty", "cosplay", "cosplay_3d", "cosplay_no_eq", "heavy", "dreamy", "anime", "portrait", "xiaohongshu", "korean_beauty", "idol", "wedding", "anime_cosplay"]


def _imread_exif(path):
    """Read image, applying EXIF orientation so pixel data is upright."""
    from PIL import ImageOps
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img) or pil_img
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _resize_for_processing(img_bgr, max_dim):
    """Downscale so longest side ≤ max_dim. Returns (resized, scale_factor)."""
    if max_dim is None:
        return img_bgr, 1.0
    h, w = img_bgr.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dim:
        return img_bgr, 1.0
    scale = max_dim / current_max
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _copy_exif(src_path, dst_path):
    try:
        from PIL.ExifTags import Base as ExifBase
        src_img = Image.open(src_path)
        exif = src_img.getexif()
        if exif:
            # Orientation has been baked into the pixel data — reset it to Normal
            if_base = None
            try:
                from PIL.ExifTags import Base
                if_base = Base
            except ImportError:
                if_base = ExifBase
            orientation_tag = getattr(if_base, 'Orientation', None)
            if orientation_tag is None:
                orientation_tag = 0x0112  # fallback
            exif[orientation_tag] = 1
            dst_img = Image.open(dst_path)
            dst_img.save(dst_path, exif=exif.tobytes())
    except Exception:
        pass


_worker_engine = None


def _init_worker():
    global _worker_engine
    _worker_engine = RetouchEngine()


def _process_single(args):
    img_path, output_dir, params, fmt, quality, force, copy_exif_flag, max_dim = args
    try:
        stem = img_path.stem
        out_path = (output_dir / f"{stem}.{fmt}") if output_dir else \
            img_path.with_suffix(f".{fmt}")

        if out_path.exists() and not force:
            return (img_path.name, "skipped")

        img_bgr = _imread_exif(img_path)
        orig_shape = img_bgr.shape[:2]
        img_bgr, _scale = _resize_for_processing(img_bgr, max_dim)

        global _worker_engine
        if _worker_engine is not None:
            engine = _worker_engine
            should_close = False
        else:
            engine = RetouchEngine()
            should_close = True

        try:
            result = engine.process(img_bgr, **params)
        finally:
            if should_close:
                engine.close()

        # Upscale back to original dimensions
        if _scale < 1.0:
            h2, w2 = result.shape[:2]
            result = cv2.resize(result, (orig_shape[1], orig_shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        write_params = []
        if fmt in ("jpg", "jpeg"):
            write_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif fmt == "webp":
            write_params = [cv2.IMWRITE_WEBP_QUALITY, quality]

        cv2.imwrite(str(out_path), result, write_params)

        if copy_exif_flag:
            _copy_exif(img_path, out_path)

        return (img_path.name, "done")
    except Exception as e:
        return (img_path.name, f"failed: {e}")


def _make_comparison(img_path, output_dir, fmt, quality, params):
    """Read the already-retouched output and stitch a side-by-side comparison."""
    try:
        retouched_path = output_dir / f"{img_path.stem}.{fmt}"
        compare_path = output_dir / f"{img_path.stem}_compare.{fmt}"

        original = _imread_exif(img_path)
        retouched = cv2.imread(str(retouched_path))
        if original is None or retouched is None:
            return

        if original.shape[:2] != retouched.shape[:2]:
            retouched = cv2.resize(retouched, (original.shape[1], original.shape[0]))

        h = original.shape[0]
        separator = np.full((h, 4, 3), 200, dtype=np.uint8)
        combined = np.hstack([original, separator, retouched])

        write_params = []
        if fmt in ("jpg", "jpeg"):
            write_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif fmt == "webp":
            write_params = [cv2.IMWRITE_WEBP_QUALITY, quality]

        cv2.imwrite(str(compare_path), combined, write_params)
    except Exception:
        pass


def find_images(input_path, recursive):
    path = Path(input_path)
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    files = []
    for f in path.glob(pattern):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(f)
    return sorted(files)


def build_params(args):
    params = {}
    if getattr(args, "recipe", None):
        params["recipe"] = args.recipe
    elif getattr(args, "preset", None):
        params["recipe"] = args.preset

    scalars = {
        "smooth": args.smooth,
        "whiten": args.whiten,
        "eye_enhance": args.eye_enhance,
        "dark_circles": args.dark_circles,
        "blemish": args.blemish,
        "lip_enhance": args.lip_enhance,
        "teeth_whiten": args.teeth_whiten,
        "equalize": args.equalize,
        "contrast": args.contrast,
        "grade_intensity": args.grade_intensity,
        "texture_opacity": args.texture_opacity,
        "mid_reduction": args.mid_reduction,
        "hair_enhance": args.hair_enhance,
        "dodge_burn": args.dodge_burn,
        "slimming": args.slimming,
        "blush": args.blush,
    }
    params.update({k: v for k, v in scalars.items() if v is not None})

    if args.color_grade:
        params["color_grade"] = args.color_grade
    if args.lip_tint:
        params["lip_tint"] = args.lip_tint
    if getattr(args, "lip_finish", None):
        params["lip_finish"] = args.lip_finish
    if getattr(args, "auto_exposure", False):
        params["auto_exposure"] = True

    return params


def main():
    parser = argparse.ArgumentParser(
        description="Professional batch face retouching tool"
    )
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-q", "--quality", type=int, default=95,
                        help="Output quality 1-100 (default: 95)")
    parser.add_argument("--format", choices=["jpg", "png", "webp", "same"],
                        default="same", help="Output format (default: same as input)")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Downscale so longest side ≤ N px before processing (faster)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Search subdirectories")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without processing")
    parser.add_argument("--auto-exposure", action="store_true",
                        help="Automatically normalize overexposed/underexposed photos")

    # Processing controls
    parser.add_argument("--recipe", choices=RECIPE_CHOICES,
                        help="Retouch recipe name")
    parser.add_argument("--preset", choices=PRESET_CHOICES,
                        help="Quick parameter preset (legacy alias)")
    parser.add_argument("--smooth", type=int, default=None,
                        help="Skin smoothing 0-100")
    parser.add_argument("--whiten", type=int, default=None,
                        help="Skin whitening 0-100")
    parser.add_argument("--eye-enhance", type=int, default=None,
                        help="Eye enhancement 0-100")
    parser.add_argument("--dark-circles", type=int, default=None,
                        help="Under-eye brightening 0-100")
    parser.add_argument("--blemish", type=int, default=None,
                        help="Blemish removal 0-100")
    parser.add_argument("--lip-enhance", type=int, default=None,
                        help="Lip enhancement 0-100")
    parser.add_argument("--lip-tint", type=str, default=None,
                        help="Lip tint preset name")
    parser.add_argument("--teeth-whiten", type=int, default=None,
                        help="Teeth whitening 0-100")
    parser.add_argument("--equalize", type=int, default=None,
                        help="Skin tone equalization 0-100")
    parser.add_argument("--contrast", type=int, default=None,
                        help="Contrast -50 to 50")
    parser.add_argument("--color-grade", type=str, default=None,
                        help="Colour grading preset")
    parser.add_argument("--grade-intensity", type=float, default=None,
                        help="Grading blend intensity 0-1")
    parser.add_argument("--hair-enhance", type=int, default=None,
                        help="Hair highlight enhancement 0-100")
    parser.add_argument("--dodge-burn", type=int, default=None,
                        help="Dodge & Burn facial sculpting 0-100")
    parser.add_argument("--slimming", type=int, default=None,
                        help="Face slimming 0-100")
    parser.add_argument("--blush", type=int, default=None,
                        help="Blush intensity 0-100")
    parser.add_argument("--lip-finish", choices=["matte", "gloss", "velvet"], default=None,
                        help="Lip finish type")

    # Advanced
    parser.add_argument("--texture-opacity", type=float, default=None,
                        help="Texture reprojection opacity 0-1")
    parser.add_argument("--mid-reduction", type=float, default=None,
                        help="Mid-frequency reduction 0-1")

    # Batch
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() // 2),
                        help="Parallel workers (default: CPU count / 2)")
    parser.add_argument("--no-compare", action="store_false", dest="compare",
                        help="Skip side-by-side comparison output")
    parser.add_argument("--no-exif", action="store_true",
                        help="Skip EXIF metadata copying")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"✖ Input not found: {input_path}")
        sys.exit(1)

    files = find_images(args.input, args.recursive)
    if not files:
        print("✖ No image files found")
        sys.exit(1)

    params = build_params(args)

    if args.dry_run:
        print(f"Dry run — {len(files)} image(s) found:\n")
        for f in files:
            print(f"  {f}")
        print(f"\nSettings: {params}")
        print(f"Workers: {args.workers}")
        return

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt == "same":
        fmt = "jpg"

    t0 = time.time()
    done = skipped = failed = 0

    if args.workers > 1 and len(files) > 1:
        pool_args = [
            (f, output_dir, params, fmt, args.quality, args.force, not args.no_exif, args.max_dim)
            for f in files
        ]
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
        ) as pool:
            futures = {pool.submit(_process_single, a): a[0] for a in pool_args}
            for future in tqdm(as_completed(futures), total=len(files),
                               desc="Retouching", unit="img"):
                name, status = future.result()
                if status == "done":
                    done += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    tqdm.write(f"  ✖ {name}: {status}")
    else:
        engine = RetouchEngine()
        try:
            for f in tqdm(files, desc="Retouching", unit="img"):
                img_bgr = _imread_exif(f)
                if img_bgr is None:
                    failed += 1
                    tqdm.write(f"  ✖ {f.name}: failed to read")
                    continue

                out_path = output_dir / f"{f.stem}.{fmt}" if output_dir else \
                    f.with_suffix(f".{fmt}")

                if out_path.exists() and not args.force:
                    skipped += 1
                    continue

                orig_shape = img_bgr.shape[:2]
                img_bgr, _scale = _resize_for_processing(img_bgr, args.max_dim)

                result = engine.process(img_bgr, **params)

                if _scale < 1.0:
                    result = cv2.resize(result, (orig_shape[1], orig_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
                write_params = []
                if fmt in ("jpg", "jpeg"):
                    write_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
                elif fmt == "webp":
                    write_params = [cv2.IMWRITE_WEBP_QUALITY, args.quality]

                cv2.imwrite(str(out_path), result, write_params)
                if not args.no_exif:
                    _copy_exif(f, out_path)
                done += 1
        finally:
            engine.close()

    if args.compare:
        tqdm.write("Generating comparisons...")
        for f in tqdm(files, desc="Comparison", unit="img"):
            _make_comparison(f, output_dir or f.parent, fmt, args.quality, params)

    elapsed = time.time() - t0
    print(f"\nDone — {done} processed, {skipped} skipped, {failed} failed"
          f"  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
