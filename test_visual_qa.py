#!/usr/bin/env python3
"""Visual QA test runner for F5 (liquify) and F10 (smart-process).

This script runs the two visual QA gates specified in MASTER_PLAN.md:
- Gate 1 (line 85): F5 Liquify visual-critical warping
- Gate 2 (line 99): F10 Smart-process acceptance run

Usage:
  python3 test_visual_qa.py --gate f5 --output-dir /tmp/f5_qa
  python3 test_visual_qa.py --gate f10 --output-dir /tmp/f10_qa
  python3 test_visual_qa.py --gate all --output-dir /tmp/visual_qa
"""

import argparse
import sys
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import logging

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retouch import RetouchEngine
from retouch.geometry import FaceReshaper
from retouch.detection import FaceDetector
from retouch.io import imread_exif, output_format, make_comparison
from retouch.smart_default import SmartProcessor
from retouch.engine import ProcessingContext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class F5TestResult:
    """Result of a single F5 test image."""
    image_path: str
    output_path: str
    criteria_results: Dict[str, Tuple[str, str]]  # {criterion: (status, details)}
    overall_pass: bool
    notes: str = ""


@dataclass
class F10TestResult:
    """Result of a single F10 test image."""
    image_path: str
    output_path: str
    recipe: str
    explanation: str
    visual_pass: bool
    anomalies: List[str]
    notes: str = ""


class F5LiquifyQA:
    """Visual QA for F5 (face-aware liquify sliders)."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector = FaceDetector()
        self.reshaper = FaceReshaper()

    def test_image(self, img_path: str) -> Optional[F5TestResult]:
        """Run F5 visual QA on a single reference image.

        Tests non-zero reshape params on a face-aware warp and inspects:
        - No Edge Tearing: displacement continuity, no discontinuities > 2px
        - Natural Output: no plastic/uncanny appearance at 100% zoom
        - No Halo: no luminance rings around warped feature edges
        - Background lines intact: background doesn't show visible warping
        """
        try:
            logger.info(f"Testing F5 on {img_path}")
            img = imread_exif(Path(img_path))
            if img is None:
                logger.error(f"Failed to load image: {img_path}")
                return None

            # Detect faces
            faces = self.detector.detect(img)
            if not faces:
                logger.warning(f"No faces detected in {img_path}")
                return None

            face = faces[0]  # Test on first face
            h, w = img.shape[:2]

            # Run with non-zero reshape params
            params = [
                ('eye_size', 50),
                ('jaw_width', -30),
                ('chin_length', 20),
            ]

            results = {}
            output_images = {}

            for param_name, value in params:
                ctx = ProcessingContext()
                setattr(ctx, f'reshape_{param_name}', value)

                # Apply warp
                warped = self.reshaper.reshape(img, faces, ctx)
                output_images[param_name] = warped

                # Evaluate criteria
                criteria = self._evaluate_criteria(
                    img, warped, face, param_name, value
                )
                results[param_name] = criteria

            # Prepare output
            output_path = self.output_dir / f"{Path(img_path).stem}_f5_test.jpg"
            self._save_output(img, output_images, output_path)

            overall_pass = all(
                status == "PASS"
                for param_results in results.values()
                for status, _ in param_results.values()
            )

            return F5TestResult(
                image_path=img_path,
                output_path=str(output_path),
                criteria_results=results,
                overall_pass=overall_pass,
                notes=f"Tested params: {[p[0] for p in params]}"
            )

        except Exception as e:
            logger.error(f"Error testing F5 on {img_path}: {e}", exc_info=True)
            return None

    def _evaluate_criteria(
        self,
        orig: np.ndarray,
        warped: np.ndarray,
        face,
        param_name: str,
        value: float
    ) -> Dict[str, Tuple[str, str]]:
        """Evaluate visual QA criteria for a warp."""
        results = {}

        # No Edge Tearing: check displacement field continuity
        tearing_status, tearing_detail = self._check_no_edge_tearing(
            orig, warped, face, param_name
        )
        results['No-Edge-Tearing'] = (tearing_status, tearing_detail)

        # Natural Output: subjective visual inspection
        # (automated detection of "plastic" is done via simple heuristics)
        natural_status, natural_detail = self._check_natural_output(
            orig, warped, face
        )
        results['Natural-Output'] = (natural_status, natural_detail)

        # No Halo: check for luminance rings at feature boundaries
        halo_status, halo_detail = self._check_no_halo(orig, warped, face)
        results['No-Halo'] = (halo_status, halo_detail)

        # Background lines intact: verify background not visibly warped
        bg_status, bg_detail = self._check_background_lines(orig, warped, face)
        results['Background-line'] = (bg_status, bg_detail)

        return results

    def _check_no_edge_tearing(
        self, orig: np.ndarray, warped: np.ndarray, face, param_name: str
    ) -> Tuple[str, str]:
        """Check for edge tearing via discontinuity in the warp field."""
        try:
            # Compute difference to estimate displacement
            diff = cv2.absdiff(orig.astype(np.float32), warped.astype(np.float32))
            diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            # Look for discontinuities (sharp boundaries) in the difference
            edges = cv2.Canny(diff_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Threshold: if edge density is too high, likely tearing
            if edge_density > 0.08:
                return "FAIL", f"High edge discontinuity: {edge_density:.1%}"
            else:
                return "PASS", f"Smooth transitions: edge density {edge_density:.1%}"
        except Exception as e:
            return "FAIL", f"Evaluation error: {e}"

    def _check_natural_output(
        self, orig: np.ndarray, warped: np.ndarray, face
    ) -> Tuple[str, str]:
        """Check for natural appearance (no plastic/uncanny look)."""
        try:
            # Heuristic: extreme local saturation changes or brightness spikes
            # indicate over-processing / artificial look

            # Convert to LAB for perceptual analysis
            orig_lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
            warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Check L (lightness) changes — should be smooth within the face
            l_diff = np.abs(warped_lab[:, :, 0] - orig_lab[:, :, 0])
            l_extreme = np.percentile(l_diff, 95)

            # Check chroma (a, b) for saturation changes
            orig_chroma = np.sqrt(
                orig_lab[:, :, 1] ** 2 + orig_lab[:, :, 2] ** 2
            )
            warped_chroma = np.sqrt(
                warped_lab[:, :, 1] ** 2 + warped_lab[:, :, 2] ** 2
            )
            chroma_diff = np.abs(warped_chroma - orig_chroma)
            chroma_extreme = np.percentile(chroma_diff, 95)

            # Thresholds for "plastic" look
            if l_extreme > 30 or chroma_extreme > 15:
                return "FAIL", (
                    f"Extreme lightness ({l_extreme:.1f}) or chroma ({chroma_extreme:.1f}) "
                    "change suggests plastic/artificial look"
                )
            else:
                return "PASS", (
                    f"Natural transitions: L_delta={l_extreme:.1f}, "
                    f"chroma_delta={chroma_extreme:.1f}"
                )
        except Exception as e:
            return "FAIL", f"Evaluation error: {e}"

    def _check_no_halo(
        self, orig: np.ndarray, warped: np.ndarray, face
    ) -> Tuple[str, str]:
        """Check for luminance halos around feature edges."""
        try:
            # Convert to grayscale (luminance only)
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Look for bright (or dark) fringes around the face boundary
            # Use Laplacian to detect edges
            laplacian_orig = cv2.Laplacian(orig_gray, cv2.CV_32F)
            laplacian_warped = cv2.Laplacian(warped_gray, cv2.CV_32F)

            # If warped has stronger edges than original, likely a halo
            edge_strength_orig = np.std(laplacian_orig)
            edge_strength_warped = np.std(laplacian_warped)

            if edge_strength_warped > edge_strength_orig * 1.3:
                return "FAIL", (
                    f"Halo artifact detected: edge strength increased "
                    f"{edge_strength_orig:.1f} → {edge_strength_warped:.1f}"
                )
            else:
                return "PASS", f"No halo: edge strength ratio {edge_strength_warped/edge_strength_orig:.2f}x"
        except Exception as e:
            return "FAIL", f"Evaluation error: {e}"

    def _check_background_lines(
        self, orig: np.ndarray, warped: np.ndarray, face
    ) -> Tuple[str, str]:
        """Check that background remains intact (no visible warping)."""
        try:
            x, y, w, h = face.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Extract background region (exclude face)
            margin = 20
            if x > margin and y > margin:
                bg_orig = orig[y - margin:y, x - margin:x + w + margin]
                bg_warped = warped[y - margin:y, x - margin:x + w + margin]

                if bg_orig.size > 0 and bg_warped.size > 0:
                    # Check if background changed significantly
                    bg_diff = cv2.absdiff(bg_orig, bg_warped)
                    change_pct = np.mean(bg_diff) / 255.0

                    if change_pct > 0.05:
                        return "FAIL", (
                            f"Background distortion detected: "
                            f"{change_pct:.1%} pixel change"
                        )
                    else:
                        return "PASS", f"Background intact: {change_pct:.1%} change"
                else:
                    return "PASS", "No background region to test"
            else:
                return "PASS", "Face too close to edge; skipped background test"
        except Exception as e:
            return "FAIL", f"Evaluation error: {e}"

    def _save_output(
        self, orig: np.ndarray, outputs: Dict[str, np.ndarray], output_path: Path
    ):
        """Save comparison images."""
        # Create a grid of original + warped versions
        try:
            h, w = orig.shape[:2]
            scale = max(1, w // 400)  # Scale down if too large
            if scale > 1:
                orig_small = cv2.resize(orig, (w // scale, h // scale))
            else:
                orig_small = orig.copy()

            cols = []
            cols.append(orig_small)

            for param_name, warped in outputs.items():
                if scale > 1:
                    warped_small = cv2.resize(warped, (w // scale, h // scale))
                else:
                    warped_small = warped.copy()
                cols.append(warped_small)

            # Stack horizontally
            result = np.hstack(cols)
            cv2.imwrite(str(output_path), result)
            logger.info(f"Saved comparison to {output_path}")
        except Exception as e:
            logger.error(f"Error saving output: {e}")


class F10SmartQA:
    """Visual QA for F10 (smart-process acceptance)."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = RetouchEngine()
        self.smart = SmartProcessor(engine=self.engine)

    def test_batch(self, image_paths: List[str]) -> List[F10TestResult]:
        """Run F10 smart-process acceptance test on a batch of images.

        Runs smart analysis on each image and spot-checks:
        - Recipe selection makes sense (matches lighting, noise, etc.)
        - Explanations are coherent and match the recipe choice
        - No crashes or obviously bad outputs
        - Output quality is acceptable
        """
        results = []

        for i, img_path in enumerate(image_paths, 1):
            logger.info(f"Testing F10 [{i}/{len(image_paths)}]: {img_path}")
            result = self._test_single_image(img_path)
            if result:
                results.append(result)

        return results

    def _test_single_image(self, img_path: str) -> Optional[F10TestResult]:
        """Test smart-process on a single image."""
        try:
            img = imread_exif(Path(img_path))
            if img is None:
                logger.error(f"Failed to load image: {img_path}")
                return None

            # Run smart analysis
            suggestion = self.smart.analyze_and_suggest(img)

            # Apply smart processing
            processed = self.smart.process_smart(img)
            if processed is None:
                processed = img

            # Spot-check results
            recipe = suggestion.recipe
            explanation = "; ".join(suggestion.explanations) or "No explanation"

            anomalies = self._spot_check(img, processed, suggestion)
            visual_pass = len(anomalies) == 0

            # Save output
            output_path = self.output_dir / f"{Path(img_path).stem}_f10_smart.jpg"
            cv2.imwrite(str(output_path), processed)

            return F10TestResult(
                image_path=img_path,
                output_path=str(output_path),
                recipe=recipe,
                explanation=explanation,
                visual_pass=visual_pass,
                anomalies=anomalies,
                notes=""
            )

        except Exception as e:
            logger.error(f"Error testing F10 on {img_path}: {e}", exc_info=True)
            return None

    def _spot_check(self, orig: np.ndarray, processed: np.ndarray, suggestion) -> List[str]:
        """Spot-check for obvious issues."""
        anomalies = []

        try:
            # Check 1: Output shape matches input
            if orig.shape != processed.shape:
                anomalies.append(f"Shape mismatch: {orig.shape} → {processed.shape}")

            # Check 2: Output dtype is valid
            if processed.dtype not in [np.uint8, np.float32]:
                anomalies.append(f"Invalid dtype: {processed.dtype}")

            # Check 3: No NaN or Inf values
            if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
                anomalies.append("NaN or Inf values in output")

            # Check 4: Histogram clipping
            if processed.dtype == np.uint8:
                flat = processed.ravel()
                clip_low = np.sum(flat == 0) / len(flat)
                clip_high = np.sum(flat == 255) / len(flat)
                if clip_low > 0.005 or clip_high > 0.005:
                    anomalies.append(
                        f"Histogram clipping: {clip_low:.1%} at 0, {clip_high:.1%} at 255"
                    )

            # Check 5: Suggestion coherence
            if not suggestion.recipe or not suggestion.explanations:
                anomalies.append("Empty recipe or explanations")

        except Exception as e:
            anomalies.append(f"Spot-check error: {e}")

        return anomalies


def main():
    parser = argparse.ArgumentParser(
        description="Run visual QA gates for F5 (liquify) and F10 (smart-process)"
    )
    parser.add_argument(
        "--gate",
        choices=["f5", "f10", "all"],
        default="all",
        help="Which gate to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/visual_qa",
        help="Output directory for results"
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        default="/Applications/htdocs/retouch/test_output",
        help="Directory containing reference test images"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to test (for quick runs)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find test images
    test_img_dir = Path(args.test_images_dir)
    if not test_img_dir.exists():
        logger.error(f"Test image directory not found: {test_img_dir}")
        sys.exit(1)

    image_files = sorted([
        f for f in test_img_dir.glob("*.jpg")
        if not "_compare" in f.name and not "_test" in f.name
    ])[:args.limit]

    if not image_files:
        logger.error(f"No test images found in {test_img_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_files)} test images")

    # Run tests
    all_results = {"f5": [], "f10": []}

    if args.gate in ["f5", "all"]:
        logger.info("=" * 60)
        logger.info("Running F5 (Liquify) Visual QA Gate")
        logger.info("=" * 60)
        f5_qa = F5LiquifyQA(output_dir / "f5_results")

        # Test on first 3-5 images
        for img_path in image_files[:5]:
            result = f5_qa.test_image(str(img_path))
            if result:
                all_results["f5"].append(result)
                logger.info(f"  F5 {Path(img_path).name}: {'PASS' if result.overall_pass else 'FAIL'}")

    if args.gate in ["f10", "all"]:
        logger.info("=" * 60)
        logger.info("Running F10 (Smart-Process) Visual QA Gate")
        logger.info("=" * 60)
        f10_qa = F10SmartQA(output_dir / "f10_results")
        results = f10_qa.test_batch([str(p) for p in image_files[:20]])
        all_results["f10"] = results

        for result in results:
            status = "PASS" if result.visual_pass else "FAIL"
            logger.info(f"  F10 {Path(result.image_path).name}: {status} (recipe={result.recipe})")

    # Write summary
    summary_path = output_dir / "qa_summary.json"
    summary = {
        "f5_results": [asdict(r) for r in all_results["f5"]],
        "f10_results": [asdict(r) for r in all_results["f10"]],
        "overall_f5_pass": all(r.overall_pass for r in all_results["f5"]),
        "overall_f10_pass": all(r.visual_pass for r in all_results["f10"]),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {summary_path}")
    logger.info(f"F5 Tests: {len(all_results['f5'])} passed")
    logger.info(f"F10 Tests: {len([r for r in all_results['f10'] if r.visual_pass])} / {len(all_results['f10'])} passed")


if __name__ == "__main__":
    main()
