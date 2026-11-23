import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from django.conf import settings
from django.test import SimpleTestCase, override_settings

from .source.processed_image import ProcessedImage
from .source.processors.mask import compute_min_component_area_px
from .source.processors.metadata import (
    CalibrationData,
    ScanInfo,
    SliceItem,
    SlicesInfo,
    SlideMetadata,
)
from .source.processors.slice_grouper import SliceGrouper
from .source.slide_converter import SlideConverter


class MaskScalingTests(SimpleTestCase):
    def test_physical_area_scales_quadratically_across_presets(self):
        src_mpp = 0.1213987
        a_s = compute_min_component_area_px(src_mpp * 128, src_mpp * 128, downsample=128.0)
        a_m = compute_min_component_area_px(src_mpp * 32, src_mpp * 32, downsample=32.0)
        a_l = compute_min_component_area_px(src_mpp * 16, src_mpp * 16, downsample=16.0)
        a_xl = compute_min_component_area_px(src_mpp * 8, src_mpp * 8, downsample=8.0)

        self.assertAlmostEqual(a_m, 6000, delta=150)
        self.assertAlmostEqual(a_xl / a_l, 4.0, delta=0.05)
        self.assertAlmostEqual(a_l / a_m, 4.0, delta=0.05)
        self.assertAlmostEqual(a_m / a_s, 16.0, delta=0.1)

    def test_downsample_fallback_without_mpp(self):
        a_l = compute_min_component_area_px(0.0, 0.0, downsample=32.0)
        a_m = compute_min_component_area_px(0.0, 0.0, downsample=64.0)
        a_xl = compute_min_component_area_px(0.0, 0.0, downsample=16.0)
        self.assertEqual(a_l, settings.SLICE_MIN_AREA)
        self.assertEqual(a_m, settings.SLICE_MIN_AREA // 4)
        self.assertEqual(a_xl, settings.SLICE_MIN_AREA * 4)

    def test_cleaning_level_is_invariant_across_conversion_presets(self):
        src_mpp = 0.1213987
        for lvl in (1, 5, 8, 10):
            px_s = compute_min_component_area_px(src_mpp * 128, src_mpp * 128, downsample=128.0, cleaning_level=lvl)
            px_l = compute_min_component_area_px(src_mpp * 16, src_mpp * 16, downsample=16.0, cleaning_level=lvl)
            um2_s = px_s * (src_mpp * 128) ** 2
            um2_l = px_l * (src_mpp * 16) ** 2
            self.assertAlmostEqual(um2_s / um2_l, 1.0, delta=0.02)

        self.assertEqual(
            compute_min_component_area_px(src_mpp * 16, src_mpp * 16, downsample=16.0, cleaning_level=0),
            1,
        )


class SliceGrouperParityTests(SimpleTestCase):
    def test_feature_vector_matches_research_pipeline(self):
        research_root = Path(settings.BASE_DIR).parent.parent / "SliceGrouperResearch"
        if not (research_root / "src" / "method_benchmarker.py").exists():
            self.skipTest("SliceGrouperResearch directory not present")

        h, w = 800, 400
        img_bgr = np.full((h, w, 3), 245, dtype=np.uint8)
        rng = np.random.RandomState(42)

        patch1 = rng.randint(60, 210, size=(120, 140, 3), dtype=np.uint8)
        patch2 = np.clip(patch1.astype(np.int32) + rng.randint(-8, 9, size=patch1.shape), 0, 255).astype(np.uint8)

        img_bgr[60:180, 120:260] = patch1
        img_bgr[580:700, 120:260] = patch2

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[60:180, 120:260] = 255
        mask[580:700, 120:260] = 255

        comp0 = np.zeros((h, w), dtype=bool)
        comp0[60:180, 120:260] = True
        comp1 = np.zeros((h, w), dtype=bool)
        comp1[580:700, 120:260] = True

        grouper = SliceGrouper()
        prod_res = grouper.group(img_bgr, [comp0, comp1])

        sys.path.insert(0, str(research_root))
        try:
            from src.io import detect_components
            from src.method_benchmarker import HybridSafetySpatialMethod

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            r_comps = detect_components(mask, image=img_rgb, min_area_px=1000)
            method = HybridSafetySpatialMethod(fallback_model_path=settings.FALLBACK_MODEL_PATH)
            pred_s, _, _, details = method.predict(img_rgb, r_comps)
        finally:
            if str(research_root) in sys.path:
                sys.path.remove(str(research_root))

        self.assertEqual(prod_res["fallback_used"], bool(details.get("is_fallback", False)))
        self.assertEqual(len(prod_res["items"]), len(np.unique(pred_s)))
        if "fallback_prob" in details:
            self.assertAlmostEqual(prod_res["fallback_prob"], float(details["fallback_prob"]), places=5)


class SlideGroupingToggleAndViewsTests(SimpleTestCase):
    def test_set_grouping_ungroup_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            slides_root = Path(tmp)
            job_id = "test-job-123"
            job_dir = slides_root / job_id
            job_dir.mkdir()

            img = np.full((200, 100, 3), 240, dtype=np.uint8)
            img[10:60, 20:80] = (120, 80, 180)
            img[130:180, 20:80] = (120, 80, 180)
            mask = np.zeros((200, 100), dtype=np.uint8)
            mask[10:60, 20:80] = 255
            mask[130:180, 20:80] = 255

            cv2.imwrite(str(job_dir / "sample.tiff"), img)
            cv2.imwrite(str(job_dir / "sample_mask.tiff"), mask)
            cleaned = img.copy()
            cleaned[mask == 0] = (255, 255, 255)
            cv2.imwrite(str(job_dir / "sample_cleaned.tiff"), cleaned)
            cv2.imwrite(str(job_dir / "sample_origin_detect.tiff"), cleaned[130:180, 20:80])
            (job_dir / "sample_repr_crop_fibrosis.tiff").write_bytes(b"stale")
            (job_dir / "glomeruli.json").write_text("[]", encoding="utf-8")

            auto_items = [
                SliceItem(0, False, [20, 10, 60, 50], [640, 320, 1920, 1600], 3000),
                SliceItem(1, True, [20, 130, 60, 50], [640, 4160, 1920, 1600], 3000),
            ]
            meta = SlideMetadata(
                source="mrxs",
                source_path="sample.mrxs",
                calibration=CalibrationData(0.12, 0.12, 5, 32.0, 3.84, 3.84),
                scan=ScanInfo([200, 100], [6400, 3200], 0, 0),
                slices=SlicesInfo(
                    representative_slice_id=1,
                    items=list(auto_items),
                    grouping_enabled=True,
                    fallback_used=False,
                    fallback_prob=0.1,
                    user_ungrouped=False,
                    auto_representative_slice_id=1,
                    auto_items=list(auto_items),
                ),
            )
            meta.save(job_dir / "sample.json")

            with override_settings(SLIDES_DIR=slides_root):
                resp = self.client.post(
                    f"/api/slides/{job_id}/grouping/",
                    data=json.dumps({"grouped": False}),
                    content_type="application/json",
                )
            self.assertEqual(resp.status_code, 200)
            payload = resp.json()
            self.assertFalse(payload["slices_summary"]["is_grouped"])
            self.assertTrue(payload["slices_summary"]["user_ungrouped"])
            self.assertEqual(payload["slices_summary"]["n_slices"], 1)
            self.assertFalse((job_dir / "sample_repr_crop_fibrosis.tiff").exists())
            self.assertFalse((job_dir / "glomeruli.json").exists())

            origin_after_ungroup = cv2.imread(str(job_dir / "sample_origin_detect.tiff"))
            self.assertEqual(origin_after_ungroup.shape[:2], (200, 100))

            with override_settings(SLIDES_DIR=slides_root):
                resp2 = self.client.post(
                    f"/api/slides/{job_id}/grouping/",
                    data=json.dumps({"grouped": True}),
                    content_type="application/json",
                )
            self.assertEqual(resp2.status_code, 200)
            payload2 = resp2.json()
            self.assertTrue(payload2["slices_summary"]["is_grouped"])
            self.assertFalse(payload2["slices_summary"]["user_ungrouped"])
            self.assertEqual(payload2["slices_summary"]["n_slices"], 2)

            origin_after_restore = cv2.imread(str(job_dir / "sample_origin_detect.tiff"))
            self.assertEqual(origin_after_restore.shape[:2], (50, 60))

    def test_get_result_image_content_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            slides_root = Path(tmp)
            job_dir = slides_root / "job1"
            job_dir.mkdir()
            (job_dir / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (job_dir / "b.tiff").write_bytes(b"II*\x00")
            (job_dir / "c.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            with override_settings(SLIDES_DIR=slides_root):
                r_jpg = self.client.get("/api/result-image/job1/a.jpg/")
                r_tif = self.client.get("/api/result-image/job1/b.tiff/")
                r_png = self.client.get("/api/result-image/job1/c.png/")

            try:
                self.assertEqual(r_jpg["Content-Type"], "image/jpeg")
                self.assertEqual(r_tif["Content-Type"], "image/tiff")
                self.assertEqual(r_png["Content-Type"], "image/png")
            finally:
                r_jpg.close()
                r_tif.close()
                r_png.close()

    def test_calculate_tissue_length_handles_missing_mask_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            tiff_path = job_dir / "sample.tiff"
            img = np.full((100, 100, 3), 240, dtype=np.uint8)
            cv2.imwrite(str(tiff_path), img)
            meta = SlideMetadata(
                source="mrxs",
                source_path="sample.mrxs",
                calibration=CalibrationData(0.12, 0.12, 5, 32.0, 3.84, 3.84),
                scan=ScanInfo([100, 100], [3200, 3200], 0, 0),
                slices=SlicesInfo(
                    representative_slice_id=0,
                    items=[SliceItem(0, True, [0, 0, 100, 100], [0, 0, 3200, 3200], 10000)],
                ),
            )
            meta.save(job_dir / "sample.json")

            pi = ProcessedImage(str(tiff_path))
            with patch("myapp.source.processed_image.TissueLengthProcessor.process_image", return_value={"length": 1.23, "image_path": "", "error": None}):
                res = pi.calculate_tissue_length()

            self.assertEqual(res["length"], 1.23)
            self.assertFalse((job_dir / "sample_length_crop_mask.tiff").exists())
