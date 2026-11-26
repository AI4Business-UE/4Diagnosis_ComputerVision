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
from .source.processors.fibrosis_processor import FibrosisProcessor
from .source.processors.glomeruli_processor import GlomeruliProcessor
from .source.processors.mask import compute_min_component_area_px, generate_mask
from .source.processors.metadata import (
    CalibrationData,
    ScanInfo,
    SliceItem,
    SlicesInfo,
    SlideMetadata,
)
from .source.processors.slice_grouper import SliceGrouper
from .source.processors.tissue_length_processor import TissueLengthProcessor
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
        a_m = compute_min_component_area_px(0.0, 0.0, downsample=32.0)
        a_lvl6 = compute_min_component_area_px(0.0, 0.0, downsample=64.0)
        a_l = compute_min_component_area_px(0.0, 0.0, downsample=16.0)
        self.assertEqual(a_m, settings.SLICE_MIN_AREA)
        self.assertEqual(a_lvl6, settings.SLICE_MIN_AREA // 4)
        self.assertEqual(a_l, settings.SLICE_MIN_AREA * 4)

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

    def test_mask_preserves_thin_tubules_and_matches_across_M_L_XL(self):
        # Build a synthetic XL slide (~0.971 um/px) containing:
        # 1) tissue block (~400x400 um = ~160k um^2 > default 90k um^2) with thin walls
        # 2) small isolated debris (~24k um^2 < 90k um^2)
        src_mpp = 0.1213987
        mpp_xl = src_mpp * 8   # ~0.971 um/px
        mpp_l = src_mpp * 16   # ~1.942 um/px
        mpp_m = src_mpp * 32   # ~3.885 um/px

        img_xl = np.full((800, 800, 3), 252, dtype=np.uint8)
        img_xl[100:520, 100:520] = (150, 90, 180)
        for r in range(110, 500, 14):
            for c in range(110, 500, 14):
                img_xl[r:r + 8, c:c + 8] = (252, 252, 252)
        img_xl[620:780, 620:780] = (150, 90, 180)

        img_l = cv2.resize(img_xl, (400, 400), interpolation=cv2.INTER_AREA)
        img_m = cv2.resize(img_xl, (200, 200), interpolation=cv2.INTER_AREA)

        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "dummy.tiff")
            res_xl = generate_mask(
                p, mode="all", visualize=False, save_mask=False, save_preview=False,
                min_component_area=compute_min_component_area_px(mpp_xl, mpp_xl, 8.0, cleaning_level=5),
                img_bgr=img_xl, mpp_x=mpp_xl, mpp_y=mpp_xl,
            )
            res_l = generate_mask(
                p, mode="all", visualize=False, save_mask=False, save_preview=False,
                min_component_area=compute_min_component_area_px(mpp_l, mpp_l, 16.0, cleaning_level=5),
                img_bgr=img_l, mpp_x=mpp_l, mpp_y=mpp_l,
            )
            res_m = generate_mask(
                p, mode="all", visualize=False, save_mask=False, save_preview=False,
                min_component_area=compute_min_component_area_px(mpp_m, mpp_m, 32.0, cleaning_level=5),
                img_bgr=img_m, mpp_x=mpp_m, mpp_y=mpp_m,
            )

        # Identical component count (1 tissue region; debris removed on all three)
        self.assertEqual(len(res_m["components"]), 1)
        self.assertEqual(len(res_l["components"]), 1)
        self.assertEqual(len(res_xl["components"]), 1)

        # Debris area completely removed on XL and L
        self.assertEqual(int(res_xl["mask"][620:780, 620:780].sum()), 0)
        self.assertEqual(int(res_l["mask"][310:390, 310:390].sum()), 0)

        # Thin tubular walls inside main tissue are preserved on XL (>90% of true tissue pixels kept)
        true_tissue_xl = (img_xl[100:520, 100:520, 0] < 200)
        kept_ratio_xl = (res_xl["mask"][100:520, 100:520] & true_tissue_xl).sum() / true_tissue_xl.sum()
        self.assertGreater(kept_ratio_xl, 0.90)


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

    def test_path_traversal_and_invalid_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            slides_root = Path(tmp) / "slides"
            slides_root.mkdir()
            (Path(tmp) / "secret.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            with override_settings(SLIDES_DIR=slides_root):
                r_escape = self.client.get("/api/result-image/../secret.png/")
                r_bad_json = self.client.post(
                    "/api/fibrosis/",
                    data="{not-json",
                    content_type="application/json",
                )
            self.assertIn(r_escape.status_code, (400, 404))
            self.assertEqual(r_bad_json.status_code, 400)

    def test_large_tiff_preview_is_cached_while_small_tiff_stays_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            slides_root = Path(tmp)
            job_dir = slides_root / "job1"
            job_dir.mkdir()
            tiff_path = job_dir / "sample.tiff"
            cv2.imwrite(str(tiff_path), np.full((32, 32, 3), 120, dtype=np.uint8))

            with override_settings(SLIDES_DIR=slides_root):
                r_small = self.client.get("/api/result-image/job1/sample.tiff/?preview=1")
                try:
                    self.assertEqual(r_small["Content-Type"], "image/tiff")
                    self.assertFalse((job_dir / ".preview_cache").exists())
                finally:
                    r_small.close()

                with patch("myapp.views.PREVIEW_LARGE_TIFF_BYTES", 1):
                    r_large = self.client.get("/api/result-image/job1/sample.tiff/?preview=1")
                    try:
                        self.assertEqual(r_large["Content-Type"], "image/jpeg")
                        self.assertTrue((job_dir / ".preview_cache" / "sample.tiff.jpg").exists())
                    finally:
                        r_large.close()


class ProcessorsRobustnessTests(SimpleTestCase):
    def test_generate_mask_lightweight_components_match_slice_grouper_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "two_slices.tiff"
            img_bgr = np.full((300, 120, 3), 250, dtype=np.uint8)
            img_bgr[20:70, 30:90] = (140, 80, 190)
            img_bgr[220:270, 30:90] = (140, 80, 190)

            res = generate_mask(
                str(img_path),
                mode="all",
                visualize=False,
                save_mask=False,
                save_preview=False,
                min_component_area=500,
                img_bgr=img_bgr,
            )
            self.assertEqual(len(res["components"]), 2)
            self.assertEqual(res["components"][0]["crop_mask"].shape, (50, 60))

            grouper = SliceGrouper(use_fallback_ml=False)
            grouped_new = grouper.group(img_bgr, res["components"])

            full_masks = []
            for c in res["components"]:
                m = np.zeros(img_bgr.shape[:2], dtype=bool)
                x, y, w, h = c["bbox"]
                m[y:y + h, x:x + w] = c["crop_mask"]
                full_masks.append(m)
            grouped_legacy = grouper.group(img_bgr, full_masks)

            self.assertEqual(grouped_new["items"], grouped_legacy["items"])
            self.assertEqual(grouped_new["representative_slice_id"], grouped_legacy["representative_slice_id"])
    def test_fibrosis_raises_clear_error_when_mask_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "nomask.tiff"
            cv2.imwrite(str(img_path), np.full((40, 40, 3), 180, dtype=np.uint8))
            proc = FibrosisProcessor(str(img_path), threshold=0.4)
            with self.assertRaises(ValueError):
                proc.compute_fibrosis_ratio(str(img_path))

    def test_tissue_length_handles_closed_loop_skeleton(self):
        skeleton = np.zeros((50, 50), dtype=bool)
        skeleton[10:31, 10] = True
        skeleton[10:31, 30] = True
        skeleton[10, 10:31] = True
        skeleton[30, 10:31] = True

        proc = TissueLengthProcessor("dummy.tiff")
        _, paths, total_len = proc._find_longest_path_in_skeleton(skeleton)
        self.assertEqual(len(paths), 1)
        self.assertGreater(total_len, 0.0)

    def test_glomeruli_simple_global_merge_unions_overlaps(self):
        proc = GlomeruliProcessor("dummy.mrxs", "dummy.pt", 0, 0, 16.0)
        detections = [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "cls": 0, "cls_name": "circle", "conf": 0.6},
            {"x1": 120, "y1": 120, "x2": 220, "y2": 220, "cls": 0, "cls_name": "circle", "conf": 0.9},
            {"x1": 500, "y1": 500, "x2": 600, "y2": 600, "cls": 1, "cls_name": "rect", "conf": 0.8},
        ]
        merged = proc.simple_global_merge(detections, overlap_thresh=0.3)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(max(d["conf"] for d in merged), 0.9)

    def test_glomeruli_passes_bgr_and_preserves_edge_tile_scale(self):
        from PIL import Image as PilImage

        captured_tiles = []

        class DummyResult:
            boxes = []

        class DummyModel:
            def predict(self, batch, **kwargs):
                captured_tiles.extend(batch)
                return [DummyResult() for _ in batch]

        class DummySlide:
            level_dimensions = [(600, 400)]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get_thumbnail(self, size):
                # Medium-gray tissue so thumbnail pre-filter passes
                return PilImage.fromarray(np.full((40, 60, 3), 120, dtype=np.uint8), mode="RGB")

            def read_region(self, loc, level, size):
                w, h = size
                # Pure red-dominant tissue in RGB: R=210, G=40, B=30
                arr = np.zeros((h, w, 3), dtype=np.uint8)
                arr[:, :] = (210, 40, 30)
                return PilImage.fromarray(arr, mode="RGB")

        proc = GlomeruliProcessor(
            "dummy.mrxs",
            "dummy.pt",
            crop_offset_x=0,
            crop_offset_y=0,
            ds=1.0,
            tile_size=400,
            overlap=0,
            imgsz=100,
        )
        with (
            patch("myapp.source.processors.glomeruli_processor._load_yolo_model", return_value=DummyModel()),
            patch("openslide.OpenSlide", return_value=DummySlide()),
        ):
            proc.detect_glomeruli()

        # Two tiles: full 400x400 and edge 200x400 (padded right half with white -> 400x400)
        self.assertEqual(len(captured_tiles), 2)
        full_tile, edge_tile = captured_tiles
        self.assertEqual(full_tile.shape, (100, 100, 3))
        self.assertEqual(edge_tile.shape, (100, 100, 3))

        # Verify BGR channel order (B≈30, G≈40, R≈210) so Ultralytics flips BGR->RGB cleanly
        center_bgr = full_tile[50, 50]
        self.assertGreater(int(center_bgr[2]), 180)
        self.assertLess(int(center_bgr[0]), 60)

        # Edge tile left half has tissue (R>180), right half is white padding (255, 255, 255)
        self.assertGreater(int(edge_tile[50, 25, 2]), 180)
        self.assertLess(int(edge_tile[50, 25, 0]), 60)
        self.assertTrue(np.all(edge_tile[50, 75] >= 250))

    def test_processors_raise_exceptions_instead_of_negative_sentinels(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.tiff")
            with self.assertRaises(FileNotFoundError):
                FibrosisProcessor(missing).process_image()
            with self.assertRaises(FileNotFoundError):
                TissueLengthProcessor(missing).process_image()

            nomask = Path(tmp) / "nomask.tiff"
            cv2.imwrite(str(nomask), np.full((40, 40, 3), 180, dtype=np.uint8))
            with self.assertRaises(ValueError):
                FibrosisProcessor(str(nomask)).process_image()

    def test_glomeruli_stream_count_and_saved_json_share_coordinate_space(self):
        import json as _json
        from myapp.source.streaming_utils import generate_glomeruli_stream

        with tempfile.TemporaryDirectory() as tmp:
            slides_root = Path(tmp)
            job_dir = slides_root / "job-coord"
            job_dir.mkdir()
            mrxs_path = job_dir / "sample.mrxs"
            mrxs_path.write_bytes(b"fake")
            tiff_path = job_dir / "sample.tiff"
            cv2.imwrite(str(tiff_path), np.full((200, 100, 3), 200, dtype=np.uint8))
            cv2.imwrite(str(job_dir / "sample_mask.tiff"), np.full((200, 100), 255, dtype=np.uint8))

            meta = SlideMetadata(
                source="mrxs",
                source_path="sample.mrxs",
                calibration=CalibrationData(0.12, 0.12, 4, 16.0, 1.92, 1.92),
                scan=ScanInfo([200, 100], [10000, 8000], crop_offset_x=50, crop_offset_y=100),
                slices=SlicesInfo(
                    representative_slice_id=1,
                    items=[
                        SliceItem(0, False, [10, 10, 40, 40], [960, 1760, 640, 640], 1600),
                        SliceItem(1, True, [20, 100, 50, 60], [1120, 3200, 800, 960], 3000),
                    ],
                ),
            )
            meta.save(job_dir / "sample.json")

            raw_level0_det = [
                {"x1": 1220, "y1": 3350, "x2": 1420, "y2": 3550, "cls": 0, "cls_name": "circle", "conf": 0.88}
            ]
            expected_shifted = [
                {"x1": 100, "y1": 150, "x2": 300, "y2": 350, "cls": 0, "cls_name": "circle", "conf": 0.88}
            ]

            class DummySlide:
                level_dimensions = [(8000, 10000)]

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            def fake_detect(self_proc, on_batch=None, on_tile=None, cancel_event=None):
                self_proc.glomeruli = list(raw_level0_det)
                if on_batch:
                    on_batch(self_proc.glomeruli)
                return self_proc.glomeruli

            with (
                override_settings(SLIDES_DIR=slides_root),
                patch("openslide.OpenSlide", return_value=DummySlide()),
                patch("myapp.source.processors.glomeruli_processor.GlomeruliProcessor.detect_glomeruli", fake_detect),
            ):
                # 1. SSE stream + saved glomeruli.json
                events = [
                    _json.loads(chunk.removeprefix("data: ").strip())
                    for chunk in generate_glomeruli_stream("job-coord")
                    if chunk.startswith("data: ")
                ]
                stream_slide_info = next(e["slide_info"] for e in events if "slide_info" in e)
                stream_final = next(e["final_glomeruli"] for e in events if e.get("done"))
                saved_after_stream = _json.loads((job_dir / "glomeruli.json").read_text(encoding="utf-8"))

                # 2. Synchronous /api/glomeruli/count/ + saved glomeruli.json
                resp = self.client.post(
                    "/api/glomeruli/count/",
                    data=_json.dumps({"job_id": "job-coord"}),
                    content_type="application/json",
                )
                self.assertEqual(resp.status_code, 200)
                count_payload = resp.json()
                saved_after_count = _json.loads((job_dir / "glomeruli.json").read_text(encoding="utf-8"))

            self.assertEqual(stream_final, expected_shifted)
            self.assertEqual(saved_after_stream, expected_shifted)
            self.assertEqual(count_payload["glomeruli"], expected_shifted)
            self.assertEqual(saved_after_count, expected_shifted)
            self.assertEqual(count_payload["slide_info"], stream_slide_info)

    def test_glomeruli_best_openslide_level_and_edge_filtering(self):
        from PIL import Image as PilImage
        from myapp.source.processors.glomeruli_processor import _select_read_level

        class FakePyramidSlide:
            level_downsamples = (1.0, 2.0, 4.0, 8.0)

        # tile_size=4000, imgsz=1024 -> max_rel_ds=3.90625 -> picks level 1 (ds=2.0), not level 2 (4.0)
        lvl, rel_ds, base_ds = _select_read_level(FakePyramidSlide(), 0, 4000, 1024)
        self.assertEqual((lvl, rel_ds, base_ds), (1, 2.0, 1.0))

        # Verify edge-only small/elongated boxes are dropped at the end, while interior or full-size boxes stay
        proc = GlomeruliProcessor("dummy.mrxs", "dummy.pt", 0, 0, 16.0)
        raw_dets = [
            # 1. Tiny edge-only sliver (30um x 150um at 0.1214 um/px -> 247 x 1235 px) -> DROP
            {"x1": 0, "y1": 1000, "x2": 247, "y2": 2235, "cls": 1, "cls_name": "rect", "conf": 0.6},
            # 2. Full-sized edge-only glomerulus (170um x 170um -> 1400 x 1400 px) -> KEEP
            {"x1": 0, "y1": 5000, "x2": 1400, "y2": 6400, "cls": 0, "cls_name": "circle", "conf": 0.8},
            # 3. Small box on edge in tile 1 + overlapping interior detection in tile 2 -> merged not edge-only -> KEEP
            {"x1": 3800, "y1": 8000, "x2": 4000, "y2": 9200, "cls": 0, "cls_name": "circle", "conf": 0.5},
            {"x1": 3700, "y1": 7900, "x2": 5100, "y2": 9300, "cls": 0, "cls_name": "circle", "conf": 0.85},
            # 4. Small interior detection (not touching edge) -> KEEP
            {"x1": 2000, "y1": 2000, "x2": 2400, "y2": 2400, "cls": 1, "cls_name": "rect", "conf": 0.7},
        ]
        edge_flags = [True, True, True, False, False]
        merged, merged_edge_only = proc._merge_with_edge_flags(raw_dets, edge_flags, overlap_thresh=0.3)
        filtered = proc._filter_small_edge_boxes(merged, merged_edge_only, mpp_base=0.1214)

        self.assertEqual(len(filtered), 3)
        kept_confs = sorted(round(d["conf"], 2) for d in filtered)
        self.assertEqual(kept_confs, [0.7, 0.8, 0.85])

        # Verify read_region actually receives level 1 and halved read size on a pyramid slide
        read_calls = []

        class DummyResult:
            boxes = []

        class DummyModel:
            def predict(self, batch, **kwargs):
                return [DummyResult() for _ in batch]

        class PyramidDummySlide:
            level_dimensions = [(4000, 4000), (2000, 2000), (1000, 1000)]
            level_downsamples = (1.0, 2.0, 4.0)
            properties = {"openslide.mpp-x": "0.1214", "openslide.mpp-y": "0.1214"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get_thumbnail(self, size):
                return PilImage.fromarray(np.full((40, 40, 3), 120, dtype=np.uint8), mode="RGB")

            def read_region(self, loc, level, size):
                read_calls.append((loc, level, size))
                w, h = size
                return PilImage.fromarray(np.full((h, w, 3), 120, dtype=np.uint8), mode="RGB")

        proc_pyr = GlomeruliProcessor(
            "dummy.mrxs", "dummy.pt", 0, 0, 16.0, tile_size=4000, overlap=1500, imgsz=1024
        )
        with (
            patch("myapp.source.processors.glomeruli_processor._load_yolo_model", return_value=DummyModel()),
            patch("openslide.OpenSlide", return_value=PyramidDummySlide()),
        ):
            proc_pyr.detect_glomeruli()

        self.assertGreaterEqual(len(read_calls), 1)
        self.assertEqual(read_calls[0], ((0, 0), 1, (2000, 2000)))

    def test_mask_matches_M_calibrated_cleaning_on_L_and_XL(self):
        """High-res (L/XL) cleaning must match level-5 (M, ~3.885 um/px) calibration used by SliceGrouper."""
        with tempfile.TemporaryDirectory() as tmp:
            img_xl = np.full((480, 480, 3), 255, dtype=np.uint8)
            img_xl[60:420, 60:420] = (140, 80, 190)
            img_xl[120:320:4, 120:320] = (253, 210, 140)
            img_xl[121:320:4, 120:320] = (253, 210, 140)
            img_xl[5:55, 5:55] = (253, 225, 195)

            min_area_xl = compute_min_component_area_px(mpp_x=0.971, mpp_y=0.971, downsample=8.0, cleaning_level=5)
            res_xl = generate_mask(
                str(Path(tmp) / "xl.tiff"),
                mode="all",
                visualize=False,
                save_mask=False,
                save_preview=False,
                min_component_area=min_area_xl,
                img_bgr=img_xl,
                mpp_x=0.971,
                mpp_y=0.971,
            )

            img_m = cv2.resize(img_xl, (120, 120), interpolation=cv2.INTER_AREA)
            min_area_m = compute_min_component_area_px(mpp_x=3.885, mpp_y=3.885, downsample=32.0, cleaning_level=5)
            res_m = generate_mask(
                str(Path(tmp) / "m.tiff"),
                mode="all",
                visualize=False,
                save_mask=False,
                save_preview=False,
                min_component_area=min_area_m,
                img_bgr=img_m,
                mpp_x=3.885,
                mpp_y=3.885,
            )

            mask_xl = res_xl["mask"]
            self.assertEqual(len(res_xl["components"]), 1)
            self.assertEqual(len(res_m["components"]), 1)
            self.assertTrue(np.all(mask_xl[125:315, 125:315]))
            self.assertFalse(np.any(mask_xl[5:55, 5:55]))
            self.assertAlmostEqual(mask_xl.mean(), res_m["mask"].mean(), delta=0.02)

            # At cleaning_level=0 (open_ksize=0, min_component_area=1), even tiny tissue fragments stay
            img_tiny = np.full((120, 120, 3), 255, dtype=np.uint8)
            img_tiny[20:70, 20:70] = (140, 80, 190)
            img_tiny[95:97, 95:97] = (140, 80, 190)  # tiny 2x2 fragment (would be eroded by open_ksize=3)
            res_lvl0 = generate_mask(
                str(Path(tmp) / "lvl0.tiff"),
                mode="all",
                visualize=False,
                save_mask=False,
                save_preview=False,
                open_ksize=0,
                min_component_area=compute_min_component_area_px(3.885, 3.885, 32.0, cleaning_level=0),
                img_bgr=img_tiny,
                mpp_x=3.885,
                mpp_y=3.885,
            )
            self.assertEqual(len(res_lvl0["components"]), 2)
            self.assertTrue(np.all(res_lvl0["mask"][95:97, 95:97]))

            # Interior protection at cleaning_level=1 (open_ksize=3): fine blue collagen next to
            # 1-2 px white micro-lumens inside tissue must NOT be eroded, while detached specks are removed.
            img_int = np.full((160, 160, 3), 255, dtype=np.uint8)
            img_int[20:120, 20:120] = (140, 80, 190)  # main tissue core
            # Blue collagen region with fine 2x2 white micro-lumens spaced 2px apart inside core
            img_int[40:100, 40:100] = (230, 170, 60)  # strong blue collagen (V=230, S=189)
            for ly in range(44, 96, 4):
                for lx in range(44, 96, 4):
                    img_int[ly:ly + 2, lx:lx + 2] = (255, 255, 255)  # micro-lumen
            # Separate small debris speck outside (< 5000 um2 = 331 px at 3.885 mpp)
            img_int[135:145, 135:145] = (230, 170, 60)  # 100 px < 331 px -> should be removed
            blue_interior_expected = (img_int[40:100, 40:100, 0] == 230)

            res_lvl1 = generate_mask(
                str(Path(tmp) / "lvl1.tiff"),
                mode="all",
                visualize=False,
                save_mask=False,
                save_preview=False,
                open_ksize=3,
                min_component_area=compute_min_component_area_px(3.885, 3.885, 32.0, cleaning_level=1),
                img_bgr=img_int,
                mpp_x=3.885,
                mpp_y=3.885,
            )
            # Detached 10x10 speck removed, main core kept
            self.assertEqual(len(res_lvl1["components"]), 1)
            self.assertFalse(np.any(res_lvl1["mask"][135:145, 135:145]))
            # 100% of blue collagen pixels inside tissue preserved (zero interior erosion!)
            self.assertTrue(np.all(res_lvl1["mask"][40:100, 40:100][blue_interior_expected]))
