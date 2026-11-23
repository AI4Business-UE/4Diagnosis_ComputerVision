from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Physical scale derived from openslide properties."""
    source_mpp_x: float        # µm/px at level-0
    source_mpp_y: float
    level: int                 # TIFF level used (e.g. 5)
    downsample: float          # level-0 / level ratio
    mpp_x: float               # µm/px in the TIFF
    mpp_y: float
    unit: str = "um_per_pixel"


@dataclass
class ScanInfo:
    """Image dimensions in two coordinate systems."""
    tiff_shape: list[int]           # [H, W] in TIFF pixels
    mrxs_level0_shape: list[int]    # [H, W] in level-0 pixels
    crop_offset_x: int = 0          # relative to WSI level_dimensions[level]
    crop_offset_y: int = 0          # relative to WSI level_dimensions[level]


@dataclass
class SliceItem:
    """One tissue slice (or grouped serial section) detected on the scan."""
    slice_id: int
    is_representative: bool
    bbox_tiff: list[int]        # [x, y, w, h] in TIFF pixels
    bbox_level0: list[int]      # [x, y, w, h] in level-0 pixels
    area_tiff_px: int


@dataclass
class SlicesInfo:
    """All slices grouped from a single scan plus grouping diagnostics."""
    representative_slice_id: int
    items: list[SliceItem] = field(default_factory=list)
    grouping_enabled: bool = True
    fallback_used: bool = False
    fallback_prob: float = 0.0
    user_ungrouped: bool = False
    auto_representative_slice_id: int | None = None
    auto_items: list[SliceItem] = field(default_factory=list)
    cleaning_level: int = 5

    def to_summary(self) -> dict:
        """Compact summary exposed to the frontend UI."""
        n_slices = len(self.items)
        is_grouped = (
            self.grouping_enabled
            and not self.user_ungrouped
            and not self.fallback_used
            and n_slices > 1
        )
        can_toggle_grouping = (
            is_grouped
            or self.user_ungrouped
            or len(self.auto_items) > 1
        )
        return {
            "n_slices": n_slices,
            "representative_slice_id": self.representative_slice_id,
            "grouping_enabled": self.grouping_enabled,
            "is_grouped": is_grouped,
            "fallback_used": self.fallback_used,
            "fallback_prob": round(float(self.fallback_prob), 4),
            "user_ungrouped": self.user_ungrouped,
            "can_toggle_grouping": can_toggle_grouping,
        }


@dataclass
class SlideMetadata:
    """
    Complete metadata for one slide job.

    Written to slides/<job_id>/<stem>.json at conversion time.
    Loaded by subsequent analysis steps and updated on user grouping toggle.
    """
    source: str             # "mrxs"
    source_path: str        # original filename e.g. "101110 Mallory.mrxs"
    calibration: CalibrationData
    scan: ScanInfo
    slices: SlicesInfo

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Metadata saved -> {path.name}")

    @classmethod
    def get_metadata_path(cls, job_dir: Path) -> Path:
        json_files = [
            f for f in job_dir.glob("*.json")
            if not f.name.startswith("glomeruli")
        ]
        if not json_files:
            raise FileNotFoundError(f"No slide metadata JSON found in {job_dir}")
        return json_files[0]

    @classmethod
    def load(cls, job_dir: Path) -> "SlideMetadata":
        """
        Load from the unified <stem>.json in job_dir.
        Backward-compatible with older JSON files missing extended slice fields.
        """
        meta_path = cls.get_metadata_path(job_dir)
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        slices_raw = data["slices"]

        return cls(
            source=data["source"],
            source_path=data["source_path"],
            calibration=CalibrationData(**data["calibration"]),
            scan=ScanInfo(**data["scan"]),
            slices=SlicesInfo(
                representative_slice_id=slices_raw["representative_slice_id"],
                items=[SliceItem(**item) for item in slices_raw.get("items", [])],
                grouping_enabled=slices_raw.get("grouping_enabled", True),
                fallback_used=slices_raw.get("fallback_used", False),
                fallback_prob=float(slices_raw.get("fallback_prob", 0.0)),
                user_ungrouped=slices_raw.get("user_ungrouped", False),
                auto_representative_slice_id=slices_raw.get("auto_representative_slice_id"),
                auto_items=[SliceItem(**item) for item in slices_raw.get("auto_items", [])],
                cleaning_level=int(slices_raw.get("cleaning_level", 5)),
            ),
        )

    def get_representative(self) -> SliceItem | None:
        """Return the representative SliceItem, or None if no slices."""
        return next(
            (s for s in self.slices.items
             if s.slice_id == self.slices.representative_slice_id),
            None,
        )

    def has_slices(self) -> bool:
        return bool(self.slices.items)
