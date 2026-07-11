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


@dataclass
class SliceItem:
    """One tissue slice (connected component) detected on the scan."""
    slice_id: int
    is_representative: bool
    bbox_tiff: list[int]        # [x, y, w, h] in TIFF pixels
    bbox_level0: list[int]      # [x, y, w, h] in level-0 pixels
    area_tiff_px: int


@dataclass
class SlicesInfo:
    """All slices grouped from a single scan."""
    representative_slice_id: int
    items: list[SliceItem] = field(default_factory=list)


@dataclass
class SlideMetadata:
    """
    Complete metadata for one slide job.

    Written to  slides/<job_id>/<stem>.json  at conversion time.
    Loaded read-only by every subsequent analysis step.
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
        logger.info(f"Metadata saved → {path.name}")

    @classmethod
    def load(cls, job_dir: Path) -> "SlideMetadata":
        """
        Load from the unified <stem>.json in job_dir.
        Raises FileNotFoundError if not found (no backward compat).
        """
        json_files = [
            f for f in job_dir.glob("*.json")
            if not f.name.startswith("glomeruli")
        ]
        if not json_files:
            raise FileNotFoundError(f"No slide metadata JSON found in {job_dir}")

        data = json.loads(json_files[0].read_text(encoding="utf-8"))

        return cls(
            source=data["source"],
            source_path=data["source_path"],
            calibration=CalibrationData(**data["calibration"]),
            scan=ScanInfo(**data["scan"]),
            slices=SlicesInfo(
                representative_slice_id=data["slices"]["representative_slice_id"],
                items=[SliceItem(**item) for item in data["slices"]["items"]],
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
