"""
src/deep_features.py — Deep Learning Feature Extraction
========================================================

Modern deep neural networks trained on large image datasets learn rich
visual representations that go far beyond hand-crafted features.
For histopathology tissue matching, deep embeddings capture:

  • High-level tissue morphology (gland architecture, stroma density)
  • Staining patterns (hue, saturation, texture gradients)
  • Cell arrangement and density
  • Pathological changes (fibrosis, necrosis, infiltration)

This module provides three levels of embedding sophistication:

1. **ResNet50** (He et al., 2016)
   Trained on ImageNet.  The penultimate fully connected layer (before the
   classification head) produces 2048-dimensional embeddings.  These work
   surprisingly well for histopathology despite domain mismatch.

2. **DINOv2** (Oquab et al., 2023)
   Self-supervised Vision Transformer trained with DINO on large curated
   datasets.  The CLS token provides 768-d embeddings with excellent
   transfer properties to medical imaging.

3. **Pathology Foundation Models** (optional, HuggingFace)
   UNI, CONCH, or PLIP — models specifically trained on millions of
   histopathology images.  These produce state-of-the-art embeddings for
   tissue analysis tasks.  Loaded with graceful fallback if unavailable.

All extractors follow a common interface and are designed to be batched
for efficiency.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

_logger = logging.getLogger("SliceGrouper.deep")


# ---------------------------------------------------------------------------
# Device selection helper
# ---------------------------------------------------------------------------


def _resolve_device(device: str = "auto") -> "torch.device":  # type: ignore
    """
    Resolve the compute device string to a PyTorch device.

    Parameters
    ----------
    device : str
        'auto', 'cpu', 'cuda', or 'mps'.

    Returns
    -------
    torch.device
    """
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _preprocess_image(
    rgb: np.ndarray,
    size: Tuple[int, int] = (224, 224),
) -> "torch.Tensor":  # type: ignore
    """
    Preprocess a single RGB image for deep model inference.

    Steps
    -----
    1. Resize to (size, size).
    2. Convert to float32, scale to [0, 1].
    3. Apply ImageNet normalisation (mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]).
    4. Add batch dimension → shape (1, 3, H, W).

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3), dtype uint8
    size : tuple of int

    Returns
    -------
    torch.Tensor, shape (1, 3, H, W)
    """
    import torch

    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (tensor - mean) / std
    tensor = tensor.transpose(2, 0, 1)  # (3, H, W)
    return torch.from_numpy(tensor).unsqueeze(0)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseEmbeddingExtractor(ABC):
    """
    Abstract base class for all deep embedding extractors.

    Subclasses must implement ``_load_model`` and ``_forward``.
    All extractors expose a consistent ``extract`` method that handles
    batching, device placement, and optional L2 normalisation.
    """

    def __init__(
        self,
        device: str = "auto",
        input_size: Tuple[int, int] = (224, 224),
        batch_size: int = 8,
        normalize: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.input_size = input_size
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
        _logger.info(f"{self.__class__.__name__}: device={self.device}")

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model onto self.device."""
        ...

    @abstractmethod
    def _forward(self, batch_tensor: "torch.Tensor") -> np.ndarray:  # type: ignore
        """Run a single batch through the model and return embeddings."""
        ...

    def load(self) -> "BaseEmbeddingExtractor":
        """Load the model (lazy loading pattern)."""
        if self._model is None:
            self._load_model()
        return self

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded into memory."""
        return self._model is not None

    def extract(
        self,
        images: List[np.ndarray],
    ) -> np.ndarray:
        """
        Extract embeddings for a list of RGB images.

        Parameters
        ----------
        images : list of np.ndarray, shape (H, W, 3), dtype uint8
            Component images (white-background tissue crops recommended).

        Returns
        -------
        np.ndarray, shape (N, embedding_dim)
            Embedding matrix. If normalise=True, each row has L2 norm = 1.
        """
        import torch

        self.load()
        all_embeddings = []

        for start in range(0, len(images), self.batch_size):
            batch_imgs = images[start : start + self.batch_size]
            tensors = [_preprocess_image(img, self.input_size) for img in batch_imgs]
            batch = torch.cat(tensors, dim=0).to(self.device)

            with torch.no_grad():
                emb = self._forward(batch)  # (B, D) numpy

            all_embeddings.append(emb)

        embeddings = np.concatenate(all_embeddings, axis=0)

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            embeddings /= norms

        return embeddings


# ---------------------------------------------------------------------------
# ResNet50 Extractor
# ---------------------------------------------------------------------------


class ResNet50Extractor(BaseEmbeddingExtractor):
    """
    Extract 2048-dimensional embeddings from the ResNet50 penultimate layer.

    ResNet50 is a deep residual network pretrained on ImageNet (1.2M images,
    1000 classes).  We remove the final classification head and use the
    average-pooled feature vector (2048-d) as the tissue representation.

    Despite training on natural images, ResNet50 embeddings transfer well
    to histopathology because low-to-mid level texture and color features
    are domain-general.
    """

    def __init__(
        self,
        weights: str = "IMAGENET1K_V2",
        device: str = "auto",
        input_size: Tuple[int, int] = (224, 224),
        batch_size: int = 8,
        normalize: bool = True,
    ) -> None:
        super().__init__(device, input_size, batch_size, normalize)
        self.weights = weights

    def _load_model(self) -> None:
        import torch
        import torchvision.models as models

        _logger.info(f"Loading ResNet50 (weights={self.weights}) …")
        try:
            # torchvision >= 0.13
            weights_enum = models.ResNet50_Weights[self.weights]
            model = models.resnet50(weights=weights_enum)
        except (KeyError, AttributeError):
            # Older torchvision fallback
            model = models.resnet50(pretrained=True)

        # Remove classification head → output is 2048-d avg pool
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(self.device)
        self._model = model
        _logger.info("ResNet50 loaded. Embedding dim: 2048")

    def _forward(self, batch: "torch.Tensor") -> np.ndarray:  # type: ignore
        return self._model(batch).cpu().numpy()


# ---------------------------------------------------------------------------
# DINOv2 Extractor
# ---------------------------------------------------------------------------


class DINOv2Extractor(BaseEmbeddingExtractor):
    """
    Extract 768-dimensional CLS token embeddings from DINOv2 ViT-B/14.

    DINOv2 is a Vision Transformer trained with self-supervised distillation
    on a curated dataset of 142M images.  The CLS token captures global
    image semantics and produces embeddings with excellent transfer properties.

    Available variants and their embedding dimensions:
      - dinov2_vits14 : 384-d
      - dinov2_vitb14 : 768-d (default)
      - dinov2_vitl14 : 1024-d
      - dinov2_vitg14 : 1536-d

    Loaded via ``torch.hub`` (requires internet on first call).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str = "auto",
        input_size: Tuple[int, int] = (224, 224),
        batch_size: int = 4,
        normalize: bool = True,
    ) -> None:
        super().__init__(device, input_size, batch_size, normalize)
        self.model_name = model_name

    def _load_model(self) -> None:
        import torch
        _logger.info(f"Loading DINOv2 {self.model_name} via torch.hub …")
        model = torch.hub.load("facebookresearch/dinov2", self.model_name, verbose=False)
        model.eval()
        model.to(self.device)
        self._model = model
        _logger.info(f"DINOv2 {self.model_name} loaded.")

    def _forward(self, batch: "torch.Tensor") -> np.ndarray:  # type: ignore
        # DINOv2 forward_features returns dict; get CLS token
        output = self._model.forward_features(batch)
        if isinstance(output, dict):
            cls = output.get("x_norm_clstoken", output.get("cls_token"))
        else:
            cls = output[:, 0]  # ViT-style: first token is CLS
        return cls.cpu().numpy()


# ---------------------------------------------------------------------------
# Pathology Foundation Model Extractor
# ---------------------------------------------------------------------------


class PathologyModelExtractor(BaseEmbeddingExtractor):
    """
    Extract embeddings from a HuggingFace pathology foundation model.

    Supported models (loaded by model_id string):
      - 'MahmoodLab/UNI'   — Universal pathology encoder (Chen et al., 2024)
      - 'MahmoodLab/CONCH' — Contrastive multimodal encoder
      - 'vinid/plip'       — Pathology Language-Image Pretraining

    These models are specifically trained on millions of histopathology
    images and produce embeddings specifically adapted for tissue analysis.

    Requires access to gated HuggingFace repositories (UNI, CONCH) and
    appropriate API tokens.  Falls back gracefully to DINOv2 if unavailable.
    """

    def __init__(
        self,
        model_id: str = "MahmoodLab/UNI",
        device: str = "auto",
        input_size: Tuple[int, int] = (224, 224),
        batch_size: int = 4,
        normalize: bool = True,
        hf_token: Optional[str] = None,
    ) -> None:
        super().__init__(device, input_size, batch_size, normalize)
        self.model_id = model_id
        self.hf_token = hf_token
        self._processor = None
        self._available = False

    def _load_model(self) -> None:
        """Try to load the HuggingFace model; raise ImportError on failure."""
        try:
            import torch
            from transformers import AutoModel, AutoImageProcessor

            _logger.info(f"Loading pathology model: {self.model_id} …")
            kwargs = {}
            if self.hf_token:
                kwargs["token"] = self.hf_token

            processor = AutoImageProcessor.from_pretrained(self.model_id, **kwargs)
            model = AutoModel.from_pretrained(self.model_id, **kwargs)
            model.eval()
            model.to(self.device)
            self._model = model
            self._processor = processor
            self._available = True
            _logger.info(f"Pathology model {self.model_id} loaded.")
        except Exception as exc:
            _logger.warning(
                f"Could not load pathology model {self.model_id}: {exc}. "
                "Falling back to DINOv2."
            )
            # Fall back to DINOv2
            self._fallback = DINOv2Extractor(
                device=str(self.device),
                input_size=self.input_size,
                batch_size=self.batch_size,
                normalize=self.normalize,
            )
            self._available = False

    def _forward(self, batch: "torch.Tensor") -> np.ndarray:  # type: ignore
        if not self._available:
            return self._fallback._forward(batch)

        import torch
        outputs = self._model(pixel_values=batch)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.cpu().numpy()
        # Use mean of last hidden state as embedding
        return outputs.last_hidden_state[:, 0].cpu().numpy()

    def extract(self, images: List[np.ndarray]) -> np.ndarray:
        self.load()
        if not self._available and hasattr(self, "_fallback"):
            return self._fallback.extract(images)
        return super().extract(images)


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine similarity matrix.

    Cosine similarity(a, b) = (a · b) / (||a|| * ||b||)
    Range: [-1, 1]. Value of 1 = identical direction, 0 = orthogonal.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        Embedding matrix. Rows need not be L2-normalised; this function
        normalises internally.

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric cosine similarity matrix. Diagonal is 1.0.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normed = embeddings / norms
    return (normed @ normed.T).astype(np.float64)


def euclidean_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)

    Returns
    -------
    np.ndarray, shape (N, N)
    """
    from scipy.spatial.distance import cdist
    return cdist(embeddings, embeddings, metric="euclidean").astype(np.float64)


def extract_deep_features_batch(
    components: List["ComponentData"],  # type: ignore[name-defined]
    extractor: BaseEmbeddingExtractor,
    use_white_background: bool = True,
) -> np.ndarray:
    """
    Extract deep embeddings for a list of components using the given extractor.

    Parameters
    ----------
    components : list of ComponentData
    extractor : BaseEmbeddingExtractor
    use_white_background : bool
        If True, use ``component.masked_rgb`` (tissue on white background),
        which reduces background influence on embeddings.

    Returns
    -------
    np.ndarray, shape (N, embedding_dim)
    """
    if use_white_background:
        images = [c.masked_rgb for c in components]
    else:
        images = [c.rgb_crop for c in components]
    return extractor.extract(images)
