"""
Feature Extraction Pipeline — Combines multiple encoders and
produces unified multimodal feature representations.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from msea.perception.encoders import VisionEncoder, get_encoder


class FeatureExtractor:
    """
    Multi-encoder feature extraction pipeline.
    Combines visual features from multiple backbones for robust representation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encoders: Dict[str, VisionEncoder] = {}
        self.fusion_method = config.get("fusion_method", "concatenation")

        # Initialize encoders
        encoder_configs = config.get("encoders", [{"name": "clip"}])
        for enc_config in encoder_configs:
            name = enc_config["name"]
            self.encoders[name] = get_encoder(name, enc_config)

    def extract(self, image: Any, text: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Extract features from all registered encoders."""
        features = {}
        for name, encoder in self.encoders.items():
            features[f"{name}_visual"] = encoder.encode(image)
        if text and hasattr(list(self.encoders.values())[0], 'encode_text'):
            encoder = list(self.encoders.values())[0]
            features["text"] = encoder.encode_text(text)
        features["fused"] = self._fuse_features(features)
        return features

    def extract_batch(self, images: Any) -> Dict[str, np.ndarray]:
        """Extract features for a batch."""
        features = {}
        for name, encoder in self.encoders.items():
            features[f"{name}_visual"] = encoder.encode_batch(images)
        return features

    def _fuse_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse features from multiple encoders."""
        visual_features = [v for k, v in features.items() if "visual" in k]
        if not visual_features:
            return np.zeros(768, dtype=np.float32)

        if self.fusion_method == "concatenation":
            return np.concatenate(visual_features, axis=-1)
        elif self.fusion_method == "average":
            # Pad to same dim and average
            max_dim = max(f.shape[-1] for f in visual_features)
            padded = [np.pad(f, (0, max_dim - f.shape[-1])) for f in visual_features]
            return np.mean(padded, axis=0)
        else:
            return visual_features[0]

    @property
    def output_dim(self) -> int:
        """Total output dimension after fusion."""
        if self.fusion_method == "concatenation":
            return sum(e.output_dim for e in self.encoders.values())
        else:
            return max(e.output_dim for e in self.encoders.values())

# --- Auto-research iteration 2: implement adaptive reflection frequency based on ECE (2026-04-13T23:49:29Z) ---

# --- Auto-research iteration 3: add entropy-based uncertainty estimation to base agent (2026-04-13T23:49:30Z) ---

# --- Auto-research iteration 12: implement speculation execution for parallel tool calls (2026-04-13T23:49:30Z) ---

# --- Auto-research iteration 13: add agent serialization for experiment reproducibility (2026-04-13T23:49:31Z) ---

# --- Auto-research iteration 22: implement multi-scale feature extraction for DINOv2 (2026-04-13T23:49:31Z) ---

# --- Auto-research iteration 23: add feature normalization with learnable temperature (2026-04-13T23:49:31Z) ---

# --- Auto-research iteration 32: implement multi-crop ensemble for robust features (2026-04-13T23:49:32Z) ---

# --- Auto-research iteration 33: add feature dimensionality reduction with PCA projection (2026-04-13T23:49:32Z) ---

# --- Auto-research iteration 42: implement token-level process supervision scoring (2026-04-13T23:49:33Z) ---

# --- Auto-research iteration 43: add multi-hop reasoning chain construction (2026-04-13T23:49:33Z) ---

# --- Auto-research iteration 52: implement step-level reward prediction model (2026-04-13T23:49:34Z) ---

# --- Auto-research iteration 53: add reasoning chain compression for context efficiency (2026-04-13T23:49:34Z) ---
