"""Tests for perception modules."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from msea.perception.encoders import CLIPEncoder, DINOv2Encoder, get_encoder
from msea.perception.features import FeatureExtractor
from msea.perception.uncertainty import PerceptualUncertainty


def test_clip_encoder():
    encoder = CLIPEncoder({"embed_dim": 512})
    features = encoder.encode(np.zeros((224, 224, 3)))
    assert features.shape == (512,)
    assert abs(np.linalg.norm(features) - 1.0) < 0.01
    print("✓ test_clip_encoder passed")


def test_dinov2_encoder():
    encoder = DINOv2Encoder({"embed_dim": 768})
    features = encoder.encode(np.zeros((224, 224, 3)))
    assert features.shape == (768,)
    print("✓ test_dinov2_encoder passed")


def test_encoder_factory():
    encoder = get_encoder("clip")
    assert isinstance(encoder, CLIPEncoder)
    encoder = get_encoder("dinov2")
    assert isinstance(encoder, DINOv2Encoder)
    print("✓ test_encoder_factory passed")


def test_feature_extractor():
    config = {"encoders": [{"name": "clip", "embed_dim": 512}], "fusion_method": "concatenation"}
    extractor = FeatureExtractor(config)
    features = extractor.extract(np.zeros((224, 224, 3)))
    assert "fused" in features
    print("✓ test_feature_extractor passed")


def test_uncertainty_estimation():
    config = {"method": "simple"}
    unc = PerceptualUncertainty(config)
    features = np.random.randn(768).astype(np.float32)
    estimate = unc.estimate(features)
    assert 0 <= estimate.total <= 1
    assert estimate.method == "simple"
    print("✓ test_uncertainty_estimation passed")


def test_ensemble_uncertainty():
    config = {"method": "ensemble"}
    unc = PerceptualUncertainty(config)
    f1 = np.random.randn(512).astype(np.float32)
    f2 = np.random.randn(512).astype(np.float32)
    f3 = np.random.randn(512).astype(np.float32)
    estimate = unc.estimate(f1, [f2, f3])
    assert 0 <= estimate.epistemic <= 1
    print("✓ test_ensemble_uncertainty passed")


if __name__ == "__main__":
    test_clip_encoder()
    test_dinov2_encoder()
    test_encoder_factory()
    test_feature_extractor()
    test_uncertainty_estimation()
    test_ensemble_uncertainty()
    print("\nAll perception tests passed! ✓")
