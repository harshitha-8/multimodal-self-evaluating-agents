"""
Cross-Modal Consistency Checker — Detects hallucinations and
inconsistencies between visual and textual modalities.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class CrossModalConsistency:
    """
    Check consistency between visual and textual reasoning.
    High inconsistency = potential hallucination.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.5)

    def check(self, visual_features: np.ndarray, text_features: np.ndarray) -> Dict[str, float]:
        """Check consistency between visual and text features."""
        # Normalize features
        v_norm = visual_features / (np.linalg.norm(visual_features) + 1e-8)
        t_norm = text_features / (np.linalg.norm(text_features) + 1e-8)

        # Compute similarity
        min_dim = min(len(v_norm), len(t_norm))
        similarity = float(np.dot(v_norm[:min_dim], t_norm[:min_dim]))

        return {
            "consistency_score": max(0, similarity),
            "is_consistent": similarity > self.similarity_threshold,
            "hallucination_risk": max(0, 1.0 - similarity),
        }

    def check_text_claims(self, text: str, visual_features: np.ndarray) -> Dict[str, Any]:
        """Check individual text claims against visual evidence."""
        claims = self._extract_claims(text)
        results = {
            "num_claims": len(claims),
            "verified_claims": 0,
            "unverified_claims": 0,
            "claims": [],
        }

        for claim in claims:
            is_grounded = self._verify_claim(claim, visual_features)
            results["claims"].append({"text": claim, "grounded": is_grounded})
            if is_grounded:
                results["verified_claims"] += 1
            else:
                results["unverified_claims"] += 1

        results["grounding_rate"] = (
            results["verified_claims"] / max(1, results["num_claims"])
        )
        return results

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return sentences

    def _verify_claim(self, claim: str, visual_features: np.ndarray) -> bool:
        """Verify a single claim against visual evidence (placeholder)."""
        # In full implementation: encode claim, compare with visual features
        return len(claim.split()) > 3  # Heuristic placeholder
