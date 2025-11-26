# nli.py
# optional entailment functions

"""
Optional Natural Language Inference (NLI) module for entailment scoring.
This can be used as an additional signal alongside vector similarity.
"""

from typing import List, Tuple
import warnings

# Suppress transformers warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False
    print("Warning: transformers not installed. NLI scoring unavailable.")


class NLIScorer:
    """Natural Language Inference scorer for claim-evidence entailment."""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base-mnli"):
        if not NLI_AVAILABLE:
            raise ImportError("transformers library required for NLI scoring")
        
        self.model_name = model_name
        self.pipeline = None
    
    def _load_model(self):
        """Lazy load the NLI model."""
        if self.pipeline is None:
            print(f"Loading NLI model: {self.model_name}...")
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=-1  # CPU
            )
    
    def score_entailment(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Score entailment between premise (evidence) and hypothesis (claim).
        
        Args:
            premise: The evidence text
            hypothesis: The claim to verify
            
        Returns:
            Tuple of (label, score) where:
                label: "entailment", "neutral", or "contradiction"
                score: Confidence score (0-1)
        """
        self._load_model()
        
        result = self.pipeline(f"{premise} [SEP] {hypothesis}")[0]
        label = result['label'].lower()
        score = result['score']
        
        return label, score
    
    def score_evidence_list(
        self, 
        claim: str, 
        evidence_texts: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Score entailment for a claim against multiple evidence texts.
        
        Args:
            claim: The claim to verify
            evidence_texts: List of evidence texts
            
        Returns:
            List of (label, score) tuples for each evidence
        """
        self._load_model()
        
        results = []
        for evidence in evidence_texts:
            label, score = self.score_entailment(evidence, claim)
            results.append((label, score))
        
        return results


# Singleton instance
_nli_scorer_instance = None

def get_nli_scorer() -> NLIScorer:
    """Get or create the global NLI scorer instance."""
    global _nli_scorer_instance
    if _nli_scorer_instance is None:
        if not NLI_AVAILABLE:
            raise ImportError("transformers library required for NLI scoring")
        _nli_scorer_instance = NLIScorer()
    return _nli_scorer_instance


if __name__ == "__main__":
    # Test NLI scorer
    if NLI_AVAILABLE:
        scorer = get_nli_scorer()
        
        claim = "The Indian government provides free electricity to farmers."
        evidence = "The Ministry of Power launched the PM Surya Ghar Muft Bijli Yojana aimed at providing free electricity up to 300 units for households with rooftop solar installations."
        
        label, score = scorer.score_entailment(evidence, claim)
        print(f"Claim: {claim}")
        print(f"Evidence: {evidence}")
        print(f"\nNLI Result: {label} (confidence: {score:.3f})")
    else:
        print("NLI module requires transformers library. Install with: pip install transformers torch")

