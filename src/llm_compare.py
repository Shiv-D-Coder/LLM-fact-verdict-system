# llm_compare.py
# builds prompt and calls LLM

import json
from typing import List, Dict, Any
from openai import AsyncOpenAI
import asyncio

from retriever import RetrievalResult
from config import OPENAI_API_KEY, LLM_MODEL


class FactChecker:
    """LLM-powered fact comparison and verdict generation."""
    
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    def _build_prompt(self, claim: str, evidence: List[RetrievalResult]) -> str:
        """
        Build a structured prompt for fact-checking.
        
        Args:
            claim: The claim to verify
            evidence: List of retrieved evidence documents
            
        Returns:
            Formatted prompt string
        """
        evidence_text = ""
        for i, result in enumerate(evidence, start=1):
            evidence_text += f"\n[Evidence {i}]\n"
            evidence_text += f"Source ID: {result.document.source_id}\n"
            evidence_text += f"Source Type: {result.document.source_type}\n"
            evidence_text += f"Source Name: {result.document.source_name}\n"
            evidence_text += f"Date: {result.document.date}\n"
            evidence_text += f"Similarity Score: {result.similarity_score:.3f}\n"
            evidence_text += f"Content: {result.document.text}\n"
        
        prompt = f"""You are a fact-checking expert. Your task is to verify a claim against retrieved evidence from trusted sources (PIB - Press Information Bureau of India and curated knowledge base documents).

**Claim to Verify:**
{claim}

**Retrieved Evidence:**
{evidence_text}

**Instructions:**
1. Carefully analyze the claim against each piece of evidence
2. Determine if the claim is:
   - "True" - The claim is supported by the evidence
   - "False" - The claim contradicts the evidence
   - "Unverifiable" - Insufficient evidence to confirm or deny
3. Provide detailed reasoning explaining your verdict
4. Cite specific evidence sources that support your conclusion
5. Identify any key facts that are relevant

**Output Format (JSON):**
{{
  "verdict": "True|False|Unverifiable",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation of your verdict, referencing specific evidence",
  "evidence_used": ["source_id_1", "source_id_2"],
  "key_facts": ["fact 1", "fact 2"],
  "contradictions": ["any contradictions found"] or null
}}

Respond ONLY with valid JSON, no additional text."""
        
        return prompt
    
    async def compare(
        self, 
        claim: str, 
        evidence: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        Compare claim against evidence using LLM.
        
        Args:
            claim: The claim to verify
            evidence: Retrieved evidence documents
            
        Returns:
            Dictionary with verdict, reasoning, and metadata
        """
        if not evidence:
            return {
                "verdict": "Unverifiable",
                "confidence": 0.0,
                "reasoning": "No relevant evidence found in the knowledge base.",
                "evidence_used": [],
                "key_facts": [],
                "contradictions": None,
                "raw_evidence": []
            }
        
        # Build prompt
        prompt = self._build_prompt(claim, evidence)
        
        try:
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise fact-checking assistant. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Add evidence metadata
            result["raw_evidence"] = [r.to_dict() for r in evidence]
            result["claim"] = claim
            
            return result
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {
                "verdict": "Error",
                "confidence": 0.0,
                "reasoning": f"Error during fact-checking: {str(e)}",
                "evidence_used": [],
                "key_facts": [],
                "contradictions": None,
                "raw_evidence": [r.to_dict() for r in evidence],
                "claim": claim
            }
    
    def compare_sync(self, claim: str, evidence: List[RetrievalResult]) -> Dict[str, Any]:
        """Synchronous wrapper for compare."""
        return asyncio.run(self.compare(claim, evidence))


# Singleton instance
_fact_checker_instance = None

def get_fact_checker() -> FactChecker:
    """Get or create the global fact checker instance."""
    global _fact_checker_instance
    if _fact_checker_instance is None:
        _fact_checker_instance = FactChecker()
    return _fact_checker_instance


async def main():
    """Test the fact checker."""
    from retriever import get_retriever
    
    retriever = get_retriever()
    fact_checker = get_fact_checker()
    
    test_claim = "The Indian government has announced free electricity to all farmers starting July 2025."
    print(f"Claim: {test_claim}\n")
    
    # Retrieve evidence
    print("Retrieving evidence...")
    evidence = await retriever.retrieve(test_claim, top_k=5)
    print(f"Found {len(evidence)} evidence documents\n")
    
    # Check fact
    print("Checking fact...")
    result = await fact_checker.compare(test_claim, evidence)
    
    print("\n=== FACT CHECK RESULT ===")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nReasoning:\n{result['reasoning']}")
    print(f"\nEvidence Used: {result['evidence_used']}")
    print(f"\nKey Facts: {result['key_facts']}")


if __name__ == "__main__":
    asyncio.run(main())

