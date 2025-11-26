# pipeline.py
# ties components together

import asyncio
import json
import argparse
from typing import Dict, Any
from datetime import datetime

from extractor import get_extractor
from retriever import get_retriever
from llm_compare import get_fact_checker
from config import TOP_K_RETRIEVAL


class FactCheckPipeline:
    """
    Main fact-checking pipeline orchestrating all components.
    
    Workflow:
        1. Extract claims and entities from input text
        2. Retrieve relevant evidence from vector store
        3. Compare claim against evidence using LLM
        4. Format and return structured output
    """
    
    def __init__(self):
        self.extractor = get_extractor()
        self.retriever = get_retriever()
        self.fact_checker = get_fact_checker()
    
    async def check_fact(
        self, 
        claim: str, 
        top_k: int = TOP_K_RETRIEVAL,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete fact-checking pipeline.
        
        Args:
            claim: The claim to verify
            top_k: Number of evidence documents to retrieve
            verbose: If True, print progress messages
            
        Returns:
            Dictionary with verdict, evidence, reasoning, and sources
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"FACT CHECKING PIPELINE")
            print(f"{'='*60}\n")
            print(f"Claim: {claim}\n")
        
        # Step 1: Extract claims and entities
        if verbose:
            print("[1/3] Extracting claims and entities...")
        
        claim_data = self.extractor.extract_claims(claim)
        keywords = self.extractor.get_search_keywords(claim_data)
        
        if verbose:
            print(f"  ✓ Extracted {len(keywords)} keywords")
            print(f"  Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}\n")
        
        # Step 2: Retrieve evidence
        if verbose:
            print(f"[2/3] Retrieving top-{top_k} evidence documents...")
        
        evidence = await self.retriever.retrieve(
            query=claim,
            top_k=top_k,
            keywords=keywords
        )
        
        if verbose:
            print(f"  ✓ Retrieved {len(evidence)} documents")
            if evidence:
                print(f"  Top match: {evidence[0].document.source_id} (similarity: {evidence[0].similarity_score:.3f})\n")
            else:
                print("  ⚠ No evidence found\n")
        
        # Step 3: LLM comparison and verdict
        if verbose:
            print("[3/3] Analyzing claim against evidence...")
        
        result = await self.fact_checker.compare(claim, evidence)
        
        if verbose:
            print(f"  ✓ Analysis complete\n")
        
        # Format final output
        output = {
            "claim": claim,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "evidence": [
                {
                    "source_id": e["source_id"],
                    "source_type": e["source_type"],
                    "source_name": e["source_name"],
                    "date": e["date"],
                    "similarity_score": e["similarity_score"],
                    "text": e["text"]
                }
                for e in result["raw_evidence"]
            ],
            "evidence_used": result.get("evidence_used", []),
            "key_facts": result.get("key_facts", []),
            "contradictions": result.get("contradictions"),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "keywords_extracted": keywords,
                "entities": claim_data["entities"],
                "num_evidence_retrieved": len(evidence)
            }
        }
        
        return output
    
    def check_fact_sync(self, claim: str, top_k: int = TOP_K_RETRIEVAL, verbose: bool = False) -> Dict[str, Any]:
        """Synchronous wrapper for check_fact."""
        return asyncio.run(self.check_fact(claim, top_k, verbose))


def format_output(result: Dict[str, Any], format_type: str = "pretty") -> str:
    """
    Format the fact-check result for display.
    
    Args:
        result: Output from check_fact()
        format_type: "pretty" or "json"
        
    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    # Pretty format
    output = []
    output.append("\n" + "="*60)
    output.append("FACT CHECK RESULT")
    output.append("="*60 + "\n")
    
    # Verdict
    verdict_emoji = {
        "True": "✅",
        "False": "❌",
        "Unverifiable": "🤷‍♂️",
        "Error": "⚠️"
    }
    emoji = verdict_emoji.get(result["verdict"], "")
    output.append(f"Verdict: {emoji} {result['verdict']}")
    output.append(f"Confidence: {result['confidence']:.2f}\n")
    
    # Reasoning
    output.append("Reasoning:")
    output.append(result["reasoning"])
    output.append("")
    
    # Key Facts
    if result.get("key_facts"):
        output.append("Key Facts:")
        for fact in result["key_facts"]:
            output.append(f"  • {fact}")
        output.append("")
    
    # Evidence
    output.append(f"Evidence Retrieved ({len(result['evidence'])} documents):")
    for i, ev in enumerate(result["evidence"], start=1):
        output.append(f"\n[{i}] {ev['source_id']} ({ev['source_type']})")
        output.append(f"    Source: {ev['source_name']}")
        output.append(f"    Date: {ev['date']}")
        output.append(f"    Similarity: {ev['similarity_score']:.3f}")
        output.append(f"    Text: {ev['text'][:150]}{'...' if len(ev['text']) > 150 else ''}")
    
    # Sources Used
    if result.get("evidence_used"):
        output.append(f"\nSources Cited: {', '.join(result['evidence_used'])}")
    
    output.append("\n" + "="*60)
    
    return "\n".join(output)


async def main():
    """CLI interface for the fact-checking pipeline."""
    parser = argparse.ArgumentParser(description="LLM-Powered Fact Checker with Advanced RAG")
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="The claim to fact-check"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K_RETRIEVAL,
        help=f"Number of evidence documents to retrieve (default: {TOP_K_RETRIEVAL})"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (default: pretty)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save output to file"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = FactCheckPipeline()
    result = await pipeline.check_fact(
        claim=args.query,
        top_k=args.top_k,
        verbose=args.verbose
    )
    
    # Format output
    formatted = format_output(result, args.format)
    
    # Display
    print(formatted)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.format == "json":
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                f.write(formatted)
        print(f"\n✓ Output saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

