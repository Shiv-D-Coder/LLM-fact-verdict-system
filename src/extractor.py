# extractor.py
# claim extraction (spaCy)

import spacy
from typing import List, Dict, Any
from config import SPACY_MODEL


class ClaimExtractor:
    """Extract claims and entities from text using spaCy."""
    
    def __init__(self, model_name: str = SPACY_MODEL):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"SpaCy model '{model_name}' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
    
    def extract_claims(self, text: str) -> Dict[str, Any]:
        """
        Extract key claims and entities from input text.
        
        Returns:
            Dictionary with:
                - main_claim: The primary claim/statement
                - entities: Named entities found
                - key_phrases: Important noun phrases
        """
        doc = self.nlp(text)
        
        # Extract named entities
        entities = {
            "organizations": [],
            "locations": [],
            "dates": [],
            "money": [],
            "persons": [],
            "other": []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GOV"]:
                entities["organizations"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "MONEY":
                entities["money"].append(ent.text)
            elif ent.label_ == "PERSON":
                entities["persons"].append(ent.text)
            else:
                entities["other"].append(ent.text)
        
        # Extract noun phrases as key phrases
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # The main claim is typically the full sentence(s)
        # For fact-checking, we use the entire input as the claim
        main_claim = text.strip()
        
        # Extract verbs to understand the action/announcement
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        
        return {
            "main_claim": main_claim,
            "entities": entities,
            "key_phrases": key_phrases[:10],  # Top 10 phrases
            "verbs": verbs,
            "full_text": text
        }
    
    def get_search_keywords(self, claim_data: Dict[str, Any]) -> List[str]:
        """
        Extract keywords for enhanced retrieval.
        
        Args:
            claim_data: Output from extract_claims()
            
        Returns:
            List of important keywords for search
        """
        keywords = []
        
        # Add all entities
        for entity_list in claim_data["entities"].values():
            keywords.extend(entity_list)
        
        # Add key phrases
        keywords.extend(claim_data["key_phrases"])
        
        # Deduplicate and return
        return list(set(keywords))


# Singleton instance
_extractor_instance = None

def get_extractor() -> ClaimExtractor:
    """Get or create the global extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ClaimExtractor()
    return _extractor_instance


if __name__ == "__main__":
    # Test the extractor
    extractor = get_extractor()
    
    test_claim = "The Indian government has announced free electricity to all farmers starting July 2025."
    result = extractor.extract_claims(test_claim)
    
    print("Main Claim:", result["main_claim"])
    print("\nEntities:")
    for entity_type, entities in result["entities"].items():
        if entities:
            print(f"  {entity_type}: {entities}")
    print("\nKey Phrases:", result["key_phrases"])
    print("\nSearch Keywords:", extractor.get_search_keywords(result))

