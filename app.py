import streamlit as st
import asyncio
import os
import sys
from pathlib import Path
import json
from src.pipeline import FactCheckPipeline
from src.config import TOP_K_RETRIEVAL
# Ensure spaCy model is available
import spacy
try:
    spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    
    
# Page configuration
st.set_page_config(
    page_title="Fact Checker",
    # page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .verdict-true {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .verdict-false {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .verdict-unverifiable {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .evidence-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #007bff;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # st.image("https://img.icons8.com/fluency/96/000000/search.png", width=80)
    st.title("⚙️ Configuration")
    
    # API Key input
    st.subheader("🔑 OpenAI API Key")
    api_key = st.text_input(
        "Enter your OpenAI API key",
        type="password",
        help="Your API key is required to run the fact checker. It will not be stored."
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to continue")
    
    st.divider()
    
    # Advanced settings
    st.subheader("🎛️ Advanced Settings")
    top_k = st.slider(
        "Number of evidence documents",
        min_value=1,
        max_value=10,
        value=TOP_K_RETRIEVAL,
        help="How many relevant documents to retrieve from the knowledge base"
    )
    
    show_metadata = st.checkbox("Show metadata", value=True)
    show_entities = st.checkbox("Show extracted entities", value=True)
    
    st.divider()
    
    # Information
    st.subheader("ℹ️ About")
    st.markdown("""
    This fact-checking system uses:
    - **Advanced RAG** with hybrid retrieval
    - **FAISS** vector search
    - **GPT-4** for analysis
    - **75 PIB facts** + 5 KB documents
    
    """)

# Main content
st.markdown('<div class="main-header"> LLM-Powered Fact Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Verify claims against trusted government sources using Advanced RAG</div>', unsafe_allow_html=True)

# Input section
st.subheader("📝 Enter a Claim to Verify")
claim_input = st.text_area(
    "Type or paste the claim you want to fact-check:",
    height=100,
    placeholder="Example: The Indian government has announced free electricity to all farmers starting July 2025.",
    help="Enter any factual claim related to Indian government policies, announcements, or initiatives"
)

# Example claims
with st.expander("💡 Try these example claims"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 PM Kisan expansion"):
            claim_input = "The PM Kisan scheme was expanded to include tenant farmers in 2024"
            st.rerun()
        if st.button("⚡ Free electricity claim"):
            claim_input = "The Indian government has announced free electricity to all farmers starting July 2025"
            st.rerun()
    with col2:
        if st.button("🤖 IndiaAI Mission"):
            claim_input = "India launched the IndiaAI Mission in 2023"
            st.rerun()
        if st.button("🚂 Vande Bharat trains"):
            claim_input = "The Ministry of Railways approved 100 new Vande Bharat trains for 2025 rollout"
            st.rerun()

# Check button
check_button = st.button("🔍 Check Fact", type="primary", use_container_width=True)

# Process fact check
if check_button and claim_input:
    if not api_key:
        st.error("❌ Please enter your OpenAI API key in the sidebar to continue")
    else:
        with st.spinner("🔄 Analyzing claim... This may take a few seconds"):
            try:
                # Run pipeline
                pipeline = FactCheckPipeline()
                result = asyncio.run(pipeline.check_fact(claim_input, top_k=top_k, verbose=False))
                
                # Display results
                st.divider()
                st.subheader("📊 Analysis Results")
                
                # Verdict section
                verdict = result["verdict"]
                confidence = result["confidence"]
                
                verdict_class = {
                    "True": "verdict-true",
                    "False": "verdict-false",
                    "Unverifiable": "verdict-unverifiable"
                }.get(verdict, "verdict-unverifiable")
                
                verdict_emoji = {
                    "True": "✅",
                    "False": "❌",
                    "Unverifiable": "🤷‍♂️"
                }.get(verdict, "⚠️")
                
                st.markdown(f'<div class="{verdict_class}">', unsafe_allow_html=True)
                st.markdown(f"### {verdict_emoji} Verdict: **{verdict}**")
                st.markdown(f"**Confidence Score:** {confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Evidence Retrieved", len(result["evidence"]))
                with col2:
                    st.metric("Sources Cited", len(result.get("evidence_used", [])))
                with col3:
                    avg_similarity = sum(e["similarity_score"] for e in result["evidence"]) / len(result["evidence"]) if result["evidence"] else 0
                    st.metric("Avg Similarity", f"{avg_similarity:.1%}")
                
                # Reasoning
                st.subheader("💭 Detailed Reasoning")
                st.info(result["reasoning"])
                
                # Key Facts
                if result.get("key_facts"):
                    st.subheader("🔑 Key Facts Identified")
                    for fact in result["key_facts"]:
                        st.markdown(f"- {fact}")
                
                # Contradictions
                if result.get("contradictions"):
                    st.subheader("⚠️ Contradictions Found")
                    for contradiction in result["contradictions"]:
                        st.warning(contradiction)
                
                # Evidence section
                st.subheader("📚 Evidence Documents")
                for i, evidence in enumerate(result["evidence"], start=1):
                    with st.expander(f"📄 Evidence {i}: {evidence['source_id']} (Similarity: {evidence['similarity_score']:.1%})"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**Source Type:** {evidence['source_type']}")
                            st.markdown(f"**Source Name:** {evidence['source_name']}")
                        with col2:
                            st.markdown(f"**Date:** {evidence['date']}")
                            st.markdown(f"**Similarity:** {evidence['similarity_score']:.3f}")
                        
                        st.markdown("**Content:**")
                        st.markdown(f"> {evidence['text']}")
                        
                        # Highlight if used in reasoning
                        if evidence['source_id'] in result.get('evidence_used', []):
                            st.success("✓ This source was cited in the reasoning")
                
                # Metadata section
                if show_metadata and result.get("metadata"):
                    with st.expander("🔍 Analysis Metadata"):
                        metadata = result["metadata"]
                        st.json({
                            "timestamp": metadata.get("timestamp"),
                            "num_evidence_retrieved": metadata.get("num_evidence_retrieved"),
                            "keywords_extracted": metadata.get("keywords_extracted", [])[:10]
                        })
                
                # Entities section
                if show_entities and result.get("metadata", {}).get("entities"):
                    with st.expander("🏷️ Extracted Entities"):
                        entities = result["metadata"]["entities"]
                        for entity_type, entity_list in entities.items():
                            if entity_list:
                                st.markdown(f"**{entity_type.title()}:** {', '.join(entity_list)}")
                
                # Download results
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=json.dumps(result, indent=2, ensure_ascii=False),
                        file_name=f"fact_check_{result['metadata']['timestamp']}.json",
                        mime="application/json"
                    )
                with col2:
                    # Create text report with document content
                    text_report = f"""FACT CHECK REPORT
{'='*60}

Claim: {result['claim']}
Verdict: {verdict} ({confidence:.1%} confidence)

Reasoning:
{result['reasoning']}

{'='*60}
EVIDENCE DOCUMENTS
{'='*60}

"""
                    for i, ev in enumerate(result["evidence"], start=1):
                        text_report += f"\n[Document {i}] {ev['source_id']} ({ev['source_type']})\n"
                        text_report += f"Source: {ev['source_name']}\n"
                        text_report += f"Date: {ev['date']}\n"
                        text_report += f"Similarity: {ev['similarity_score']:.3f}\n"
                        text_report += f"Content:\n{ev['text']}\n"
                        text_report += "-"*40 + "\n"

                    text_report += f"\nGenerated: {result['metadata']['timestamp']}\n"
                    
                    st.download_button(
                        label="📄 Download Text Report",
                        data=text_report,
                        file_name=f"fact_check_{result['metadata']['timestamp']}.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Error during fact-checking: {str(e)}")
                st.exception(e)

elif check_button and not claim_input:
    st.warning("⚠️ Please enter a claim to fact-check")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with ❤️ By <strong>Shiv Patel</strong> | Powered by OpenAI GPT & Advanced RAG</p>
    <p style='font-size: 0.8rem;'>Data sources: PIB (Press Information Bureau of India) & Curated Knowledge Base</p>
</div>
""", unsafe_allow_html=True)
