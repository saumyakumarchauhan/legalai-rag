import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import google.generativeai as genai
import os
import re
from typing import List, Dict, Tuple

# ---------------- OCR & Document Imports ----------------
import pytesseract
import docx
from PIL import Image
from langdetect import detect_langs
import easyocr
import fitz  # PyMuPDF - Better PDF handling
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
# ========== CONFIG ==========
INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_index.bin.meta.json"
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K_EXTERNAL = 3  # Top-K from external FAISS index
TOP_K_DOCUMENT = 2  # Top-K from uploaded document
TOP_K_VERIFICATION = 3  # Top-K for document verification
GEMINI_MODEL = "gemini-1.5-flash"
DOCUMENT_PATH = "pan.pdf"  # can now be txt, pdf, docx, images
QUESTION_PATH = "question.txt"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between chunks

# ============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


# ----------------------------- 
# Enhanced PDF Extraction Functions
# ----------------------------- 
def extract_text_from_pdf_pymupdf(pdf_path):
    """Extract text from PDF using PyMuPDF (fitz) - more reliable than PyPDF2."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            text += page_text + "\n"
            
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"⚠️ PyMuPDF extraction failed: {e}")
        return ""

# def extract_text_from_pdf_with_ocr(pdf_path):
#     """Extract text from PDF using OCR as fallback."""
#     try:
#         import fitz
#         doc = fitz.open(pdf_path)
#         text = ""
        
#         for page_num in range(len(doc)):
#             page = doc[page_num]
            
#             # First try to get text directly
#             page_text = page.get_text()
            
#             # If no text or very little text, use OCR
#             if len(page_text.strip()) < 50:
#                 # Convert page to image
#                 pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
#                 img_data = pix.tobytes("png")
                
#                 # Save temporarily and OCR
#                 temp_img_path = f"temp_page_{page_num}.png"
#                 with open(temp_img_path, "wb") as f:
#                     f.write(img_data)
                
#                 # OCR the image
#                 try:
#                     img = Image.open(temp_img_path)
#                     ocr_text = ocr_with_fallback_strategies(img)
#                     page_text = ocr_text if len(ocr_text) > len(page_text) else page_text
#                 finally:
#                     # Clean up temp file
#                     if os.path.exists(temp_img_path):
#                         os.remove(temp_img_path)
            
#             text += page_text + "\n"
        
#         doc.close()
#         return text.strip()
        
#     except Exception as e:
#         print(f"⚠️ PDF OCR extraction failed: {e}")
#         return ""
def extract_text_from_pdf_with_ocr(pdf_path):
    """Extract text from PDF using OCR as fallback."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            if len(page_text.strip()) < 50:
                # Convert to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                from io import BytesIO
                img = Image.open(BytesIO(pix.tobytes("png")))
                
                # OCR without score filtering
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                
                # If OCR fails, try EasyOCR
                if len(ocr_text.strip()) < 20:
                    ocr_text = ocr_with_easyocr(img)
                
                page_text = ocr_text  # ALWAYS replace if PyMuPDF is empty
            
            text += page_text + "\n"
        
        doc.close()
        return text.strip()
        
    except Exception as e:
        print(f"⚠️ PDF OCR extraction failed: {e}")
        return ""


def detect_languages(text_chunk):
    """Detect probable languages in a text chunk and return Tesseract-compatible language string."""
    try:
        langs = detect_langs(text_chunk)
        langs_sorted = sorted(langs, key=lambda x: x.prob, reverse=True)
        tess_langs = []
        
        for l in langs_sorted:
            code = l.lang
            if code == "en":
                tess_langs.append("eng")
            elif code in ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa"]:
                mapping = {"hi":"hin","ta":"tam","te":"tel","bn":"ben","mr":"mar",
                          "gu":"guj","kn":"kan","ml":"mal","pa":"pan"}
                tess_langs.append(mapping[code])
        
        tess_langs = list(set(tess_langs))
        return "+".join(tess_langs) if tess_langs else "eng"
    except:
        return "eng"

def ocr_with_fallback_strategies(img: Image.Image):
    """Generic OCR with multiple fallback strategies for any document type."""
    # Try multiple OCR strategies in order of likelihood
    strategies = [
        # Strategy 1: English only with optimal settings
        {"lang": "eng", "config": "--psm 6 -c preserve_interword_spaces=1"},
        # Strategy 2: Try different page segmentation modes
        {"lang": "eng", "config": "--psm 4"},
        {"lang": "eng", "config": "--psm 3"},
        # Strategy 3: English with common languages
        {"lang": "eng+hin", "config": "--psm 6"},
        {"lang": "eng+ben", "config": "--psm 6"},
    ]
    
    best_text = ""
    best_score = 0
    
    for i, strategy in enumerate(strategies):
        try:
            text = pytesseract.image_to_string(img, 
                                             lang=strategy["lang"], 
                                             config=strategy["config"])
            
            # Score the text quality generically
            score = score_text_quality_generic(text)
            
            if score > best_score:
                best_text = text
                best_score = score
                
            if score > 0.7:  # Good enough quality
                break
                
        except Exception as e:
            continue
    
    return best_text.strip()

def score_text_quality_generic(text):
    """Generic text quality scoring for any document type."""
    if not text or len(text.strip()) < 20:
        return 0
    
    # Count valid characteristics
    lines = text.split('\n')
    words = text.split()
    
    # Check for reasonable word length distribution
    valid_words = [word for word in words if 2 <= len(word) <= 25]
    word_ratio = len(valid_words) / len(words) if words else 0
    
    # Check for reasonable line length (not too long/short)
    valid_lines = [line for line in lines if 5 <= len(line.strip()) <= 120]
    line_ratio = len(valid_lines) / len(lines) if lines else 0
    
    # Check for common document elements
    common_elements = sum([
        bool(re.search(r'\d', text)),  # Contains numbers
        bool(re.search(r'[A-Za-z]{3,}', text)),  # Contains meaningful words
        bool(re.search(r'[.!?]', text)),  # Contains punctuation
        len(set(words)) > 5,  # Has vocabulary variety
    ])
    
    # Calculate final score
    score = (word_ratio * 0.3) + (line_ratio * 0.3) + (common_elements * 0.1)
    return min(score, 1.0)

def ocr_with_easyocr(img_path):
    """Alternative OCR using EasyOCR (no poppler needed)."""
    try:
        reader = easyocr.Reader(['en'])  # Initialize with English
        result = reader.readtext(img_path, paragraph=True)
        text = " ".join([item[1] for item in result])
        return text
    except Exception as e:
        print(f"⚠️ EasyOCR failed: {e}")
        return ""

def load_document(path):
    """Load and extract text from various document formats with improved PDF handling."""
    ext = os.path.splitext(path)[1].lower()
    text = ""
    
    if ext == ".txt":
        print("📝 Text file detected → Reading directly...")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    
    elif ext == ".pdf":
        print("📄 PDF detected → Using enhanced extraction methods...")
        
        # Method 1: Try PyMuPDF first (most reliable)
        text = extract_text_from_pdf_pymupdf(path)
        print(f"   PyMuPDF extracted {len(text)} characters")
        
        # Method 2: If little text, try PyPDF2 as backup
        if len(text.strip()) < 100:
            print("   Trying PyPDF2 as backup...")
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(path)
                pypdf2_text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    pypdf2_text += page_text + "\n"
                
                if len(pypdf2_text) > len(text):
                    text = pypdf2_text
                    print(f"   Used PyPDF2, extracted {len(text)} characters")
            except Exception as e:
                print(f"⚠️ PyPDF2 extraction failed: {e}")
        
        # Method 3: If still no text, try pdfminer
        if len(text.strip()) < 100:
            print("   Trying pdfminer as backup...")
            try:
                from pdfminer.high_level import extract_text as pdfminer_extract_text
                pdfminer_text = pdfminer_extract_text(path)
                if pdfminer_text and len(pdfminer_text) > len(text):
                    text = pdfminer_text
                    print(f"   Used pdfminer, extracted {len(text)} characters")
            except ImportError:
                print("💡 Install pdfminer.six: pip install pdfminer.six")
            except Exception as e:
                print(f"⚠️ pdfminer failed: {e}")
        
        # Method 4: If still no meaningful text, try OCR
        if len(text.strip()) < 50:
            print("   Trying OCR extraction as final fallback...")
            ocr_text = extract_text_from_pdf_with_ocr(path)
            if ocr_text and len(ocr_text) > len(text):
                text = ocr_text
                print(f"   Used OCR, extracted {len(text)} characters")
    
    elif ext in [".doc", ".docx"]:
        print("📝 Word document detected → Extracting text...")
        doc = docx.Document(path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        print("🖼️  Image detected → Using OCR...")
        # First try pytesseract
        img = Image.open(path)
        text = ocr_with_fallback_strategies(img)
        
        # If that fails, try EasyOCR
        if len(text.strip()) < 50:
            print("   Trying EasyOCR as fallback...")
            easyocr_text = ocr_with_easyocr(path)
            if len(easyocr_text) > len(text):
                text = easyocr_text
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Clean up the extracted text
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Text Chunking Functions
# -----------------------------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if not text.strip():
        return [""]  # Return empty chunk instead of empty list
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []
    total_len = 0
    
    for sentence in sentences:
        if total_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current).strip())
            current = current[-overlap//10:]  # keep ~overlap words
            total_len = sum(len(s) for s in current)
        current.append(sentence)
        total_len += len(sentence)
    
    if current:
        chunks.append(" ".join(current).strip())
    
    return chunks if chunks else [""]

# ----------------------------- 
# FAISS + Embedding Functions
# ----------------------------- 
def load_index():
    print(f"📖 Loading FAISS index from {INDEX_PATH}...")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if index.ntotal != len(meta["ids"]):
        print("⚠️ WARNING: Index and metadata size mismatch!")
    print(f"✅ Loaded index with {index.ntotal} entries, dim {index.d}")
    return index, meta

def embed_texts(texts, model, batch_size=32):
    """Generate embeddings for texts in batches for speed + memory efficiency."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.append(emb)
    return np.vstack(all_embeddings).astype("float32")

def search_external_index(index, meta, model, query, k=TOP_K_EXTERNAL):
    """Search the external FAISS index."""
    q_emb = embed_texts([query], model)
    scores, indices = index.search(q_emb, k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx >= 0 and idx < len(meta["ids"]):  # Valid index check
            case_id = meta["ids"][idx]
            case_text = meta["texts"][idx]
            results.append({
                "id": case_id, 
                "score": float(score), 
                "text": case_text,
                "source": "external"
            })
    
    return results

def search_document_chunks(chunks: List[str], model, query: str, k=TOP_K_DOCUMENT):
    """Find most relevant chunks from the uploaded document."""
    if not chunks or not any(chunk.strip() for chunk in chunks):
        return []
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk.strip()]
    if not valid_chunks:
        return []
    
    # Create embeddings for document chunks
    chunk_embeddings = embed_texts(valid_chunks, model)
    query_embedding = embed_texts([query], model)
    
    # Calculate similarities
    similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
    
    # Get top-k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            "id": f"doc_chunk_{idx}",
            "score": float(similarities[idx]),
            "text": valid_chunks[idx],
            "source": "document"
        })
    
    return results

# ----------------------------- 
# Legal Document Verification Functions (Integrated from Reference Code)
# ----------------------------- 
def run_document_verifier_rules(text: str):
    """Rule-based checker using your proven legal document validation logic."""
    checks = {
        "signatures": bool(re.search(r"signature|signed by", text, re.IGNORECASE)),
        "dates": bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)) or bool(re.search(r"\b\d{4}\b", text)),
        "parties": bool(re.search(r"between\s+\w+", text, re.IGNORECASE)),
        "jurisdiction": bool(re.search(r"jurisdiction|court|state of|high court|supreme court", text, re.IGNORECASE)),
    }
    score = sum(checks.values())
    return checks, score

def run_document_verifier(document_text: str, index, meta, model, top_k=TOP_K_VERIFICATION):
    """Main document verifier integrated from your reference code."""
    # Rule-based checking using your proven logic
    rules, sufficiency_score = run_document_verifier_rules(document_text)

    # Chunk the document for semantic analysis
    chunks = chunk_text(document_text)
    chunk_results = []

    # Only do semantic analysis if we have valid chunks and text
    if document_text.strip():
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            try:
                # Get embedding for this chunk (adapted to work with your model)
                chunk_emb = embed_texts([chunk], model)
                
                # Search in FAISS index (using D, I format from your reference)
                D, I = index.search(chunk_emb, top_k)
                similar_cases = []
                
                for idx, score in zip(I[0], D[0]):
                    if 0 <= idx < len(meta["ids"]):  # safety check
                        doc_meta = {
                            "id": meta["ids"][idx],
                            "summary": meta["texts"][idx][:300] + "...",  # preview
                            "similarity_score": float(score)
                        }
                        similar_cases.append(doc_meta)

            except Exception as e:
                similar_cases = []
            
            # ✅ this must be INSIDE the loop (as per your reference)
            chunk_results.append({
                "chunk_index": i,
                "chunk_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                "similar_cases": similar_cases
            })
    
    return {
        "sufficiency_score": sufficiency_score,
        "rule_checklist": rules,
        "chunks": chunk_results
    }

def print_verification_results(verification_results, document_text):
    """Simplified verification results display focusing on core legal elements."""
    print("\n" + "="*80)
    print("📋 LEGAL DOCUMENT VERIFICATION REPORT")
    print("="*80)
    
    # Document preview (first 200 chars)
    if document_text.strip():
        preview = document_text[:200] + "..." if len(document_text) > 200 else document_text
        print(f"\n📄 Document Preview: {preview}")
    else:
        print(f"\n📄 Document Preview: [EMPTY DOCUMENT - NO TEXT EXTRACTED]")
    print(f"📊 Document Length: {len(document_text)} characters")
    
    # Core legal document requirements (simplified as per your reference)
    total_checks = len(verification_results['rule_checklist'])
    print(f"\n✅ Legal Document Sufficiency Score: {verification_results['sufficiency_score']}/{total_checks}")
    
    # Display all rule checks
    for check, passed in verification_results['rule_checklist'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        check_name = check.replace('_', ' ').title()
        print(f"   - {check_name}: {status}")
    
    # Simplified overall assessment based on 4 core checks
    core_score = verification_results['sufficiency_score']
    
    if core_score == 4:
        print("\n📋 Overall Assessment: ✅ Document appears to be a complete legal document")
    elif core_score >= 3:
        print("\n📋 Overall Assessment: ⚠️ Document appears mostly complete but may be missing key elements")
    elif core_score >= 2:
        print("\n📋 Overall Assessment: ⚠️ Document may be incomplete or informal")
    else:
        print("\n📋 Overall Assessment: ❌ Document appears incomplete or not a formal legal document")
    
    # Semantic similarity results (only show if we have meaningful chunks)
    meaningful_chunks = [chunk for chunk in verification_results['chunks'] 
                        if chunk['similar_cases'] and len(chunk['chunk_preview'].strip()) > 10]
    
    if meaningful_chunks:
        print(f"\n🔍 Legal Precedent Analysis ({len(meaningful_chunks)} meaningful chunks):")
        for chunk in meaningful_chunks:
            if chunk['similar_cases']:
                print(f"\n   Chunk: '{chunk['chunk_preview']}'")
                print(f"     Top match: {chunk['similar_cases'][0]['id']} (score: {chunk['similar_cases'][0]['similarity_score']:.4f})")
    
    print("="*80)

# ----------------------------- 
# Document Briefing Functions
# ----------------------------- 
def clean_empty(d):
    """Remove empty values from nested dictionary/list structure."""
    if isinstance(d, dict):
        return {k: clean_empty(v) for k, v in d.items() if v not in [None, [], {}, ""]}
    elif isinstance(d, list):
        return [clean_empty(v) for v in d if v not in [None, [], {}, ""]]
    else:
        return d

def extract_json(text):
    """Extract JSON object from string, removing backticks if present."""
    # Remove Markdown json block if exists
    text = re.sub(r"```json\s*|\s*```", "", text, flags=re.IGNORECASE).strip()
    
    # Try multiple extraction strategies
    patterns = [
        r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Nested JSON
        r'(\{.*\})',  # Simple JSON capture
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                continue
    
    # If no valid JSON found, try parsing the whole text
    try:
        return json.loads(text)
    except:
        return None

def run_brief_mode(query: str, document_text: str, model="gemini-1.5-flash"):
    """Generate structured document briefing with metadata extraction."""
    if not document_text:
        return None
    
    # # Use the global GEMINI_MODEL if no model specified
    # if model is None:
    #     model = GEMINI_MODEL

    prompt = f"""You are a Legal Document Analysis System. Return ONLY valid JSON.

REQUIRED FORMAT:
{{
  "metadata": {{
    "document_type": "string (e.g., License, Agreement, Order, Certificate)",
    "issuer": "string (authority/organization that issued document)",
    "date": "string (date found in document)",
    "recipient": "string (person/organization receiving document)",
    "reference_numbers": ["list", "of", "reference/file numbers"]
  }},
  "summary": "Brief 2-3 sentence plain-language summary of what this document does",
  "key_sections": [
    {{"section": "section name", "content": "key content from that section"}}
  ],
  "obligations": ["list of key obligations or requirements"],
  "important_dates": ["list of important dates with context"],
  "legal_implications": ["list of key legal implications or consequences"]
}}

Document Text: {document_text[:2500]}...

User Query: {query}

Return only the JSON structure above. Do not include any explanatory text before or after the JSON."""

    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(prompt)

    # Try parsing JSON from the model output
    result_json = extract_json(response.text)

    if not result_json:
        # Fallback if parsing fails
        result_json = {
            "metadata": {"document_type": "Unknown Document"},
            "summary": response.text.strip()[:500] + "..." if len(response.text.strip()) > 500 else response.text.strip()
        }

    result_json = clean_empty(result_json)
    return result_json

# ----------------------------- 
# Gemini Helper
# ----------------------------- 
def ask_gemini_enhanced(query: str, external_results: List[Dict], document_results: List[Dict]):
    """Ask Gemini with enhanced context from both external and document sources."""
    
    # Prepare context from external sources
    external_context = ""
    if external_results:
        external_context = "RELEVANT LEGAL CASES/PRECEDENTS:\n"
        for i, r in enumerate(external_results, 1):
            external_context += f"[External-{i}] {r['text'][:800]}...\n\n"
    
    # Prepare context from document
    document_context = ""
    if document_results:
        document_context = "RELEVANT SECTIONS FROM UPLOADED DOCUMENT:\n"
        for i, r in enumerate(document_results, 1):
            document_context += f"[Document-{i}] {r['text'][:800]}...\n\n"
    
    # Combined context
    full_context = f"{external_context}{document_context}".strip()
    
    if not full_context:
        return "❌ No relevant information found in either the knowledge base or uploaded document. The document may be empty or corrupted."
    
    prompt = f"""You are a legal research assistant with access to legal precedents and case law.

{full_context}

User Query: {query}

Instructions:
1. Answer the query based ONLY on the provided context above
2. Cite your sources using the reference IDs (e.g., [External-1], [Document-2])
3. If the uploaded document contains relevant information, prioritize it in your answer
4. If external legal cases provide precedent or support, reference them as well
5. Be clear, concise, and legally accurate
6. If the context doesn't contain enough information to answer the query, state this clearly

Give a comprehensive legal-style response with proper citations."""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

def ask_gemini_simple(document_text: str, mode="summary"):
    """Simple Gemini call for document summary when no questions are provided."""
    if not document_text.strip():
        return "❌ Cannot provide summary - the document appears to be empty or text extraction failed."
    
    if mode == "summary":
        prompt = f"""You are a legal research assistant. The user has provided the following document:

{document_text[:3000]}...

Summarize this document in 3-4 simple sentences for a non-legal audience. 
Clearly state the main outcome, effective dates, and authority."""
    
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

# ----------------------------- 
# Main Function
# ----------------------------- 
def main():
    print(f"🔄 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ GPU not found, using CPU")
    
    # Load external FAISS index
    try:
        external_index, external_meta = load_index()
        external_available = True
    except Exception as e:
        print(f"⚠️ Could not load external FAISS index: {e}")
        external_available = False
    
    # Check if document is provided
    if os.path.exists(DOCUMENT_PATH):
        print(f"📄 Found {DOCUMENT_PATH}, entering Enhanced Document Mode...")
        
        # Load and chunk the document
        print("📄 Loading document...")
        document_text = load_document(DOCUMENT_PATH)
        print(f"📄 Document loaded: {len(document_text)} characters")
        
        # NEW: Run document verification
        if external_available:
            print("🔍 Running document verification...")
            verification_results = run_document_verifier(document_text, external_index, external_meta, model)
            print_verification_results(verification_results, document_text)
        else:
            print("⚠️ Skipping document verification - external index not available")
        
        print("✂️ Chunking document...")
        document_chunks = chunk_text(document_text)
        print(f"✂️ Created {len(document_chunks)} chunks")
        
        # Check for questions
        if os.path.exists(QUESTION_PATH):
            print(f"❓ Found {QUESTION_PATH}, processing questions...")
            with open(QUESTION_PATH, "r", encoding="utf-8") as f:
                questions = [q.strip() for q in f if q.strip()]
            
            for i, question in enumerate(questions, 1):
                print(f"\n🔎 Question {i}/{len(questions)}: {question}")
                
                # Search external index
                external_results = []
                if external_available:
                    external_results = search_external_index(external_index, external_meta, model, question)
                    print(f"🔍 Found {len(external_results)} relevant external cases")
                
                # Search document chunks
                document_results = search_document_chunks(document_chunks, model, question)
                print(f"📄 Found {len(document_results)} relevant document sections")
                
                # Get enhanced answer
                answer = ask_gemini_enhanced(question, external_results, document_results)
                print(f"\n🤖 Enhanced Answer:\n{answer}")
                print("-" * 80)
        
        else:
            print("📝 No questions found. Running document briefing and summary...")
            
            # Generate structured brief
            brief_result = run_brief_mode("Brief this legal document with key details", document_text, GEMINI_MODEL)
            if brief_result:
                print(f"\n📋 STRUCTURED DOCUMENT BRIEF:")
                print("="*60)
                print(json.dumps(brief_result, indent=2))
                print("="*60)
            
            # Also provide simple summary as backup
            summary = ask_gemini_simple(document_text, mode="summary")
            print(f"\n🤖 Document Summary:\n{summary}")
    
    else:
        if not external_available:
            print("❌ No document provided and external index unavailable. Exiting.")
            return
            
        print("🔍 Entering Interactive Search Mode...")
        while True:
            query = input("\n🔎 Enter your query (or 'exit'): ").strip()
            if query.lower() == "exit":
                break
            if not query:
                continue
            
            external_results = search_external_index(external_index, external_meta, model, query)
            answer = ask_gemini_enhanced(query, external_results, [])
            print(f"\n🤖 Answer:\n{answer}")

if __name__ == "__main__":

    main()
