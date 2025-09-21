
# LegalAI-RAG: Advanced Legal Document Analysis System

A **RAG-powered legal document analysis system** with OCR, semantic search, document verification, and AI-assisted legal research.

---

## 🚀 Features

- **Multi-format Processing**: PDF, DOCX/DOC, TXT, images (JPG, PNG, BMP, TIFF)  
- **Advanced OCR**: Tesseract + EasyOCR fallback  
- **Semantic Search**: FAISS vector search with multilingual embeddings  
- **Legal Verification**: Rule-based & semantic checks, sufficiency scoring  
- **Structured Briefing**: Extracts key legal elements & metadata  
- **Interactive Q&A**: Context-aware answers with external knowledge  

---

## ⚙️ Requirements

- **Python** 3.8+  
- **RAM**: 8GB+ (16GB recommended)  
- **GPU**: Optional (CUDA support)  
- **Tesseract OCR**, **Poppler** for PDF processing  

---

## 🛠️ Installation

```bash
git clone <repo-url>
cd legalai-rag
pip install -r requirements.txt
```

**System dependencies**  

- Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin poppler-utils`  
- macOS: `brew install tesseract poppler`  
- Windows: Install Tesseract and add to PATH  

Set Gemini API key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

---

## 🚦 Quick Start

**CLI Usage**
```bash
python rag.py
# Use 'pan.pdf' and 'question.txt' for input
```

**Flask API**
```bash
python app.py
# Access at http://localhost:7860
```

**Example: Analyze Document**
```python
import requests

with open('legal_document.pdf', 'rb') as f:
    resp = requests.post('http://localhost:7860/api/upload', files={'file': f})
file_id = resp.json()['file_id']

resp = requests.post('http://localhost:7860/api/analyze', json={
    'question': 'Key obligations in the contract?',
    'file_id': file_id
})
print(resp.json()['response'])
```

---

## 🔍 Document Verification
- **Checks**: Signatures, Dates, Parties, Jurisdiction  
- **Scoring**:  
  - 4/4: Complete  
  - 3/4: Mostly complete  
  - 2/4: Incomplete  
  - 1/4 or less: Inadequate  

---

## 🐳 Docker Deployment
```dockerfile
FROM python:3.9-slim
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin poppler-utils
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python", "app.py"]
```

---

## 📄 License
MIT License – see LICENSE file

---

## 🙏 Acknowledgments
- Hugging Face, FAISS, Google Gemini, Tesseract, Flask  
