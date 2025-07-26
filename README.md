# PDF Search Tool

A simple Flask web application for searching PDF documents using semantic search.

## Features

- Upload multiple PDF files
- Two extraction methods: Normal text extraction and OCR
- Semantic search using sentence transformers
- View extracted text chunks and raw text
- Web interface for easy interaction

## Technology Stack

- **Flask** - Web framework
- **SentenceTransformers** - Semantic search embeddings
- **PyMuPDF** - PDF text extraction
- **Tesseract OCR** - Optical character recognition
- **spaCy** - Natural language processing
- **langdetect** - Language detection

## Prerequisites

- Python 3.7+
- Tesseract OCR installed on your system

## Installation

1. **Clone or download this repository**

2. **Install Tesseract OCR:**
   - Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`

3. **Create virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

5. **Update Tesseract path in `app.py` if needed:**
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

6. **Run the application:**
   ```bash
   python app.py
   ```

7. **Open browser:** Go to `http://localhost:5000`

## Usage

1. **Upload PDFs**: Select one or more PDF files and choose extraction method
2. **Search**: Type your query in the search box for real-time results
3. **View Results**: See matching snippets with page numbers and relevance scores
4. **View Chunks/Raw Text**: Optionally view processed text chunks or complete extracted text

## File Structure

```
pdfsearch/
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Web interface
├── uploads/            # Uploaded PDF files
├── requirements.txt    # Python dependencies
└── README.md          # This file
```
