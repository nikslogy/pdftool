import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import fitz
from sentence_transformers import SentenceTransformer, util
import logging
import pytesseract
from PIL import Image
import io
import spacy
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = SentenceTransformer('all-MiniLM-L6-v2')

logging.basicConfig(level=logging.INFO)

pdf_chunks = []
raw_texts = [] 

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

nlp = spacy.load('en_core_web_sm')

@app.route('/')
def index():
    return render_template('index.html')


def clean_text_with_nlp(text):
    try:
        if len(text.strip()) > 50:  
            try:
                if detect(text) != 'en':
                    return ""
            except LangDetectException:
                return ""

        doc = nlp(text)
        
        cleaned_text = []
        for token in doc:
            if token.is_alpha and not token.is_stop:
                cleaned_text.append(token.text)
        
        return " ".join(cleaned_text)
    except Exception as e:
        logging.error(f"Error in NLP processing: {str(e)}")
        return ""


def extract_text_chunks(filepath, chunk_size=5000, method='normal'):
    text_chunks = []
    try:
        temp_filepath = os.path.join(UPLOAD_FOLDER, f"temp_{os.path.basename(filepath)}")
        with open(filepath, 'rb') as src, open(temp_filepath, 'wb') as dst:
            dst.write(src.read())

        with fitz.open(temp_filepath) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                if method == 'ocr':
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text = pytesseract.image_to_string(img, lang='eng+mar+hin')
                    except Exception as e:
                        logging.error(f"OCR error on page {page_num + 1}: {e}")
                        text = ""
                else:
                    raw_text = page.get_text()
                    text = clean_text_with_nlp(raw_text)
                
                if text.strip():
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i+chunk_size]
                        if len(chunk.strip()) > 50:
                            text_chunks.append({
                                'text': chunk,
                                'page': page_num + 1
                            })
        
        os.remove(temp_filepath)
    except Exception as e:
        logging.error(f"Error processing {filepath}: {str(e)}")
        raise
    return text_chunks

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_files' not in request.files:
        return "No file part", 400

    extraction_method = request.form.get('extraction_method', 'normal')
    files = request.files.getlist('pdf_files')
    results = []
    
    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            try:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                chunks = extract_text_chunks(filepath, method=extraction_method)
                
                # Store raw text
                with fitz.open(filepath) as doc:
                    raw_text = ""
                    for page in doc:
                        raw_text += page.get_text()
                    raw_texts.append({
                        'filename': file.filename,
                        'text': raw_text
                    })

                if not chunks:
                    results.append(f"Warning: No text extracted from {file.filename}")
                    continue

                for chunk_data in chunks:
                    embedding = model.encode(chunk_data['text'], convert_to_tensor=True)
                    pdf_chunks.append({
                        'filename': file.filename,
                        'chunk': chunk_data['text'],
                        'page': chunk_data['page'],
                        'embedding': embedding
                    })
                results.append(f"Successfully processed {file.filename}")
            except Exception as e:
                results.append(f"Error processing {file.filename}: {str(e)}")
                logging.error(f"Error processing {file.filename}: {str(e)}")
                continue
            
    return jsonify({"status": "complete", "messages": results})


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify(results=[])

    query_embedding = model.encode(query, convert_to_tensor=True)
    
    matches = []
    for chunk_info in pdf_chunks:
        cos_score = util.cos_sim(query_embedding, chunk_info['embedding']).item()
        if cos_score >= 0.1:
            matches.append({
                'filename': chunk_info['filename'],
                'snippet': chunk_info['chunk'][:200].replace('\n', ' '),
                'page': chunk_info['page'],
                'score': round(cos_score, 3)
            })
    
    grouped_results = {}
    for match in sorted(matches, key=lambda x: x['score'], reverse=True):
        filename = match['filename']
        if filename not in grouped_results:
            grouped_results[filename] = []
        if len(grouped_results[filename]) < 3:
            grouped_results[filename].append(match)

    results = []
    for filename, matches in grouped_results.items():
        results.append({
            'filename': filename,
            'matches': matches,
            'score': max(m['score'] for m in matches)
        })
    
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return jsonify(results=results)


@app.route('/get-chunks')
def get_chunks():
    chunks_data = [{'filename': c['filename'], 'text': c['chunk']} for c in pdf_chunks]
    return jsonify(chunks=chunks_data)

@app.route('/get-raw-text')
def get_raw_text():
    return jsonify(texts=raw_texts)


if __name__ == '__main__':
    app.run(debug=True)
