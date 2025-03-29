#curriculum-preprocessing.py
"""
Turkish Mathematics Curriculum NLP Analysis - PDF Processor
==========================================================
This script processes Turkish PDF documents, extracts text, and performs
NLP analysis on the content. It builds on the existing preprocessing framework
but adds robust PDF handling capabilities.
"""

import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import io
import time

# For NLP processing
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy

# For PDF processing
import PyPDF2
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Try to load Turkish language model for spaCy
try:
    nlp = spacy.load("tr_core_news_trf")
    print("Turkish spaCy model loaded successfully")
except OSError:
    print("Turkish spaCy model not found. Using English model instead.")
    print("Install Turkish model with: pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_trf/resolve/main/tr_core_news_trf-1.0-py3-none-any.whl")
    #stop script
    exit(1)
    

# Define paths
DATA_DIR = "data"
PROCESSED_DIR = "processed_data"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Function to extract text from PDF using PyPDF2 (simpler but less precise)
def extract_text_pypdf2(pdf_path):
    """Extract text from PDF using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Extracting text from {num_pages} pages...")
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
                
        return text
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {e}")
        return ""

# Function to extract text from PDF using PDFMiner (more precise for complex layouts)
def extract_text_pdfminer(pdf_path, encoding='utf-8'):
    """Extract text from PDF using PDFMiner with Turkish language support."""
    try:
        output_string = io.StringIO()
        laparams = LAParams(
            line_margin=0.5,  # Adjust for Turkish text layout
            char_margin=2.0,  # More liberal character margins
            all_texts=True    # Extract all text elements
        )
        
        with open(pdf_path, 'rb') as file:
            resource_manager = PDFResourceManager()
            device = TextConverter(resource_manager, output_string, laparams=laparams, codec=encoding)
            interpreter = PDFPageInterpreter(resource_manager, device)
            
            for page in PDFPage.get_pages(file, check_extractable=True):
                interpreter.process_page(page)
                
            text = output_string.getvalue()
            
        return text
    except Exception as e:
        print(f"Error extracting text with PDFMiner: {e}")
        return ""

# Function to extract text from PDF with Turkish language support
def extract_text_from_pdf(pdf_path):
    """Try multiple methods to extract text from PDF, optimized for Turkish language."""
    # First try PDFMiner (better for complex layouts and non-Latin characters)
    text = extract_text_pdfminer(pdf_path)
    
    # If PDFMiner didn't extract much text, try PyPDF2 as fallback
    if len(text.strip()) < 100:
        print("PDFMiner extracted very little text. Trying PyPDF2 as fallback...")
        text = extract_text_pypdf2(pdf_path)
    
    # Check if text extraction was successful
    if len(text.strip()) < 100:
        print("Warning: Very little text extracted from PDF. The PDF might be scanned or have restricted permissions.")
    
    return text

# Text cleaning functions (from original script)
def clean_text(text):
    """Basic cleaning of the text."""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that aren't relevant but keep mathematical symbols
    text = re.sub(r'[^\w\s\.\,\;\:\(\)\[\]\{\}\+\-\*\/\=\<\>\'\"\!\?\&\%\$\#\@\^\~\`\|\\çğıöşüÇĞİÖŞÜ]', ' ', text)
    
    # Normalize whitespace around punctuation
    text = re.sub(r'\s*([\.;:,!?])\s*', r'\1 ', text)
    
    return text.strip()

def normalize_mathematical_notation(text):
    """Standardize mathematical notation across documents with expanded support for 2024 curriculum."""
    # Basic function notation
    text = re.sub(r'f\s*\(\s*x\s*\)', 'f(x)', text)
    
    # Enhanced function notation for 2024 curriculum
    text = re.sub(r'f\s*\(x\)\s*=\s*x\^2', 'f(x) = x²', text)
    text = re.sub(r'f\s*\(x\)\s*=\s*\\sqrt\s*x', 'f(x) = √x', text)
    text = re.sub(r'f\s*\(x\)\s*=\s*\\frac\s*1\s*x', 'f(x) = 1/x', text)
    
    # Standardize derivatives notation
    text = re.sub(r'f\s*\'\s*\(\s*x\s*\)', 'f\'(x)', text)
    text = re.sub(r'\\frac{d}{dx}', 'd/dx', text)
    
    # Standardize set notation
    text = re.sub(r'[\{]([^\}]+)[\}]', r'{\\1}', text)
    
    # Standardize mathematical operators
    text = re.sub(r'\s*\+\s*', ' + ', text)
    text = re.sub(r'\s*\-\s*', ' - ', text)
    text = re.sub(r'\s*\*\s*', ' * ', text)
    text = re.sub(r'\s*\/\s*', ' / ', text)
    text = re.sub(r'\s*\=\s*', ' = ', text)
    
    # Standardize mathematical symbols for 2024 curriculum
    text = re.sub(r'\\mathbb{R}', 'ℝ', text)
    text = re.sub(r'\\mathbb{N}', 'ℕ', text)
    text = re.sub(r'\\mathbb{Z}', 'ℤ', text)
    text = re.sub(r'\\mathbb{Q}', 'ℚ', text)
    
    return text

# Enhanced function to segment curriculum by sections - handles Turkish patterns
def segment_curriculum(text):
    """
    Segment curriculum into logical sections based on patterns.
    Enhanced to better handle both 2018 and 2024 curriculum formats.
    """
    sections = {}
    
    # Detect curriculum version
    curriculum_version = detect_curriculum_version(text)
    
    # Common patterns in Turkish mathematics curriculum
    if curriculum_version == "2018":
        patterns = [
            # Standard numeric patterns (e.g., "10.2.1.1. Fonksiyonlarla ilgili...")
            r'(\d+\.\d+\.\d+\.\d+\.\s+[^\n]+)',
            
            # Unit/chapter headings (e.g., "ÜNİTE 5: İstatistik ve Olasılık")
            r'(ÜNİTE \d+\:?\s+[^\n]+)',
            
            # Section headings with Turkish characters
            r'(BÖLÜM \d+\:?\s+[^\n]+)'
        ]
    else:  # 2024 or unknown
        # Use separate patterns for each grade to avoid regex range issues
        patterns = []
        
        # Add patterns for each grade separately
        for grade in ["H", "9", "10", "11", "12"]:
            pattern = r'(MAT\.' + grade + r'\.\d+\.\d+\.\s+[^\n]+)'
            patterns.append(pattern)
        
        # Add theme headings pattern
        patterns.append(r'(\d+\.\s+TEMA\:?\s+[^\n]+)')
    
    # Try all patterns and choose the one with most matches
    best_pattern = None
    most_matches = 0
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, text)
            if len(matches) > most_matches:
                most_matches = len(matches)
                best_pattern = pattern
        except re.error as e:
            print(f"Regex error with pattern {pattern}: {e}")
            continue
    
    if not best_pattern or most_matches == 0:
        print("Warning: No curriculum sections detected. Using whole document as a single section.")
        sections['full_document'] = text
        return sections, "unknown"
    
    # Determine the type based on the best pattern
    if curriculum_version == "2024":
        section_type = "2024-style"
    elif "ÜNİTE" in best_pattern or "BÖLÜM" in best_pattern:
        section_type = "2018-style"
    else:
        section_type = "numeric-based"
    
    # Split by the chosen pattern
    try:
        section_splits = re.split(best_pattern, text)
        
        # First element is text before any section
        if section_splits[0].strip():
            sections['intro'] = section_splits[0].strip()
        
        # Process the rest of the sections
        for i in range(1, len(section_splits), 2):
            if i < len(section_splits) - 1:
                section_title = section_splits[i].strip()
                section_content = section_splits[i+1].strip()
                sections[section_title] = section_content
    except re.error as e:
        print(f"Error splitting sections with pattern {best_pattern}: {e}")
        sections['full_document'] = text
        return sections, "unknown"
    
    return sections, section_type

# Enhanced function to detect curriculum version based on patterns
def detect_curriculum_version(text):
    """Detect which curriculum version the document represents."""
    # 2024 curriculum specific patterns - avoiding character ranges
    patterns_2024 = [
        r'MAT\.H\.\d+',        # MAT.H format
        r'MAT\.9\.\d+',        # MAT.9 format
        r'MAT\.10\.\d+',       # MAT.10 format 
        r'MAT\.11\.\d+',       # MAT.11 format
        r'MAT\.12\.\d+',       # MAT.12 format
        r'Türkiye Yüzyılı Maarif',
        r'Beceriler Arası İlişkiler',
        r'Alan Becerileri'
    ]
    
    # 2018 curriculum specific patterns
    patterns_2018 = [
        r'\d+\.\d+\.\d+\.\d+\.',
        r'Öğretim Programlarının Perspektifi',
        r'Millî Eğitim Temel Kanunu'
    ]
    
    # Count matches for each version
    count_2024 = 0
    count_2018 = 0
    
    for pattern in patterns_2024:
        if re.search(pattern, text):
            count_2024 += 1
    
    for pattern in patterns_2018:
        if re.search(pattern, text):
            count_2018 += 1
    
    if count_2024 > count_2018:
        return "2024"
    elif count_2018 > count_2024:
        return "2018"
    else:
        return "unknown"    

def process_text(text):
    """
    Process text with NLP tools.
    Returns various tokenized forms and linguistic features.
    Enhanced for Turkish language support and mathematical notation.
    """
    # Clean the text
    cleaned_text = clean_text(text)
    normalized_text = normalize_mathematical_notation(cleaned_text)
    
    # Remove mathematical symbols before lemmatization
    # This prevents lemmatization failures on objectives with mathematical notation
    math_symbols_pattern = r'[±∈∉∋∌∩∪⊂⊃⊆⊇⊄⊅∧∨¬→↔∀∃∄∑∏∫∮≤≥≠≈←↑↓↔⇒⇔√∛∜∝∞∟∠∡∢∴∵∷∼⊕⊗⊥∥∦∣∤π°′″‴⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎(){}[]<>+\-*/=]'
    math_cleaned_text = re.sub(math_symbols_pattern, ' ', normalized_text)
    # Remove multiple spaces created by symbol removal
    math_cleaned_text = re.sub(r'\s+', ' ', math_cleaned_text).strip()
    
    # Basic tokenization (using math_cleaned_text for better results)
    try:
        sentences = sent_tokenize(math_cleaned_text, language='turkish')
    except ValueError:
        sentences = sent_tokenize(math_cleaned_text)
    
    try:
        words = word_tokenize(math_cleaned_text, language='turkish')
    except ValueError:
        words = word_tokenize(math_cleaned_text)
    
    # Process with spaCy for more advanced linguistic features
    doc = nlp(math_cleaned_text)
    
    # Extract linguistic features
    tokens = []
    lemmas = []
    for token in doc:
        token_info = {
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_,
            'is_stop': token.is_stop,
            'is_punctuation': token.is_punct
        }
        tokens.append(token_info)
        # Add lemmas (or original text if lemmatization failed)
        if not token.is_punct and not token.is_space:
            # If lemmatization produces an empty string or the same as the token,
            # use the original token text
            if not token.lemma_ or token.lemma_ == token.text:
                lemmas.append(token.text)
            else:
                lemmas.append(token.lemma_)
            
    # Join lemmas into a single string
    lemmatized_text = ' '.join(lemmas)
    
    # Handle case where lemmatization produced an empty string
    if not lemmatized_text.strip() and math_cleaned_text.strip():
        lemmatized_text = math_cleaned_text.strip()
        
    return {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'normalized_text': normalized_text,
        'lemmatized_text': lemmatized_text,
        'sentences': sentences,
        'words': words,
        'spacy_tokens': tokens
    }

# Extract learning objectives - enhanced for Turkish curriculum
def extract_learning_objectives(sections, section_type="numeric-based"):
    """
    Extract learning objectives from curriculum sections.
    Different patterns based on curriculum type.
    Enhanced for Turkish formatting patterns and mathematical notation.
    """
    objectives = []
    
    for section_title, content in sections.items():
        # Skip intro section
        if section_title == 'intro' or section_title == 'full_document':
            continue
        
        # Different patterns for objectives based on detected format
        if section_type == "2024-style":
            # Pattern like " a) Rasyonel referans fonksiyonun..."
            pattern = r'([a-z]\))\s+([^\n]+)'
        elif section_type == "chapter-based":
            # Pattern for numbered items within chapters
            pattern = r'(\d+\.\d+\.)\s+([^\n]+)'
        else:  # Default: numeric-based
            # Pattern like "a) Fonksiyon kavramı açıklanır."
            pattern = r'([a-z]\))\s+([^\n]+)'
        
        # Additional Turkish-specific patterns to try
        turkish_patterns = [
            # Bullet points with Turkish characters
            r'(•)\s+([^\n]+)',
            
            # Numbered list items (common in Turkish educational documents)
            r'(\d+\))\s+([^\n]+)'
        ]
        
        # Mathematical notation specific patterns
        math_patterns = [
            # Pattern that may start with mathematical symbols
            r'([±∈∉\(\)\[\]\{\}])\s+([^\n]+)',
            
            # Pattern with math notation embedded in item marker
            r'([a-z][\)\.])\s*([±∈∉\(\)\[\]\{\}][^\n]+)'
        ]
        
        # Try main pattern first
        matches = re.findall(pattern, content)
        
        # If few or no matches with primary pattern, try Turkish-specific patterns
        if len(matches) < 3:
            for turkish_pattern in turkish_patterns:
                turkish_matches = re.findall(turkish_pattern, content)
                if len(turkish_matches) > len(matches):
                    matches = turkish_matches
                    break
        
        # Try math patterns if still few matches
        if len(matches) < 3:
            for math_pattern in math_patterns:
                math_matches = re.findall(math_pattern, content)
                if len(math_matches) > len(matches):
                    matches = math_matches
                    break
        
        # Add all found objectives
        for item_num, item_text in matches:
            objectives.append({
                'section': section_title,
                'item': item_num.strip(),
                'text': item_text.strip(),
                'curriculum_type': section_type,
                'contains_math': bool(re.search(r'[±∈∉∋∌∩∪⊂⊃⊆⊇⊄⊅∧∨¬→↔∀∃∄∑∏∫∮≤≥≠≈←↑↓↔⇒⇔√∛∜∝∞∟∠∡∢∴∵∷∼⊕⊗⊥∥∦∣∤π°′″‴⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎]', item_text))
            })
    
    return objectives


# Process a curriculum PDF
def process_curriculum_pdf(pdf_path, name="curriculum"):
    """Process a curriculum PDF file and return structured data."""
    print(f"Processing {name} PDF: {pdf_path}...")
    
    # Check if the extracted text file already exists
    text_output_path = os.path.join(PROCESSED_DIR, f"{name}_extracted_text.txt")
    
    # If the text file exists, read it instead of extracting again
    if os.path.exists(text_output_path):
        print(f"Found existing extracted text for {name}. Loading from file...")
        with open(text_output_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text)} characters of text")
    else:
        # Extract text from PDF (only if we don't have it already)
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"Failed to extract text from {pdf_path}")
            return None
        
        print(f"Successfully extracted {len(text)} characters of text")
        
        # Save raw extracted text for inspection
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    # Segment by sections
    sections, detected_type = segment_curriculum(text)
    print(f"Detected curriculum type: {detected_type}")
    print(f"Found {len(sections)} sections")
    
    # Process each section
    processed_sections = {}
    for section_title, content in sections.items():
        processed_sections[section_title] = process_text(content)
    
    # Extract learning objectives
    objectives = extract_learning_objectives(sections, detected_type)
    print(f"Extracted {len(objectives)} learning objectives")
    
    # Process each learning objective
    processed_objectives = []
    for obj in objectives:
        obj_text = obj['text']
        processed_obj = process_text(obj_text)
        processed_obj['section'] = obj['section']
        processed_obj['item'] = obj['item']
        processed_obj['curriculum_type'] = obj['curriculum_type']
        processed_objectives.append(processed_obj)
    
    return {
        'name': name,
        'full_text': text,
        'processed_full_text': process_text(text),
        'sections': processed_sections,
        'objectives': objectives,
        'processed_objectives': processed_objectives,
        'detected_type': detected_type
    }
  
def analyze_curriculum_topics_unsupervised(processed_data):
    """
    Analyze curriculum topics using unsupervised methods instead of predefined categories.
    This replaces the previous AI relevance tagging with a data-driven approach.
    """
    # Group objectives by curriculum version
    objectives_by_year = {'2018': [], '2024': []}
    metadata_by_year = {'2018': [], '2024': []}
    
    # Extract texts from processed_objectives
    for name, curriculum in processed_data.items():
        if not curriculum or 'processed_objectives' not in curriculum:
            continue
            
        # Determine curriculum year
        if '2018' in curriculum.get('detected_type', ''):
            year = '2018'
        elif '2024' in curriculum.get('detected_type', ''):
            year = '2024'
        else:
            # Try to guess from name if type not detected
            if '2018' in name:
                year = '2018'
            elif '2024' in name:
                year = '2024'
            else:
                print(f"Warning: Could not determine year for {name}. Skipping.")
                continue
        
        # Collect objectives
        for obj in curriculum['processed_objectives']:
            # Use lemmatized text when available
            if 'lemmatized_text' in obj and obj['lemmatized_text'].strip():
                text = obj['lemmatized_text']
            else:
                text = obj.get('cleaned_text', '')
            
            if text.strip():
                objectives_by_year[year].append(text)
                metadata_by_year[year].append({
                    'section': obj.get('section', ''),
                    'item': obj.get('item', ''),
                    'original_text': obj.get('original_text', '')
                })
    
    # Add this information to the processed data
    for name, curriculum in processed_data.items():
        if not curriculum:
            continue
            
        # Determine curriculum year
        if '2018' in curriculum.get('detected_type', ''):
            year = '2018'
        elif '2024' in curriculum.get('detected_type', ''):
            year = '2024'
        else:
            continue  # Skip if year not determined
        
        # Add metadata to track which objectives belong to which curriculum
        curriculum['year'] = year
        
        # Mark processed data as ready for topic modeling
        curriculum['ready_for_topic_modeling'] = True
    
    # Return the collected objectives and metadata for topic modeling
    return processed_data, objectives_by_year, metadata_by_year


def repair_lemmatization(processed_data):
    """Fix objectives with failed lemmatization by keeping original words when lemmatization fails."""
    repairs_made = 0
    
    for name, curriculum in processed_data.items():
        if not curriculum or 'processed_objectives' not in curriculum:
            continue
            
        for i, obj in enumerate(curriculum['processed_objectives']):
            # Fix missing or empty lemmatized text
            if 'lemmatized_text' not in obj or not obj['lemmatized_text'].strip():
                if 'cleaned_text' in obj and obj['cleaned_text']:
                    try:
                        # Clean math symbols
                        math_cleaned_text = re.sub(r'[±∈∉∋∌∩∪⊂⊃⊆⊇⊄⊅∧∨¬→↔∀∃∄∑∏∫∮≤≥≠≈←↑↓↔⇒⇔√∛∜∝∞∟∠∡∢∴∵∷∼⊕⊗⊥∥∦∣∤π°′″‴⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎(){}[]<>+\-*/=]', ' ', obj['cleaned_text'])
                        math_cleaned_text = re.sub(r'\s+', ' ', math_cleaned_text).strip()
                        
                        # Get words
                        words = word_tokenize(math_cleaned_text)
                        
                        # Process with spaCy
                        doc = nlp(math_cleaned_text)
                        
                        # Use original word if lemmatization fails
                        lemmas = []
                        for token in doc:
                            if not token.is_punct and not token.is_space:
                                if not token.lemma_ or token.lemma_ == token.text:
                                    lemmas.append(token.text)
                                else:
                                    lemmas.append(token.lemma_)
                        
                        obj['lemmatized_text'] = ' '.join(lemmas)
                        
                        # If still empty, use the tokenized words
                        if not obj['lemmatized_text'].strip():
                            obj['lemmatized_text'] = ' '.join(words)
                            
                        repairs_made += 1
                    except Exception as e:
                        # Use the cleaned text as fallback
                        print(f"Error during repair: {e}. Using cleaned text as fallback.")
                        obj['lemmatized_text'] = obj['cleaned_text']
                        repairs_made += 1
    
    print(f"Repaired {repairs_made} objectives with failed lemmatization")
    return processed_data


# Main function to run the preprocessing
def main():
    """Main function to execute the preprocessing pipeline."""
    
    # Check if processed data already exists
    processed_data_path = os.path.join(PROCESSED_DIR, 'processed_curriculum_data.pkl')
    if os.path.exists(processed_data_path):
        print(f"Found existing processed data at {processed_data_path}")
        user_input = input("Do you want to reprocess all data? (y/n): ")
        if user_input.lower() != 'y':
            print("Loading existing processed data...")
            with open(processed_data_path, 'rb') as f:
                processed_data = pickle.load(f)
            print("Existing data loaded. Skipping PDF processing.")
            
            # Repair any missing lemmatization
            processed_data = repair_lemmatization(processed_data)
            
             # Use unsupervised topic analysis instead of AI relevance tagging
            print("Preparing data for unsupervised topic analysis...")
            processed_data, objectives_by_year, metadata_by_year = analyze_curriculum_topics_unsupervised(processed_data)
            
            # Save processed data and extracted objectives for topic modeling
            print("Saving prepared data for topic modeling...")
            with open(processed_data_path, 'wb') as f:
                pickle.dump(processed_data, f)
                
            # Save objectives by year for topic modeling
            for year in ['2018', '2024']:
                if objectives_by_year[year]:
                    objectives_path = os.path.join(PROCESSED_DIR, f'objectives_{year}.pkl')
                    metadata_path = os.path.join(PROCESSED_DIR, f'metadata_{year}.pkl')
                    
                    with open(objectives_path, 'wb') as f:
                        pickle.dump(objectives_by_year[year], f)
                    
                    with open(metadata_path, 'wb') as f:
                        pickle.dump(metadata_by_year[year], f)
                    
                    print(f"Saved {len(objectives_by_year[year])} objectives for {year}")
            
            
            # Continue with saving other outputs
            save_analysis_outputs(processed_data)
            return
    
    # Get list of PDF files to process
    pdf_files = []
    for file in os.listdir(DATA_DIR):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(DATA_DIR, file))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}. Please add PDF files and run again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    processed_data = {}
    for i, pdf_path in enumerate(pdf_files):
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Processing PDF {i+1}/{len(pdf_files)}: {name}")
        
        # Check if extracted text already exists
        text_output_path = os.path.join(PROCESSED_DIR, f"{name}_extracted_text.txt")
        if os.path.exists(text_output_path):
            print(f"  Found existing extracted text for {name}. Skipping PDF extraction.")
            # Still process the extracted text
            with open(text_output_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create a minimal processed data structure with the text
            if name not in processed_data:
                processed_data[name] = {
                    'name': name,
                    'full_text': text
                }
            # Now process this text instead of extracting from PDF again
            processed_data[name] = process_curriculum_text(text, name)
        else:
            # Process PDF from scratch
            processed_data[name] = process_curriculum_pdf(pdf_path, name)
    
     # Replace AI relevance tagging with unsupervised topic analysis
    print("Preparing data for unsupervised topic analysis...")
    processed_data, objectives_by_year, metadata_by_year = analyze_curriculum_topics_unsupervised(processed_data)
    
    
    # Save processed data
    print("Saving processed data...")
    with open(processed_data_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    # Save objectives by year for topic modeling
    for year in ['2018', '2024']:
        if objectives_by_year[year]:
            objectives_path = os.path.join(PROCESSED_DIR, f'objectives_{year}.pkl')
            metadata_path = os.path.join(PROCESSED_DIR, f'metadata_{year}.pkl')
            
            with open(objectives_path, 'wb') as f:
                pickle.dump(objectives_by_year[year], f)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_by_year[year], f)
            
            print(f"Saved {len(objectives_by_year[year])} objectives for {year}")
    
    
    # Save other analysis outputs
    save_analysis_outputs(processed_data)

def process_curriculum_text(text, name="curriculum"):
    """Process curriculum text without re-extracting from PDF."""
    print(f"Processing existing text for {name}...")
    
    # Segment by sections
    print("  Segmenting curriculum...")
    sections, detected_type = segment_curriculum(text)
    print(f"  Detected curriculum type: {detected_type}")
    print(f"  Found {len(sections)} sections")
    
    # Process each section
    print("  Processing sections...")
    processed_sections = {}
    for i, (section_title, content) in enumerate(sections.items()):
        if i % 5 == 0:  # Progress update every 5 sections
            print(f"    Processing section {i+1}/{len(sections)}")
        processed_sections[section_title] = process_text(content)
    
    # Extract learning objectives
    print("  Extracting learning objectives...")
    objectives = extract_learning_objectives(sections, detected_type)
    print(f"  Extracted {len(objectives)} learning objectives")
    
    # Process each learning objective
    print("  Processing learning objectives...")
    processed_objectives = []
    for i, obj in enumerate(objectives):
        if i % 50 == 0:  # Progress update every 50 objectives
            print(f"    Processing objective {i+1}/{len(objectives)}")
        obj_text = obj['text']
        processed_obj = process_text(obj_text)
        
        # Add debug check
        if 'lemmatized_text' not in processed_obj or not processed_obj['lemmatized_text']:
            print(f"WARNING: Lemmatization failed for objective: {obj_text[:50]}...")
            
        processed_obj['section'] = obj['section']
        processed_obj['item'] = obj['item']
        processed_obj['curriculum_type'] = obj['curriculum_type']
        processed_objectives.append(processed_obj)
    
    return {
        'name': name,
        'full_text': text,
        'processed_full_text': process_text(text),
        'sections': processed_sections,
        'objectives': objectives,
        'processed_objectives': processed_objectives,
        'detected_type': detected_type
    }

def save_analysis_outputs(processed_data):
    """Save analysis outputs to various files."""
    # Extract all objectives and save in CSV format
    all_objectives = []
    for name, curriculum in processed_data.items():
        if not curriculum:
            continue
        
        for obj in curriculum.get('objectives', []):
            obj['curriculum_name'] = name
            all_objectives.append(obj)
    
    if all_objectives:
        print("Saving learning objectives to CSV...")
        objectives_df = pd.DataFrame(all_objectives)
        objectives_df.to_csv(os.path.join(PROCESSED_DIR, 'learning_objectives.csv'), index=False, encoding='utf-8')
    
    print(f"Processing complete. Data saved to {PROCESSED_DIR} directory.")
    print(f"Files created:")
    print(f"  - processed_curriculum_data.pkl: Full processed data for further analysis")
    print(f"  - learning_objectives.csv: Extracted learning objectives")
    print(f"  - objectives_2018.pkl and objectives_2024.pkl: Extracted objectives for topic modeling")
    print(f"  - metadata_2018.pkl and metadata_2024.pkl: Metadata for topic modeling")
    
    for name in processed_data.keys():
        if os.path.exists(os.path.join(PROCESSED_DIR, f"{name}_extracted_text.txt")):
            print(f"  - {name}_extracted_text.txt: Raw extracted text from PDF")
            

if __name__ == "__main__":
        main()