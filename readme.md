# Turkish Mathematics Curriculum NLP Analysis

This repository contains the code used in the research paper "Revealing Curricular Evolution Through Unsupervised NLP Analysis: A Comparative Study of Turkish Mathematics Curricula (2018-2024)". The code implements an unsupervised NLP approach to analyze and compare the Turkish mathematics curricula from 2018 and 2024, revealing significant shifts in content, cognitive complexity, and educational focus.

## Project Overview

This research uses natural language processing techniques to analyze how the Turkish mathematics curriculum has evolved between 2018 and 2024. Our approach includes:

- PDF text extraction with Turkish language support
- Specialized preprocessing for Turkish mathematical text
- Topic modeling using Latent Dirichlet Allocation (LDA)
- Cognitive complexity analysis based on Bloom's taxonomy
- Mathematical domain classification
- Terminology frequency analysis
- Visualization of curricular shifts

## Repository Structure

- `curriculum-preprocessing.py`: PDF extraction and text preprocessing pipeline
- `curriculum-topic-modeling.py`: Topic modeling and semantic analysis
- `eda.py`: Exploratory data analysis of curriculum text
- `curriculum-gap-analysis.py`: Analysis of gaps between 2018 and 2024 curricula
- `custom_stopwords.py`: Custom Turkish stopwords for mathematics curriculum analysis
- `figures/`: Generated visualizations
- `processed_data/`: Intermediate data files
- `models/`: Saved topic models

## Installation

### Requirements

This code has been tested on:

- Python 3.10.16 on Windows 11
- Python 3.9.21 on macOS

### Setup Using Conda (Recommended)

```bash
# Create a conda environment
conda create -n turkish-nlp python=3.9
conda activate turkish-nlp

# Install PyTorch
pip install torch torchvision torchaudio

# Install Turkish NLP model
pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_trf/resolve/main/tr_core_news_trf-1.0-py3-none-any.whl

# Install other dependencies
pip install pandas matplotlib seaborn nltk gensim pyldavis pdfminer.six PyPDF2 wordcloud networkx
```

### Setup Using pip

```bash
# Install PyTorch
pip install torch torchvision torchaudio

# Install Turkish NLP model
pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_trf/resolve/main/tr_core_news_trf-1.0-py3-none-any.whl

# Install other dependencies
pip install pandas matplotlib seaborn nltk gensim pyldavis pdfminer.six PyPDF2 wordcloud networkx
```

## Usage

### Data Preparation

1. Place the Turkish mathematics curriculum PDF files in the `data/` directory
   - The 2018 curriculum PDF should have "2018" in its filename
   - The 2024 curriculum PDF should have "2024" in its filename

### Running the Analysis Pipeline

1. **Text Extraction and Preprocessing**

   ```bash
   python curriculum-preprocessing.py
   ```

   This extracts text from PDFs, segments by sections, and processes learning objectives.

2. **Exploratory Data Analysis**

   ```bash
   python eda.py
   ```

   This performs basic statistical analysis and generates initial visualizations.

3. **Topic Modeling**

   ```bash
   python curriculum-topic-modeling.py
   ```

   This identifies topics in each curriculum and analyzes topic similarities.

4. **Curriculum Gap Analysis**
   ```bash
   python curriculum-gap-analysis.py
   ```
   This analyzes differences between the 2018 and 2024 curricula.

### Outputs

The analysis generates various outputs in the following directories:

- `processed_data/`: Contains CSV and JSON files with analysis results
- `figures/`: Contains visualizations including:
  - Topic model visualizations
  - Cognitive complexity comparisons
  - Word frequency charts
  - Domain coverage comparisons
  - Topic similarity matrices

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions regarding the code or research, please contact [batinorene@gmail.com](mailto:ybatinorene@gmail.com).
