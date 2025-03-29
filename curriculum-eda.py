"""
Turkish Mathematics Curriculum NLP Analysis - Exploratory Data Analysis
=====================================================================
This script performs exploratory data analysis on the preprocessed curriculum data,
including:
1. Basic statistics on the curriculum texts
2. Visualization of key term frequencies
3. Comparative analysis between 2018 and 2024 curricula
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from wordcloud import WordCloud
import time
import networkx as nx

# Import at the beginning of your eda.py
from custom_stopwords import get_turkish_curriculum_stopwords, get_extended_stopwords


# Configure visualizations
plt.style.use('fivethirtyeight')
sns.set(style="whitegrid")

# Define paths
PROCESSED_DIR = "processed_data"
FIGURES_DIR = "figures"

# Create figures directory if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load processed data
def load_processed_data():
    """Load the preprocessed curriculum data."""
    try:
        with open(os.path.join(PROCESSED_DIR, 'processed_curriculum_data.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Processed data file not found. Run preprocessing script first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Check and fix data structure
def check_and_fix_data_structure(processed_data):
    """Check and fix the data structure if needed."""
    # Check if the data is already organized by year
    if '2018' in processed_data and '2024' in processed_data:
        print("Data already organized by year.")
        return processed_data
    
    print("Reorganizing data by curriculum year...")
    # Otherwise, try to organize it
    processed_data_by_year = {
        '2018': None,
        '2024': None
    }
    
    # Try to identify which curriculum is which
    for name, curriculum in processed_data.items():
        if not curriculum:
            continue
            
        detected_type = curriculum.get('detected_type', '')
        
        if '2018' in detected_type or detected_type == '2018-style' or '2018' in name:
            processed_data_by_year['2018'] = curriculum
            print(f"Identified '{name}' as 2018 curriculum.")
        elif '2024' in detected_type or detected_type == '2024-style' or '2024' in name:
            processed_data_by_year['2024'] = curriculum
            print(f"Identified '{name}' as 2024 curriculum.")
        else:
            print(f"Warning: Could not determine year for curriculum '{name}'.")
    
    return processed_data_by_year

# Basic statistics
def compute_basic_stats(processed_data):
    """Compute basic statistics for both curricula."""
    stats = {
        '2018': {},
        '2024': {}
    }
    
    for year in ['2018', '2024']:
        if year not in processed_data or not processed_data[year]:
            print(f"No data available for {year} curriculum. Skipping basic stats calculation.")
            continue
            
        curriculum = processed_data[year]
        
        # Text length statistics
        if 'full_text' not in curriculum or 'processed_full_text' not in curriculum:
            print(f"Missing required data for {year} curriculum. Skipping basic stats calculation.")
            continue
            
        full_text = curriculum['full_text']
        
        if 'words' not in curriculum['processed_full_text'] or 'sentences' not in curriculum['processed_full_text']:
            print(f"Missing processed text data for {year} curriculum. Skipping detailed stats.")
            stats[year]['total_characters'] = len(full_text)
            continue
            
        words = curriculum['processed_full_text']['words']
        sentences = curriculum['processed_full_text']['sentences']
        
        stats[year]['total_characters'] = len(full_text)
        stats[year]['total_words'] = len(words)
        stats[year]['total_sentences'] = len(sentences)
        stats[year]['avg_word_length'] = np.mean([len(word) for word in words])
        stats[year]['avg_sentence_length'] = np.mean([len(sentence.split()) for sentence in sentences])
        
        # Objective statistics
        if 'objectives' not in curriculum:
            print(f"No objectives found for {year} curriculum. Skipping objective stats.")
            continue
            
        objectives = curriculum['objectives']
        stats[year]['total_objectives'] = len(objectives)
        
        if stats[year]['total_objectives'] > 0:
            stats[year]['avg_objective_length'] = np.mean([len(obj['text'].split()) for obj in objectives])
            
            # Count objectives by section
            section_counts = Counter([obj['section'] for obj in objectives])
            stats[year]['objectives_by_section'] = dict(section_counts)
    
    return stats

# Frequency analysis
def analyze_term_frequencies(processed_data):
    """Analyze term frequencies in both curricula using lemmatization."""
    # Create DataFrames for word frequency in each curriculum
    word_freq = {'2018': {}, '2024': {}}
    
    
    custom_stopwords = get_extended_stopwords()
    
    for year in ['2018', '2024']:
        if year not in processed_data or not processed_data[year]:
            print(f"No data available for {year} curriculum. Skipping term frequency analysis.")
            continue
            
        if 'processed_full_text' not in processed_data[year]:
            print(f"Missing processed text data for {year} curriculum. Skipping term frequency analysis.")
            continue
        
        # Get all lemmas from processed objectives for more accurate analysis
        all_lemmas = []
        if 'processed_objectives' in processed_data[year]:
            for obj in processed_data[year]['processed_objectives']:
                if 'spacy_tokens' in obj:
                    # Extract lemmas, exclude punctuation and very short words
                    obj_lemmas = [token['lemma'].lower() for token in obj['spacy_tokens'] 
                                if not token.get('is_punctuation', False) 
                                and not token.get('is_stop', False)
                                and token['lemma'].lower() not in custom_stopwords
                                and len(token['lemma']) > 2]
                    all_lemmas.extend(obj_lemmas)
        
        # Fallback to non-lemmatized words if no objectives found
        if not all_lemmas and 'processed_full_text' in processed_data[year] and 'words' in processed_data[year]['processed_full_text']:
            print(f"Warning: Using non-lemmatized words for {year} curriculum due to missing processed objectives.")
            all_lemmas = [word.lower() for word in processed_data[year]['processed_full_text']['words'] if word.isalnum() and len(word) > 2]
        
        # Count frequencies
        word_freq[year] = Counter(all_lemmas)
    
    # Create a unified DataFrame for comparison
    all_unique_words = list(set(word_freq['2018'].keys()) | set(word_freq['2024'].keys()))
    freq_df = pd.DataFrame(index=all_unique_words, columns=['2018', '2024'])
    
    for word in all_unique_words:
        freq_df.loc[word, '2018'] = word_freq['2018'].get(word, 0)
        freq_df.loc[word, '2024'] = word_freq['2024'].get(word, 0)
    
    # Sort by total frequency
    freq_df['total'] = freq_df['2018'] + freq_df['2024']
    freq_df = freq_df.sort_values('total', ascending=False)
    
    # Add relative frequencies
    for year in ['2018', '2024']:
        if word_freq[year]:  # Only if we have data for this year
            total_words = sum(word_freq[year].values())
            freq_df[f'{year}_relative'] = freq_df[year] / total_words
    
    # Calculate the difference in relative frequencies
    if '2018_relative' in freq_df.columns and '2024_relative' in freq_df.columns:
        freq_df['relative_diff'] = freq_df['2024_relative'] - freq_df['2018_relative']
    
    return freq_df



# Create word clouds
def create_word_clouds(processed_data):
    """Create word clouds for both curricula using lemmatized words."""
    word_clouds = {}
    
    # Get custom stopwords
    custom_stopwords = get_extended_stopwords()
    
    for year in ['2018', '2024']:
        if year not in processed_data or not processed_data[year]:
            print(f"No data available for {year} curriculum. Skipping word cloud creation.")
            continue
        
        # Get lemmas from processed objectives for more accurate visualization
        all_lemmas = []
        if 'processed_objectives' in processed_data[year]:
            for obj in processed_data[year]['processed_objectives']:
                if 'spacy_tokens' in obj:
                    # Extract meaningful lemmas, exclude punctuation, stopwords and very short words
                    obj_lemmas = [token['lemma'].lower() for token in obj['spacy_tokens'] 
                                if not token.get('is_punctuation', False) 
                                and not token.get('is_stop', False)
                                and token['lemma'].lower() not in custom_stopwords
                                and len(token['lemma']) > 2]
                    all_lemmas.extend(obj_lemmas)
        
        # Fallback to words if no lemmas found
        if not all_lemmas and 'processed_full_text' in processed_data[year] and 'words' in processed_data[year]['processed_full_text']:
            print(f"Warning: Using non-lemmatized words for {year} word cloud due to missing processed objectives.")
            all_lemmas = [word.lower() for word in processed_data[year]['processed_full_text']['words'] 
                        if word.isalnum() and len(word) > 2]
        
        if not all_lemmas:
            print(f"No words available for {year} curriculum word cloud.")
            continue
        
        # Count lemma frequencies
        word_freq = Counter(all_lemmas)
        
        # Generate word cloud
        wc = WordCloud(width=800, height=400, background_color='white', 
                     max_words=100, colormap='viridis', 
                     collocations=False).generate_from_frequencies(word_freq)
        
        word_clouds[year] = wc
        
        # Save the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{year} Curriculum Word Cloud (Lemmatized)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'wordcloud_{year}_lemmatized.png'), dpi=300)
        plt.close()
    
    return word_clouds


# Analyze verb usage
def analyze_verb_usage(processed_data):
    """Analyze verb usage to understand cognitive demands."""
    # Load verb analysis if it exists
    try:
        verb_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'verb_analysis.csv'), index_col=0)
        print("Loaded existing verb analysis from file.")
        return verb_df
    except FileNotFoundError:
        print("Verb analysis file not found. Analyzing verbs from processed data...")
        # Extract verbs from processed data
        verbs_2018 = []
        verbs_2024 = []
        
        # First group curricula by year
        curricula_by_year = {'2018': [], '2024': []}
        
        for name, curriculum in processed_data.items():
            if not curriculum:
                continue
                
            # Check if we can determine year from detected_type
            detected_type = curriculum.get('detected_type', '')
            
            if '2018' in detected_type or detected_type == '2018-style' or '2018' in name:
                curricula_by_year['2018'].append(curriculum)
            elif '2024' in detected_type or detected_type == '2024-style' or '2024' in name:
                curricula_by_year['2024'].append(curriculum)
        
        # Process each year
        for year, year_verbs in [('2018', verbs_2018), ('2024', verbs_2024)]:
            if not curricula_by_year[year]:
                print(f"No curricula found for year {year}")
                continue
                
            # Collect all objectives from all curricula for this year
            for curriculum in curricula_by_year[year]:
                if 'processed_objectives' not in curriculum:
                    print(f"No processed objectives found for a {year} curriculum")
                    continue
                    
                # Get verbs from each objective
                for obj in curriculum['processed_objectives']:
                    if 'spacy_tokens' not in obj:
                        continue
                        
                    for token in obj['spacy_tokens']:
                        if token['pos'] == 'VERB':
                            year_verbs.append(token['lemma'])
        
        # Count frequencies
        verbs_2018_freq = Counter(verbs_2018)
        verbs_2024_freq = Counter(verbs_2024)
        
        # Create DataFrame - Convert set to list to avoid the error
        all_verbs = list(set(verbs_2018_freq.keys()) | set(verbs_2024_freq.keys()))
        verb_df = pd.DataFrame(index=all_verbs, columns=['2018', '2024'])
        
        for verb in all_verbs:
            verb_df.loc[verb, '2018'] = verbs_2018_freq.get(verb, 0)
            verb_df.loc[verb, '2024'] = verbs_2024_freq.get(verb, 0)
        
        # Sort by total
        verb_df['total'] = verb_df['2018'] + verb_df['2024']
        verb_df = verb_df.sort_values('total', ascending=False)
        
        # Save verb analysis for future use
        verb_df.to_csv(os.path.join(PROCESSED_DIR, 'verb_analysis.csv'))
    
    # Classify verbs by cognitive level (simplified Bloom's taxonomy)
    bloom_categories = {
        'remember': ['tanımla', 'belirle', 'listele', 'hatırla', 'göster', 'bil', 'tanı', 'tekrarla', 'isimlendir', 'sırala', 'ilişkilendir', 'ezberle', 'kaydet', 'adlandır', 'edin'],
        
        'understand': ['açıkla', 'özetle', 'yorumla', 'örnekle', 'sınıflandır', 'anla', 'kavra', 'karşılaştır', 'ifade et', 'tercüme et', 'yeniden ifade et', 'yerini belirle', 'rapor et', 'farkına var', 'ayırt et', 'tartış', 'betimle', 'gözden geçir', 'çıkarımda bulun', 'örneklendir', 'çiz', 'temsil et', 'farklılaştır', 'sonuçlandır'],
        
        'apply': ['uygula', 'hesapla', 'çöz', 'göster', 'kullan', 'yap', 'gerçekleştir', 'işlet', 'üret', 'keşfet', 'ilişkilendir', 'geliştir', 'çevir', 'düzenle', 'işe koş', 'yeniden yapılandır', 'yorumla', 'resmet', 'pratik yap', 'sergile', 'canlandır'],
        
        'analyze': ['analiz et', 'incele', 'ayırt et', 'sorgula', 'ayrıştır', 'kategorize et', 'araştır', 'düzenle', 'çıkar', 'karşılaştır', 'soruştur', 'zıtlık göster', 'tespit et', 'sınıflandır', 'çıkarım yap', 'deney yap', 'dikkatle incele', 'keşfet', 'parçalara ayır', 'ayrım yap', 'ayır'],
        
        'evaluate': ['değerlendir', 'savun', 'yargıla', 'eleştir', 'karar ver', 'öner', 'ölç', 'seç', 'destekle', 'karşılaştır', 'sonuca var', 'sonuç çıkar', 'tartış', 'derecelendir', 'tahmin et', 'doğrula', 'göz önünde bulundur', 'takdir et', 'değer biç', 'çıkarımda bulun'],
        
        'create': ['oluştur', 'tasarla', 'geliştir', 'planla', 'üret', 'yarat', 'icat et', 'kur', 'formüle et', 'yeniden düzenle', 'bir araya getir', 'hazırla', 'tahmin et', 'değiştir', 'anlat', 'topla', 'genelle', 'belgele', 'birleştir', 'ilişkilendir', 'öner', 'inşa et', 'organize et', 'başlat', 'türet', 'yaz']
    }
    
    # Add Bloom's taxonomy classification
    for category, verbs in bloom_categories.items():
        verb_df[f'bloom_{category}'] = verb_df.index.isin(verbs).astype(int)
    
    return verb_df


# Analyze objective complexity
def analyze_objective_complexity(processed_data):
    """Analyze the complexity of learning objectives."""
    # Check if complexity analysis files already exist
    complexity_2018_path = os.path.join(PROCESSED_DIR, 'objective_complexity_2018.csv')
    complexity_2024_path = os.path.join(PROCESSED_DIR, 'objective_complexity_2024.csv')
    
    complexity_dfs = {}
    
    if os.path.exists(complexity_2018_path):
        print("Loading existing 2018 complexity analysis from file.")
        complexity_dfs['2018'] = pd.read_csv(complexity_2018_path)
    
    if os.path.exists(complexity_2024_path):
        print("Loading existing 2024 complexity analysis from file.")
        complexity_dfs['2024'] = pd.read_csv(complexity_2024_path)
    
    # If both years exist, return early
    if '2018' in complexity_dfs and '2024' in complexity_dfs:
        return complexity_dfs
    
    # Otherwise calculate what's missing
    print("Calculating objective complexity from processed data...")
    complexity_metrics = {
        '2018': [],
        '2024': []
    }
    
    # First group curricula by year
    curricula_by_year = {'2018': [], '2024': []}
    
    for name, curriculum in processed_data.items():
        if not curriculum:
            continue
            
        # Check if we can determine year from detected_type
        detected_type = curriculum.get('detected_type', '')
        
        if '2018' in detected_type or detected_type == '2018-style' or '2018' in name:
            curricula_by_year['2018'].append(curriculum)
        elif '2024' in detected_type or detected_type == '2024-style' or '2024' in name:
            curricula_by_year['2024'].append(curriculum)
    
    for year in ['2018', '2024']:
        # Skip if we already loaded this year's data
        if year in complexity_dfs:
            continue
            
        if not curricula_by_year[year]:
            print(f"No curricula found for year {year}")
            continue
            
        # Process each curriculum
        for curriculum in curricula_by_year[year]:
            if 'processed_objectives' not in curriculum:
                print(f"No processed objectives found for a {year} curriculum")
                continue
                
            objectives = curriculum['processed_objectives']
            
            for obj in objectives:
                # Skip if missing required data
                if 'spacy_tokens' not in obj or 'sentences' not in obj:
                    continue
                    
                # Count tokens, excluding punctuation
                token_count = sum(1 for token in obj['spacy_tokens'] 
                                if not token.get('is_punctuation', False))
                
                # Count unique tokens
                unique_tokens = len(set(token['text'].lower() for token in obj['spacy_tokens'] 
                                        if not token.get('is_punctuation', False)))
                
                # Calculate average token length
                token_lengths = [len(token['text']) for token in obj['spacy_tokens'] 
                                if not token.get('is_punctuation', False)]
                avg_token_length = np.mean(token_lengths) if token_lengths else 0
                
                # Count sentences
                sentence_count = len(obj['sentences'])
                
                # Store metrics
                complexity_metrics[year].append({
                    'section': obj.get('section', ''),
                    'item': obj.get('item', ''),
                    'token_count': token_count,
                    'unique_tokens': unique_tokens,
                    'lexical_diversity': unique_tokens / token_count if token_count > 0 else 0,
                    'avg_token_length': avg_token_length,
                    'sentence_count': sentence_count,
                })
    
    # Convert to DataFrames and merge with any loaded data
    for year in ['2018', '2024']:
        if complexity_metrics[year] and year not in complexity_dfs:
            year_df = pd.DataFrame(complexity_metrics[year])
            complexity_dfs[year] = year_df
            
            # Save for future use
            year_df.to_csv(os.path.join(PROCESSED_DIR, f'objective_complexity_{year}.csv'), index=False)
    
    return complexity_dfs

def analyze_word_relations(processed_data):
    """Analyze word co-occurrence relations using lemmatization."""
    # Create co-occurrence matrices for each year
    cooccurrence = {'2018': {}, '2024': {}}
    
     # Get custom stopwords
    custom_stopwords = get_extended_stopwords()
    
    for year in ['2018', '2024']:
        if year not in processed_data or not processed_data[year]:
            print(f"No data available for {year} curriculum. Skipping word relation analysis.")
            continue
            
        if 'processed_objectives' not in processed_data[year]:
            print(f"No processed objectives available for {year} curriculum. Skipping word relation analysis.")
            continue
        
        # Initialize co-occurrence dictionary
        word_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Process each objective
        for obj in processed_data[year]['processed_objectives']:
            if 'spacy_tokens' not in obj:
                continue
                
            # Get lemmas from this objective (excluding stopwords, punctuation, and short words)
            lemmas = [token['lemma'].lower() for token in obj['spacy_tokens']
                     if not token.get('is_punctuation', False)
                     and not token.get('is_stop', False)
                     and token['lemma'].lower() not in custom_stopwords
                     and len(token['lemma']) > 2]
            
            # Record co-occurrences in this objective
            for i, lemma1 in enumerate(lemmas):
                for lemma2 in lemmas[i+1:]:
                    if lemma1 != lemma2:  # Avoid self-co-occurrence
                        word_cooccurrence[lemma1][lemma2] += 1
                        word_cooccurrence[lemma2][lemma1] += 1
        
        cooccurrence[year] = word_cooccurrence
    
    # For each year, create and save a network visualization of top co-occurrences
    for year, cooc_data in cooccurrence.items():
        if not cooc_data:
            continue
            
        # Create a graph
        G = nx.Graph()
        
        # Get top 50 most frequent words
        word_counts = Counter()
        for word, coocs in cooc_data.items():
            word_counts[word] = sum(coocs.values())
        
        top_words = [word for word, _ in word_counts.most_common(50)]
        
        # Add nodes and edges for top words
        for word in top_words:
            G.add_node(word, count=word_counts[word])
            
            # Add edges to other top words
            for cooc_word, weight in cooc_data[word].items():
                if cooc_word in top_words and weight > 1:  # Minimum co-occurrence threshold
                    G.add_edge(word, cooc_word, weight=weight)
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if not G.nodes():
            print(f"No significant word co-occurrences found for {year} curriculum.")
            continue
        
        # Calculate layout
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # Create visualization
        plt.figure(figsize=(12, 12))
        
        # Node sizes based on frequency
        node_sizes = [G.nodes[node]['count'] * 10 for node in G.nodes()]
        
        # Edge widths based on co-occurrence weight
        edge_widths = [G[u][v]['weight'] / 2 for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8, 
                             node_color='skyblue', linewidths=1, edgecolors='gray')
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_weight='bold')
        
        plt.title(f'{year} Curriculum Word Relations (Lemmatized)', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(FIGURES_DIR, f'word_relations_{year}.png'), dpi=300)
        plt.close()
        
        # Save the network data for potential further analysis
        with open(os.path.join(PROCESSED_DIR, f'word_relations_{year}.pkl'), 'wb') as f:
            pickle.dump((G, pos), f)
    
    return cooccurrence

def analyze_mathematical_terminology(processed_data):
    """Analyze mathematical terminology using lemmatization."""
    # Mathematical term categories (using lemmatized forms)
    math_categories = {
        'algebra': ['cebir', 'denklem', 'eşitlik', 'ifade', 'polinom', 'fonksiyon', 'değişken'],
        'geometry': ['geometri', 'açı', 'üçgen', 'çokgen', 'doğru', 'nokta', 'çember', 'alan', 'hacim'],
        'calculus': ['türev', 'integral', 'limit', 'süreklilik', 'diferansiyel', 'gradient'],
        'statistics': ['istatistik', 'olasılık', 'ortalama', 'medyan', 'dağılım', 'standart sapma', 'varyans'],
        'number_theory': ['sayı', 'asal', 'tam sayı', 'kesir', 'rasyonel', 'bölünebilme'],
        'logic': ['mantık', 'önerme', 'çıkarım', 'doğruluk', 'yanlışlık', 'ispat']
    }
    
    term_counts = {
        '2018': {category: 0 for category in math_categories},
        '2024': {category: 0 for category in math_categories}
    }
    
    for year in ['2018', '2024']:
        if year not in processed_data or not processed_data[year]:
            print(f"No data available for {year} curriculum. Skipping mathematical terminology analysis.")
            continue
            
        if 'processed_objectives' not in processed_data[year]:
            print(f"No processed objectives for {year} curriculum. Skipping mathematical terminology analysis.")
            continue
        
        # Count terms in each category
        for obj in processed_data[year]['processed_objectives']:
            if 'spacy_tokens' not in obj:
                continue
                
            # Get lemmas from this objective
            lemmas = [token['lemma'].lower() for token in obj['spacy_tokens']]
            lemma_text = ' '.join(lemmas)
            
            # Check each category
            for category, terms in math_categories.items():
                for term in terms:
                    if term in lemma_text:
                        term_counts[year][category] += 1
                        break  # Count each category only once per objective
    
    # Create a DataFrame for visualization
    term_df = pd.DataFrame(term_counts)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    term_df.plot(kind='bar')
    plt.title('Mathematical Terminology Categories')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'mathematical_terminology.png'), dpi=300)
    plt.close()
    
    # Save the data
    term_df.to_csv(os.path.join(PROCESSED_DIR, 'mathematical_terminology.csv'))
    
    return term_df


def plot_comparative_charts(data_dict):
    """Create comparative visualizations."""
    # Basic stats comparison
    if 'basic_stats' in data_dict:
        stats = data_dict['basic_stats']
        
        # Create a DataFrame for basic metrics
        metrics = ['total_words', 'total_sentences', 'total_objectives', 
                   'avg_word_length', 'avg_sentence_length']
        
        # Extract metrics that exist in both years
        existing_metrics = []
        for metric in metrics:
            if metric in stats.get('2018', {}) and metric in stats.get('2024', {}):
                existing_metrics.append(metric)
        
        if existing_metrics:
            metrics_df = pd.DataFrame({
                '2018': [stats['2018'].get(m, 0) for m in existing_metrics],
                '2024': [stats['2024'].get(m, 0) for m in existing_metrics]
            }, index=existing_metrics)
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Basic Curriculum Metrics Comparison')
            plt.ylabel('Value')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'basic_metrics_comparison.png'), dpi=300)
            plt.close()
    
    # Term frequency comparison
    if 'term_freq' in data_dict:
        freq_df = data_dict['term_freq']
        
        # Plot top 20 words by difference in relative frequency
        if 'relative_diff' in freq_df.columns:
            top_increased = freq_df.sort_values('relative_diff', ascending=False).head(20)
            top_decreased = freq_df.sort_values('relative_diff', ascending=True).head(20)
            
            # Plot increased terms
            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_increased.index, y=top_increased['relative_diff'])
            plt.title('Top 20 Terms with Increased Relative Frequency in 2024')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'top_increased_terms.png'), dpi=300)
            plt.close()
            
            # Plot decreased terms
            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_decreased.index, y=top_decreased['relative_diff'])
            plt.title('Top 20 Terms with Decreased Relative Frequency in 2024')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'top_decreased_terms.png'), dpi=300)
            plt.close()
    
    # Verb usage comparison
    if 'verb_usage' in data_dict:
        verb_df = data_dict['verb_usage']
        
        # Plot top 20 verbs
        top_verbs = verb_df.head(20)
        
        plt.figure(figsize=(12, 6))
        top_verbs[['2018', '2024']].plot(kind='bar')
        plt.title('Top 20 Verbs Comparison')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'top_verbs_comparison.png'), dpi=300)
        plt.close()
        
        # Plot Bloom's taxonomy distribution if available
        bloom_cols = [col for col in verb_df.columns if col.startswith('bloom_')]
        if bloom_cols:
            # Sum up verbs in each category
            bloom_totals = verb_df[bloom_cols].sum()
            
            plt.figure(figsize=(10, 6))
            bloom_totals.plot(kind='bar')
            plt.title("Bloom's Taxonomy Verb Distribution")
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'blooms_taxonomy_distribution.png'), dpi=300)
            plt.close()
    
    # Objective complexity comparison
    if 'complexity' in data_dict and '2018' in data_dict['complexity'] and '2024' in data_dict['complexity']:
        complexity_2018 = data_dict['complexity']['2018']
        complexity_2024 = data_dict['complexity']['2024']
        
        # Compare average complexity metrics (excluding AI relevance score)
        metrics = ['token_count', 'unique_tokens', 'lexical_diversity', 'avg_token_length']
        
        avg_metrics = {
            '2018': {metric: complexity_2018[metric].mean() for metric in metrics if metric in complexity_2018},
            '2024': {metric: complexity_2024[metric].mean() for metric in metrics if metric in complexity_2024}
        }
        
        # Common metrics between both years
        common_metrics = set(avg_metrics['2018'].keys()) & set(avg_metrics['2024'].keys())
        
        if common_metrics:
            avg_df = pd.DataFrame({
                '2018': [avg_metrics['2018'][m] for m in common_metrics],
                '2024': [avg_metrics['2024'][m] for m in common_metrics]
            }, index=list(common_metrics))
            
            plt.figure(figsize=(12, 6))
            avg_df.plot(kind='bar')
            plt.title('Average Objective Complexity Metrics')
            plt.ylabel('Value')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'objective_complexity_comparison.png'), dpi=300)
            plt.close()



# Replace the AI relevance analysis function with a curriculum theme analysis
def analyze_curriculum_themes(processed_data):
    """
    Analyze curriculum themes based on word frequencies and distributions
    without relying on predefined AI categories.
    """
    # Check if we've already analyzed this
    themes_path = os.path.join(PROCESSED_DIR, 'curriculum_themes.json')
    if os.path.exists(themes_path):
        try:
            print("Loading existing curriculum theme analysis")
            with open(themes_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing theme analysis: {e}")
    
    # Group data by curriculum year
    curricula_by_year = {'2018': [], '2024': []}
    
    for name, curriculum in processed_data.items():
        if not curriculum:
            continue
            
        # Determine year from detected_type
        if '2018' in curriculum.get('detected_type', '') or '2018' in name:
            curricula_by_year['2018'].append(curriculum)
        elif '2024' in curriculum.get('detected_type', '') or '2024' in name:
            curricula_by_year['2024'].append(curriculum)
    
    # Extract most frequent terms for each year
    term_frequencies = {'2018': Counter(), '2024': Counter()}
    custom_stopwords = get_extended_stopwords()
    
    for year, curricula in curricula_by_year.items():
        all_tokens = []
        for curriculum in curricula:
            if 'processed_objectives' not in curriculum:
                continue
                
            for obj in curriculum['processed_objectives']:
                if 'spacy_tokens' not in obj:
                    continue
                    
                # Extract meaningful lemmas, exclude stopwords and very short words
                tokens = [token['lemma'].lower() for token in obj['spacy_tokens'] 
                         if not token.get('is_punctuation', False) 
                         and not token.get('is_stop', False)
                         and token['lemma'].lower() not in custom_stopwords
                         and len(token['lemma']) > 2]
                
                all_tokens.extend(tokens)
        
        term_frequencies[year] = Counter(all_tokens)
    
    # Identify top terms for each year
    top_terms = {
        '2018': [term for term, count in term_frequencies['2018'].most_common(100)],
        '2024': [term for term, count in term_frequencies['2024'].most_common(100)]
    }
    
    # Identify unique and shared terms
    unique_to_2018 = set(top_terms['2018']) - set(top_terms['2024'])
    unique_to_2024 = set(top_terms['2024']) - set(top_terms['2018'])
    shared_terms = set(top_terms['2018']) & set(top_terms['2024'])
    
    # Calculate term importance scores (normalized frequency)
    term_importance = {'2018': {}, '2024': {}}
    
    for year in ['2018', '2024']:
        total_count = sum(term_frequencies[year].values())
        if total_count > 0:
            term_importance[year] = {term: count/total_count 
                                   for term, count in term_frequencies[year].items()}
    
    # Prepare results
    themes_analysis = {
        'term_frequencies': {
            '2018': dict(term_frequencies['2018'].most_common(500)),
            '2024': dict(term_frequencies['2024'].most_common(500))
        },
        'top_terms': top_terms,
        'unique_terms': {
            '2018': list(unique_to_2018),
            '2024': list(unique_to_2024)
        },
        'shared_terms': list(shared_terms),
        'term_importance': {
            '2018': {term: importance for term, importance in 
                    sorted(term_importance['2018'].items(), key=lambda x: x[1], reverse=True)[:200]},
            '2024': {term: importance for term, importance in 
                    sorted(term_importance['2024'].items(), key=lambda x: x[1], reverse=True)[:200]}
        }
    }
    
    # Save results
    with open(themes_path, 'w', encoding='utf-8') as f:
        json.dump(themes_analysis, f, ensure_ascii=False, indent=2)
    
    # Create visualizations
    visualize_curriculum_themes(themes_analysis)
    
    return themes_analysis

def visualize_curriculum_themes(themes_analysis):
    """Create visualizations for curriculum theme analysis."""
    # Visualize unique terms for each curriculum
    for year in ['2018', '2024']:
        # Get unique terms with their importance scores
        unique_terms = themes_analysis['unique_terms'][year]
        if not unique_terms:
            continue
            
        # Get importance scores for these terms
        term_scores = {term: themes_analysis['term_importance'][year].get(term, 0) 
                      for term in unique_terms}
        
        # Sort terms by importance
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Only use top 20 terms for visualization clarity
        top_unique_terms = sorted_terms[:20]
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        terms = [t[0] for t in top_unique_terms]
        scores = [t[1] for t in top_unique_terms]
        
        plt.barh(range(len(terms)), scores, align='center')
        plt.yticks(range(len(terms)), terms)
        plt.xlabel('Normalized Frequency')
        plt.title(f'Top Unique Terms in {year} Curriculum')
        plt.gca().invert_yaxis()  # Display terms in descending order
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_DIR, f'unique_terms_{year}.png'), dpi=300)
        plt.close()
    
    # Visualize term frequency changes
    # Get terms that exist in both years
    common_terms = set(themes_analysis['term_frequencies']['2018'].keys()) & \
                  set(themes_analysis['term_frequencies']['2024'].keys())
    
    freq_changes = []
    for term in common_terms:
        freq_2018 = themes_analysis['term_importance']['2018'].get(term, 0)
        freq_2024 = themes_analysis['term_importance']['2024'].get(term, 0)
        
        # Calculate change ratio
        if freq_2018 > 0:
            change_ratio = (freq_2024 - freq_2018) / freq_2018
        else:
            change_ratio = float('inf')  # Term didn't exist in 2018
            
        freq_changes.append((term, change_ratio))
    
    # Sort by change ratio
    freq_changes.sort(key=lambda x: x[1], reverse=True)
    
    # Plot terms with highest increase
    if freq_changes:
        increased_terms = [(t, r) for t, r in freq_changes if r > 0][:15]
        
        if increased_terms:
            plt.figure(figsize=(12, 8))
            terms = [t[0] for t in increased_terms]
            ratios = [min(t[1], 10) for t in increased_terms]  # Cap at 10x for visualization
            
            plt.barh(range(len(terms)), ratios, align='center')
            plt.yticks(range(len(terms)), terms)
            plt.xlabel('Frequency Change Ratio (2024/2018 - 1)')
            plt.title('Terms with Largest Frequency Increase in 2024 Curriculum')
            plt.gca().invert_yaxis()  # Display terms in descending order
            plt.tight_layout()
            
            plt.savefig(os.path.join(FIGURES_DIR, 'terms_increased.png'), dpi=300)
            plt.close()
        
        # Plot terms with highest decrease
        decreased_terms = [(t, r) for t, r in freq_changes if r < 0][-15:]
        decreased_terms.reverse()  # Show largest decrease first
        
        if decreased_terms:
            plt.figure(figsize=(12, 8))
            terms = [t[0] for t in decreased_terms]
            ratios = [abs(t[1]) for t in decreased_terms]  # Use absolute values for clearer visualization
            
            plt.barh(range(len(terms)), ratios, align='center')
            plt.yticks(range(len(terms)), terms)
            plt.xlabel('Absolute Frequency Change Ratio (|2024/2018 - 1|)')
            plt.title('Terms with Largest Frequency Decrease in 2024 Curriculum')
            plt.gca().invert_yaxis()  # Display terms in descending order
            plt.tight_layout()
            
            plt.savefig(os.path.join(FIGURES_DIR, 'terms_decreased.png'), dpi=300)
            plt.close()

def analyze_cognitive_complexity(processed_data):
    """Analyze cognitive complexity based on verb usage in objectives."""
    # Define cognitive levels based on Bloom's taxonomy
    bloom_categories = {
        'remember': ['tanımla', 'belirle', 'listele', 'hatırla', 'göster', 'bilir', 'tanır', 'tekrarlar'],
        'understand': ['açıkla', 'özetle', 'yorumla', 'örnek', 'sınıflandır', 'anlar', 'kavrar', 'karşılaştır'],
        'apply': ['uygula', 'hesapla', 'çöz', 'göster', 'kullan', 'yapar', 'kullanır', 'uygular'],
        'analyze': ['analiz', 'karşılaştır', 'incele', 'ayırt', 'test', 'analiz eder', 'ayrıştırır', 'sorgular'],
        'evaluate': ['değerlendir', 'eleştir', 'savun', 'yargıla', 'seç', 'karar verir', 'değer biçer'],
        'create': ['oluştur', 'tasarla', 'geliştir', 'planla', 'üret', 'yaratır', 'icat eder', 'kurar']
    }
    
    # Assign cognitive level scores
    cognitive_levels = {
        'remember': 1,
        'understand': 2,
        'apply': 3,
        'analyze': 4,
        'evaluate': 5,
        'create': 6
    }
    
    # Collect verbs by year and objective
    objectives_by_year = {'2018': [], '2024': []}
    
    for name, curriculum in processed_data.items():
        if not curriculum:
            continue
            
        # Determine year
        year = '2024' if '2024' in curriculum.get('detected_type', '') or '2024' in name else '2018'
        
        if 'processed_objectives' not in curriculum:
            continue
            
        for obj in curriculum['processed_objectives']:
            if 'spacy_tokens' not in obj:
                continue
                
            # Extract verbs
            verbs = [token['lemma'].lower() for token in obj['spacy_tokens'] 
                    if token['pos'] == 'VERB']
            
            if verbs:
                objectives_by_year[year].append({
                    'section': obj.get('section', ''),
                    'text': obj.get('cleaned_text', ''),
                    'verbs': verbs
                })
    
    # Analyze cognitive levels by year
    cognitive_analysis = {'2018': {}, '2024': {}}
    
    for year in ['2018', '2024']:
        # Count objectives at each cognitive level
        level_counts = {level: 0 for level in cognitive_levels}
        total_objectives = len(objectives_by_year[year])
        
        # Track the highest cognitive level for each objective
        for obj in objectives_by_year[year]:
            max_level = 0
            max_level_name = 'none'
            
            for verb in obj['verbs']:
                for level, verbs in bloom_categories.items():
                    if verb in verbs and cognitive_levels[level] > max_level:
                        max_level = cognitive_levels[level]
                        max_level_name = level
            
            if max_level > 0:
                level_counts[max_level_name] += 1
            
        # Calculate percentages
        level_percentages = {}
        if total_objectives > 0:
            for level, count in level_counts.items():
                level_percentages[level] = count / total_objectives * 100
                
        # Record results
        cognitive_analysis[year] = {
            'level_counts': level_counts,
            'level_percentages': level_percentages,
            'total_objectives': total_objectives
        }
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Order levels by cognitive complexity
    ordered_levels = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    # Prepare data for comparison
    x = np.arange(len(ordered_levels))
    width = 0.35
    
    # Plot bars for each year
    percentages_2018 = [cognitive_analysis['2018']['level_percentages'].get(level, 0) 
                      for level in ordered_levels]
    percentages_2024 = [cognitive_analysis['2024']['level_percentages'].get(level, 0) 
                      for level in ordered_levels]
    
    plt.bar(x - width/2, percentages_2018, width, label='2018 Curriculum')
    plt.bar(x + width/2, percentages_2024, width, label='2024 Curriculum')
    
    plt.xlabel('Cognitive Level (Bloom\'s Taxonomy)')
    plt.ylabel('Percentage of Objectives')
    plt.title('Cognitive Complexity Analysis')
    plt.xticks(x, ordered_levels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cognitive_complexity.png'), dpi=300)
    plt.close()
    
    # Save results
    with open(os.path.join(PROCESSED_DIR, 'cognitive_complexity.json'), 'w', encoding='utf-8') as f:
        json.dump(cognitive_analysis, f, ensure_ascii=False, indent=2)
    
    return cognitive_analysis


def generate_eda_summary(basic_stats, term_freq, verb_usage, complexity, math_terms):
    """Generate a comprehensive text summary of the exploratory data analysis results."""
    summary = []
    
    # Create a header
    summary.append("# Turkish Mathematics Curriculum Exploratory Analysis (2018 vs 2024)")
    summary.append("=" * 70)
    summary.append("")
    
    # Basic statistics summary
    summary.append("## Basic Curriculum Statistics")
    summary.append("-" * 40)
    
    if basic_stats:
        # Create comparison table
        summary.append("\n| Metric | 2018 Curriculum | 2024 Curriculum | Change |")
        summary.append("| ------ | --------------- | --------------- | ------ |")
        
        # Common metrics to display
        metrics = [
            ('total_words', 'Total Words'),
            ('total_sentences', 'Total Sentences'),
            ('total_objectives', 'Learning Objectives'),
            ('avg_word_length', 'Avg. Word Length'),
            ('avg_sentence_length', 'Avg. Sentence Length'),
            ('avg_objective_length', 'Avg. Objective Length')
        ]
        
        for metric_key, metric_name in metrics:
            if metric_key in basic_stats.get('2018', {}) and metric_key in basic_stats.get('2024', {}):
                val_2018 = basic_stats['2018'][metric_key]
                val_2024 = basic_stats['2024'][metric_key]
                
                # Format based on metric type
                if 'avg' in metric_key:
                    val_2018_str = f"{val_2018:.2f}"
                    val_2024_str = f"{val_2024:.2f}"
                    change = f"{val_2024 - val_2018:+.2f}"
                else:
                    val_2018_str = f"{val_2018:,}"
                    val_2024_str = f"{val_2024:,}"
                    change = f"{val_2024 - val_2018:+,}"
                
                summary.append(f"| {metric_name} | {val_2018_str} | {val_2024_str} | {change} |")
    
    # Term frequency analysis
    summary.append("\n\n## Key Terminology Changes")
    summary.append("-" * 40)
    
    if isinstance(term_freq, pd.DataFrame) and 'relative_diff' in term_freq.columns:
        # Top 10 increasing terms
        top_increased = term_freq.sort_values('relative_diff', ascending=False).head(10)
        
        summary.append("\n### Terms with Largest Increase in Frequency (2024 vs 2018)")
        summary.append("\n| Term | 2018 Frequency | 2024 Frequency | Change |")
        summary.append("| ---- | -------------- | -------------- | ------ |")
        
        for term, row in top_increased.iterrows():
            rel_2018 = row['2018_relative'] if '2018_relative' in row else 0
            rel_2024 = row['2024_relative'] if '2024_relative' in row else 0
            rel_diff = row['relative_diff']
            
            # Format percentages
            summary.append(f"| {term} | {rel_2018:.4%} | {rel_2024:.4%} | {rel_diff:+.4%} |")
        
        # Top 10 decreasing terms
        top_decreased = term_freq.sort_values('relative_diff', ascending=True).head(10)
        
        summary.append("\n### Terms with Largest Decrease in Frequency (2024 vs 2018)")
        summary.append("\n| Term | 2018 Frequency | 2024 Frequency | Change |")
        summary.append("| ---- | -------------- | -------------- | ------ |")
        
        for term, row in top_decreased.iterrows():
            rel_2018 = row['2018_relative'] if '2018_relative' in row else 0
            rel_2024 = row['2024_relative'] if '2024_relative' in row else 0
            rel_diff = row['relative_diff']
            
            # Format percentages
            summary.append(f"| {term} | {rel_2018:.4%} | {rel_2024:.4%} | {rel_diff:+.4%} |")
    
    # Verb usage analysis (cognitive dimensions)
    summary.append("\n\n## Cognitive Dimensions (Verb Usage)")
    summary.append("-" * 40)
    
    if isinstance(verb_usage, pd.DataFrame):
        # Top 10 most common verbs
        top_verbs = verb_usage.head(10)
        
        summary.append("\n### Most Common Verbs in Both Curricula")
        summary.append("\n| Verb | 2018 Count | 2024 Count | Total |")
        summary.append("| ---- | ---------- | ---------- | ----- |")
        
        for verb, row in top_verbs.iterrows():
            count_2018 = row['2018'] if '2018' in row else 0
            count_2024 = row['2024'] if '2024' in row else 0
            total = row['total'] if 'total' in row else count_2018 + count_2024
            
            summary.append(f"| {verb} | {count_2018} | {count_2024} | {total} |")
        
        # Bloom's taxonomy distribution
        bloom_cols = [col for col in verb_usage.columns if col.startswith('bloom_')]
        if bloom_cols:
            bloom_counts = verb_usage[bloom_cols].sum()
            
            summary.append("\n### Bloom's Taxonomy Category Distribution")
            for col in bloom_cols:
                category = col.replace('bloom_', '').capitalize()
                count = bloom_counts[col]
                summary.append(f"- **{category}**: {count} verbs")
    
    # Mathematical terminology analysis
    summary.append("\n\n## Mathematical Content Focus")
    summary.append("-" * 40)
    
    if isinstance(math_terms, pd.DataFrame):
        summary.append("\n### Mathematical Domain Coverage")
        summary.append("\n| Domain | 2018 Coverage | 2024 Coverage | Change |")
        summary.append("| ------ | ------------- | ------------- | ------ |")
        
        for domain, row in math_terms.iterrows():
            val_2018 = row['2018'] if '2018' in row else 0
            val_2024 = row['2024'] if '2024' in row else 0
            change = val_2024 - val_2018
            
            summary.append(f"| {domain.replace('_', ' ').capitalize()} | {val_2018} | {val_2024} | {change:+} |")
    
    # Objective complexity analysis
    summary.append("\n\n## Learning Objective Complexity")
    summary.append("-" * 40)
    
    if complexity and '2018' in complexity and '2024' in complexity:
        # Average complexity metrics
        metrics = ['token_count', 'unique_tokens', 'lexical_diversity', 'avg_token_length']
        
        summary.append("\n### Objective Complexity Metrics")
        summary.append("\n| Metric | 2018 Avg. | 2024 Avg. | Change |")
        summary.append("| ------ | --------- | --------- | ------ |")
        
        for metric in metrics:
            if metric in complexity['2018'] and metric in complexity['2024']:
                avg_2018 = complexity['2018'][metric].mean()
                avg_2024 = complexity['2024'][metric].mean()
                change = avg_2024 - avg_2018
                
                # Format based on metric
                if metric == 'lexical_diversity':
                    summary.append(f"| {metric.replace('_', ' ').title()} | {avg_2018:.3f} | {avg_2024:.3f} | {change:+.3f} |")
                else:
                    summary.append(f"| {metric.replace('_', ' ').title()} | {avg_2018:.2f} | {avg_2024:.2f} | {change:+.2f} |")
    
    # Key insights summary
    summary.append("\n\n## Key Insights from Exploratory Analysis")
    summary.append("-" * 40)
    summary.append("\nBased on the exploratory analysis, the following key insights emerge:")
    
    # Generate insights based on available data
    insights = []
    
    # Insight from basic stats
    if basic_stats and 'total_objectives' in basic_stats.get('2018', {}) and 'total_objectives' in basic_stats.get('2024', {}):
        obj_change = basic_stats['2024']['total_objectives'] - basic_stats['2018']['total_objectives']
        if abs(obj_change) > 10:
            if obj_change > 0:
                insights.append(f"The 2024 curriculum has {obj_change} more learning objectives than the 2018 version, suggesting expanded coverage or more detailed specification of learning outcomes.")
            else:
                insights.append(f"The 2024 curriculum has {abs(obj_change)} fewer learning objectives than the 2018 version, suggesting streamlining or consolidation of learning outcomes.")
    
    # Insight from term frequency
    if isinstance(term_freq, pd.DataFrame) and 'relative_diff' in term_freq.columns:
        top_terms = term_freq.sort_values('relative_diff', ascending=False).head(3).index.tolist()
        if top_terms:
            term_str = ", ".join([f"'{t}'" for t in top_terms])
            insights.append(f"The most substantially increased terms in the 2024 curriculum are {term_str}, suggesting new or expanded emphasis in these areas.")
    
    # Insight from mathematical terminology
    if isinstance(math_terms, pd.DataFrame):
        # Find domains with biggest changes
        domain_changes = []
        for domain in math_terms.index:
            val_2018 = math_terms.loc[domain, '2018'] if '2018' in math_terms.columns else 0
            val_2024 = math_terms.loc[domain, '2024'] if '2024' in math_terms.columns else 0
            change = val_2024 - val_2018
            domain_changes.append((domain, change))
        
        domain_changes.sort(key=lambda x: abs(x[1]), reverse=True)
        if domain_changes:
            top_domain, change = domain_changes[0]
            if change > 0:
                insights.append(f"The '{top_domain.replace('_', ' ')}' domain shows the largest increase in coverage (+{change}), indicating expanded emphasis in this area of mathematics.")
            elif change < 0:
                insights.append(f"The '{top_domain.replace('_', ' ')}' domain shows the largest decrease in coverage ({change}), suggesting reduced emphasis in this area of mathematics.")
    
    # Add collected insights
    for i, insight in enumerate(insights):
        summary.append(f"\n{i+1}. {insight}")
    
    # Conclusion
    summary.append("\n\nThis exploratory analysis provides a quantitative foundation for understanding the evolution of the Turkish Mathematics Curriculum from 2018 to 2024, identifying significant changes in content focus, complexity, and cognitive demands.")
    
    return "\n".join(summary)

# Main function
def main():
    """Main function to execute the exploratory data analysis."""
    start_time = time.time()
    
    print("Loading processed data...")
    processed_data = load_processed_data()
    
    if not processed_data:
        print("Error: Could not load processed data.")
        return
    
    print("Checking and fixing data structure...")
    processed_data = check_and_fix_data_structure(processed_data)
    
    print("Computing basic statistics...")
    basic_stats = compute_basic_stats(processed_data)
    
    print("Analyzing term frequencies...")
    term_freq = analyze_term_frequencies(processed_data)
    
    print("Creating word clouds...")
    word_clouds = create_word_clouds(processed_data)
    
    print("Analyzing curriculum themes...")
    themes_analysis = analyze_curriculum_themes(processed_data)
    
    print("Analyzing verb usage...")
    verb_usage = analyze_verb_usage(processed_data)
    
    print("Analyzing cognitive complexity...")
    cognitive_analysis = analyze_cognitive_complexity(processed_data)
    
    print("Analyzing objective complexity...")
    complexity = analyze_objective_complexity(processed_data)
    
    print("Analyzing mathematical terminology...")
    math_terms = analyze_mathematical_terminology(processed_data)

    print("Analyzing word relations...")
    word_relations = analyze_word_relations(processed_data)

    # Collect all results
    results = {
        'basic_stats': basic_stats,
        'term_freq': term_freq,
        'verb_usage': verb_usage,
        'complexity': complexity,
        'math_terms': math_terms,
        'word_relations': word_relations
    }
    
    print("Creating visualizations...")
    plot_comparative_charts(results)
    
    # Save results to CSV files
    print("Saving results...")
    term_freq.to_csv(os.path.join(PROCESSED_DIR, 'term_frequency_analysis.csv'))
    verb_usage.to_csv(os.path.join(PROCESSED_DIR, 'verb_usage_detailed.csv'))
    
    for year in ['2018', '2024']:
        if year in complexity:
            complexity[year].to_csv(os.path.join(PROCESSED_DIR, f'objective_complexity_{year}.csv'))
    
    # Save basic stats as JSON
    with open(os.path.join(PROCESSED_DIR, 'basic_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(basic_stats, f, indent=2)
    
    print("Generating exploratory analysis summary...")
    eda_summary = generate_eda_summary(
        basic_stats,
        term_freq,
        verb_usage,
        complexity,
        math_terms
    )

    # Save to file
    eda_summary_path = os.path.join(PROCESSED_DIR, 'exploratory_analysis_summary.txt')
    with open(eda_summary_path, 'w', encoding='utf-8') as f:
        f.write(eda_summary)

    print(f"Exploratory analysis summary saved to: {eda_summary_path}")
    
    elapsed_time = time.time() - start_time
    print(f"Exploratory data analysis complete in {elapsed_time:.2f} seconds.")
    print(f"Results saved to {PROCESSED_DIR} and {FIGURES_DIR} directories.")
    print(f"Files created in {PROCESSED_DIR}:")
    print("  - term_frequency_analysis.csv")
    print("  - verb_usage_detailed.csv")
    print("  - curriculum_themes.json")
    print("  - cognitive_complexity.json")
    print("  - objective_complexity_2018.csv (if 2018 data available)")
    print("  - objective_complexity_2024.csv (if 2024 data available)")
    print("  - basic_stats.json")
    print("  - mathematical_terminology.csv")
    
    print(f"\nVisualization files created in {FIGURES_DIR}:")
    print("  - wordcloud_2018_lemmatized.png (if 2018 data available)")
    print("  - wordcloud_2024_lemmatized.png (if 2024 data available)")
    print("  - unique_terms_2018.png")
    print("  - unique_terms_2024.png")
    print("  - terms_increased.png")
    print("  - terms_decreased.png")
    print("  - cognitive_complexity.png")
    print("  - basic_metrics_comparison.png")
    print("  - top_increased_terms.png")
    print("  - top_decreased_terms.png")
    print("  - top_verbs_comparison.png")
    print("  - blooms_taxonomy_distribution.png")
    print("  - objective_complexity_comparison.png")
    print("  - mathematical_terminology.png")
    print("  - word_relations_2018.png (if 2018 data available)")
    print("  - word_relations_2024.png (if 2024 data available)")
    
if __name__ == "__main__":
    main()