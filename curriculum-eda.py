"""
Turkish Mathematics Curriculum NLP Analysis - Exploratory Data Analysis
=====================================================================
This script performs exploratory data analysis on the preprocessed curriculum data,
including:
1. Basic statistics on the curriculum texts
2. Visualization of key term frequencies
3. Comparative analysis between 2018 and 2024 curricula
4. Basic AI relevance analysis
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

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

# Basic statistics
def compute_basic_stats(processed_data):
    """Compute basic statistics for both curricula."""
    stats = {
        '2018': {},
        '2024': {}
    }
    
    for year in ['2018', '2024']:
        if not processed_data[year]:
            continue
            
        curriculum = processed_data[year]
        
        # Text length statistics
        full_text = curriculum['full_text']
        words = curriculum['processed_full_text']['words']
        sentences = curriculum['processed_full_text']['sentences']
        
        stats[year]['total_characters'] = len(full_text)
        stats[year]['total_words'] = len(words)
        stats[year]['total_sentences'] = len(sentences)
        stats[year]['avg_word_length'] = np.mean([len(word) for word in words])
        stats[year]['avg_sentence_length'] = np.mean([len(sentence.split()) for sentence in sentences])
        
        # Objective statistics
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
    """Analyze term frequencies in both curricula."""
    # Create DataFrames for word frequency in each curriculum
    word_freq = {'2018': {}, '2024': {}}
    
    for year in ['2018', '2024']:
        if not processed_data[year]:
            continue
            
        # Get all words
        all_words = processed_data[year]['processed_full_text']['words']
        
        # Remove punctuation and convert to lowercase
        all_words = [word.lower() for word in all_words if word.isalnum()]
        
        # Count frequencies
        word_freq[year] = Counter(all_words)
    
    # Create a unified DataFrame for comparison
    all_unique_words = set(word_freq['2018'].keys()) | set(word_freq['2024'].keys())
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
    """Create word clouds for both curricula."""
    word_clouds = {}
    
    for year in ['2018', '2024']:
        if not processed_data[year]:
            continue
            
        # Get all words and their frequencies
        all_words = processed_data[year]['processed_full_text']['words']
        all_words = [word.lower() for word in all_words if word.isalnum()]
        word_freq = Counter(all_words)
        
        # Generate word cloud
        wc = WordCloud(width=800, height=400, background_color='white', 
                       max_words=100, colormap='viridis', 
                       collocations=False).generate_from_frequencies(word_freq)
        
        word_clouds[year] = wc
        
        # Save the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{year} Curriculum Word Cloud')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'wordcloud_{year}.png'), dpi=300)
        plt.close()
    
    return word_clouds

# Analyze AI relevance
def analyze_ai_relevance(processed_data):
    """Analyze AI relevance metrics for both curricula."""
    # Load AI relevance summary if it exists
    try:
        with open(os.path.join(PROCESSED_DIR, 'ai_relevance_summary.json'), 'r', encoding='utf-8') as f:
            ai_summary = json.load(f)
    except FileNotFoundError:
        # Calculate from processed data if summary doesn't exist
        ai_summary = {
            '2018': {'overall': {}},
            '2024': {'overall': {}}
        }
        
        for year in ['2018', '2024']:
            if not processed_data[year]:
                continue
                
            # Get AI relevance scores for objectives
            objectives = processed_data[year]['processed_objectives']
            
            # Calculate aggregate scores
            category_scores = {}
            for category in ['computational_thinking', 'mathematical_reasoning', 
                            'pattern_recognition', 'data_concepts']:
                scores = [obj['ai_relevance'].get(category, 0) for obj in objectives]
                category_scores[f'{category}_avg'] = np.mean(scores) if scores else 0
                category_scores[f'{category}_total'] = sum(scores)
            
            ai_summary[year] = category_scores
    
    # Create a DataFrame for easier comparison
    ai_relevance_df = pd.DataFrame({
        '2018': ai_summary['2018'],
        '2024': ai_summary['2024']
    }).T
    
    return ai_relevance_df

# Analyze verb usage
def analyze_verb_usage(processed_data):
    """Analyze verb usage to understand cognitive demands."""
    # Load verb analysis if it exists
    try:
        verb_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'verb_analysis.csv'), index_col=0)
    except FileNotFoundError:
        # Extract verbs from processed data
        verbs_2018 = []
        verbs_2024 = []
        
        for year, year_label in [('2018', verbs_2018), ('2024', verbs_2024)]:
            if not processed_data[year]:
                continue
                
            # Get verbs from each objective
            for obj in processed_data[year]['processed_objectives']:
                for token in obj['spacy_tokens']:
                    if token['pos'] == 'VERB':
                        year_label.append(token['lemma'])
        
        # Count frequencies
        verbs_2018_freq = Counter(verbs_2018)
        verbs_2024_freq = Counter(verbs_2024)
        
        # Create DataFrame
        all_verbs = set(verbs_2018_freq.keys()) | set(verbs_2024_freq.keys())
        verb_df = pd.DataFrame(index=all_verbs, columns=['2018', '2024'])
        
        for verb in all_verbs:
            verb_df.loc[verb, '2018'] = verbs_2018_freq.get(verb, 0)
            verb_df.loc[verb, '2024'] = verbs_2024_freq.get(verb, 0)
        
        # Sort by total
        verb_df['total'] = verb_df['2018'] + verb_df['2024']
        verb_df = verb_df.sort_values('total', ascending=False)
    
    # Classify verbs by cognitive level (simplified Bloom's taxonomy)
    # This would require a more sophisticated approach in a real implementation
    bloom_categories = {
        'remember': ['tanımla', 'belirle', 'listele', 'hatırla', 'göster'],
        'understand': ['açıkla', 'özetle', 'yorumla', 'örnek', 'sınıflandır'],
        'apply': ['uygula', 'hesapla', 'çöz', 'göster', 'kullan'],
        'analyze': ['analiz', 'karşılaştır', 'incele', 'ayırt', 'test'],
        'evaluate': ['değerlendir', 'eleştir', 'savun', 'yargıla', 'seç'],
        'create': ['oluştur', 'tasarla', 'geliştir', 'planla', 'üret']
    }
    
    # Add Bloom's taxonomy classification
    for category, verbs in bloom_categories.items():
        verb_df[f'bloom_{category}'] = verb_df.index.isin(verbs).astype(int)
    
    return verb_df

# Analyze objective complexity
def analyze_objective_complexity(processed_data):
    """Analyze the complexity of learning objectives."""
    complexity_metrics = {
        '2018': [],
        '2024': []
    }
    
    for year in ['2018', '2024']:
        if not processed_data[year]:
            continue
            
        # Get objectives
        objectives = processed_data[year]['processed_objectives']
        
        for obj in objectives:
            # Count tokens, excluding punctuation
            token_count = sum(1 for token in obj['spacy_tokens'] 
                              if not token['is_punctuation'])
            
            # Count unique tokens
            unique_tokens = len(set(token['text'].lower() for token in obj['spacy_tokens'] 
                                    if not token['is_punctuation']))
            
            # Calculate average token length
            avg_token_length = np.mean([len(token['text']) for token in obj['spacy_tokens'] 
                                        if not token['is_punctuation']])
            
            # Count sentences
            sentence_count = len(obj['sentences'])
            
            # Store metrics
            complexity_metrics[year].append({
                'section': obj['section'],
                'item': obj['item'],
                'token_count': token_count,
                'unique_tokens': unique_tokens,
                'lexical_diversity': unique_tokens / token_count if token_count > 0 else 0,
                'avg_token_length': avg_token_length,
                'sentence_count': sentence_count,
                'ai_relevance_score': obj['ai_relevance_score']
            })
    
    # Convert to DataFrames
    complexity_dfs = {}
    for year in ['2018', '2024']:
        if complexity_metrics[year]:
            complexity_dfs[year] = pd.DataFrame(complexity_metrics[year])
    
    return complexity_dfs

# Visualize comparative data
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
            if metric in stats['2018'] and metric in stats['2024']:
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
    
    # AI relevance comparison
    if 'ai_relevance' in data_dict:
        ai_df = data_dict['ai_relevance']
        
        # Plot AI relevance categories
        if not ai_df.empty:
            # Reorganize data for plotting
            categories = ['computational_thinking_total', 'mathematical_reasoning_total',
                         'pattern_recognition_total', 'data_concepts_total']
            
            # Filter to categories that exist
            existing_categories = [cat for cat in categories if cat in ai_df.columns]
            
            if existing_categories:
                ai_plot_df = ai_df[existing_categories]
                
                plt.figure(figsize=(12, 6))
                ai_plot_df.plot(kind='bar')
                plt.title('AI Relevance Category Comparison')
                plt.ylabel('Score')
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURES_DIR, 'ai_relevance_comparison.png'), dpi=300)
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
        
        # Compare average complexity metrics
        metrics = ['token_count', 'unique_tokens', 'lexical_diversity', 
                  'avg_token_length', 'ai_relevance_score']
        
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
            
            # Scatter plot of token count vs. AI relevance
            if 'token_count' in complexity_2018 and 'ai_relevance_score' in complexity_2018:
                plt.figure(figsize=(10, 6))
                plt.scatter(complexity_2018['token_count'], complexity_2018['ai_relevance_score'], 
                          alpha=0.7, label='2018')
                if 'token_count' in complexity_2024 and 'ai_relevance_score' in complexity_2024:
                    plt.scatter(complexity_2024['token_count'], complexity_2024['ai_relevance_score'], 
                              alpha=0.7, label='2024')
                plt.xlabel('Token Count')
                plt.ylabel('AI Relevance Score')
                plt.title('Objective Complexity vs. AI Relevance')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURES_DIR, 'complexity_vs_ai_relevance.png'), dpi=300)
                plt.close()

# Main function
def main():
    """Main function to execute the exploratory data analysis."""
    print("Loading processed data...")
    processed_data = load_processed_data()
    
    if not processed_data:
        print("Error: Could not load processed data.")
        return
    
    print("Computing basic statistics...")
    basic_stats = compute_basic_stats(processed_data)
    
    print("Analyzing term frequencies...")
    term_freq = analyze_term_frequencies(processed_data)
    
    print("Creating word clouds...")
    word_clouds = create_word_clouds(processed_data)
    
    print("Analyzing AI relevance...")
    ai_relevance = analyze_ai_relevance(processed_data)
    
    print("Analyzing verb usage...")
    verb_usage = analyze_verb_usage(processed_data)
    
    print("Analyzing objective complexity...")
    complexity = analyze_objective_complexity(processed_data)
    
    # Collect all results
    results = {
        'basic_stats': basic_stats,
        'term_freq': term_freq,
        'ai_relevance': ai_relevance,
        'verb_usage': verb_usage,
        'complexity': complexity
    }
    
    print("Creating visualizations...")
    plot_comparative_charts(results)
    
    # Save results to CSV files
    print("Saving results...")
    term_freq.to_csv(os.path.join(PROCESSED_DIR, 'term_frequency_analysis.csv'))
    ai_relevance.to_csv(os.path.join(PROCESSED_DIR, 'ai_relevance_analysis.csv'))
    verb_usage.to_csv(os.path.join(PROCESSED_DIR, 'verb_usage_detailed.csv'))
    
    for year in ['2018', '2024']:
        if year in complexity:
            complexity[year].to_csv(os.path.join(PROCESSED_DIR, f'objective_complexity_{year}.csv'))
    
    # Save basic stats as JSON
    with open(os.path.join(PROCESSED_DIR, 'basic_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(basic_stats, f, indent=2)
    
    print(f"Exploratory data analysis complete. Results saved to {PROCESSED_DIR} and {FIGURES_DIR} directories.")
    print(f"Files created in {PROCESSED_DIR}:")
    print("  - term_frequency_analysis.csv")
    print("  - ai_relevance_analysis.csv")
    print("  - verb_usage_detailed.csv")
    print("  - objective_complexity_2018.csv (if 2018 data available)")
    print("  - objective_complexity_2024.csv (if 2024 data available)")
    print("  - basic_stats.json")
    
    print(f"\nVisualization files created in {FIGURES_DIR}:")
    print("  - wordcloud_2018.png (if 2018 data available)")
    print("  - wordcloud_2024.png (if 2024 data available)")
    print("  - basic_metrics_comparison.png")
    print("  - top_increased_terms.png")
    print("  - top_decreased_terms.png")
    print("  - ai_relevance_comparison.png")
    print("  - top_verbs_comparison.png")
    print("  - blooms_taxonomy_distribution.png")
    print("  - objective_complexity_comparison.png")
    print("  - complexity_vs_ai_relevance.png")

if __name__ == "__main__":
    main()