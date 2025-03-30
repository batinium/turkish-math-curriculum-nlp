"""
Turkish Mathematics Curriculum NLP Analysis - Curriculum Gap Analysis
=====================================================================
This script analyzes gaps between 2018 and 2024 mathematics curricula using 
data-driven topic analysis and cognitive complexity assessment.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

# Define paths
PROCESSED_DIR = "processed_data"
FIGURES_DIR = "figures"
MODELS_DIR = "models"

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load processed data
def load_processed_data():
    """Load the preprocessed curriculum data and normalize the keys."""
    try:
        with open(os.path.join(PROCESSED_DIR, 'processed_curriculum_data.pkl'), 'rb') as f:
            raw_data = pickle.load(f)
            
        # Create a normalized structure that maps to the expected '2018' and '2024' keys
        normalized_data = {'2018': None, '2024': None}
        
        # Map the actual keys to the expected keys
        for key, value in raw_data.items():
            if '2018' in key or ('detected_type' in value and '2018' in value['detected_type']):
                normalized_data['2018'] = value
                print(f"Mapped '{key}' to '2018'")
            elif '2024' in key or ('detected_type' in value and '2024' in value['detected_type']):
                normalized_data['2024'] = value
                print(f"Mapped '{key}' to '2024'")
        
        return normalized_data
        
    except FileNotFoundError:
        print(f"Processed data file not found. Run preprocessing script first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_topic_coverage(processed_data):
    """
    Analyze curriculum coverage across topics identified by topic modeling.
    Identifies gaps in topic coverage between 2018 and 2024 curricula.
    """
    # Load topic model data
    topic_coverage = {'2018': {}, '2024': {}}
    topic_keywords = {'2018': {}, '2024': {}}
    
    # Try loading topic models
    for year in ['2018', '2024']:
        # Find any topic model for this year
        model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f'lda_model_{year}_')]
        
        if not model_files:
            print(f"No topic model found for {year}. Skipping topic coverage analysis for this year.")
            continue
            
        # Load the model with highest number of topics
        model_nums = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
        model_path = os.path.join(MODELS_DIR, f'lda_model_{year}_{max(model_nums)}.pkl')
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            
            # Store topic keywords
            for topic_id in range(model.num_topics):
                topic_keywords[year][topic_id] = [word for word, _ in model.show_topic(topic_id, topn=10)]
    
    # Load topic distributions
    for year in ['2018', '2024']:
        topic_dist_path = os.path.join(PROCESSED_DIR, f'topic_distribution_{year}.csv')
        if os.path.exists(topic_dist_path):
            topic_df = pd.read_csv(topic_dist_path)
            
            # Calculate distribution of documents across topics
            topic_counts = topic_df['dominant_topic'].value_counts().sort_index()
            total_docs = len(topic_df)
            
            # Calculate coverage percentages
            if total_docs > 0:
                for topic_id, count in topic_counts.items():
                    coverage_pct = count / total_docs * 100
                    topic_coverage[year][int(topic_id)] = {
                        'count': int(count),
                        'coverage_percentage': float(coverage_pct),
                        'keywords': topic_keywords[year].get(int(topic_id), [])
                    }
    
    # Calculate topic similarity between years
    similarity_matrix = None
    if topic_keywords['2018'] and topic_keywords['2024']:
        # Load similarity matrix if it exists
        similarity_path = os.path.join(PROCESSED_DIR, 'topic_similarity_matrix.npy')
        if os.path.exists(similarity_path):
            similarity_matrix = np.load(similarity_path)
    
    # Identify gaps and emerging topics
    curriculum_gaps = {
        'topic_coverage': topic_coverage,
        'topic_keywords': topic_keywords,
        'missing_topics': {
            '2018': [],  # Topics in 2018 not well-represented in 2024
            '2024': []   # New topics in 2024 not present in 2018
        }
    }
    
    # Find missing and emerging topics using similarity matrix
    if similarity_matrix is not None:
        # Topics in 2018 without good matches in 2024 (missing/reduced)
        for i in range(similarity_matrix.shape[0]):
            if i not in topic_coverage['2018']:
                continue
                
            max_sim = np.max(similarity_matrix[i, :])
            if max_sim < 0.2:  # Low similarity threshold
                best_match = int(np.argmax(similarity_matrix[i, :]))
                curriculum_gaps['missing_topics']['2018'].append({
                    'topic_id': int(i),
                    'keywords': topic_keywords['2018'].get(i, []),
                    'coverage': topic_coverage['2018'].get(i, {}).get('coverage_percentage', 0),
                    'best_match_2024': best_match,
                    'similarity': float(max_sim)
                })
        
        # Topics in 2024 without good matches in 2018 (new/emerging)
        for j in range(similarity_matrix.shape[1]):
            if j not in topic_coverage['2024']:
                continue
                
            max_sim = np.max(similarity_matrix[:, j])
            if max_sim < 0.2:  # Low similarity threshold
                best_match = int(np.argmax(similarity_matrix[:, j]))
                curriculum_gaps['missing_topics']['2024'].append({
                    'topic_id': int(j),
                    'keywords': topic_keywords['2024'].get(j, []),
                    'coverage': topic_coverage['2024'].get(j, {}).get('coverage_percentage', 0),
                    'best_match_2018': best_match,
                    'similarity': float(max_sim)
                })
    
    # Save results
    with open(os.path.join(PROCESSED_DIR, 'topic_coverage_gaps.json'), 'w', encoding='utf-8') as f:
        json.dump(curriculum_gaps, f, ensure_ascii=False, indent=2)
    
    # Create visualizations
    visualize_topic_coverage_gaps(curriculum_gaps)
    
    return curriculum_gaps

def visualize_topic_coverage_gaps(curriculum_gaps):
    """Create visualizations of topic coverage and gaps between curricula."""
    # Topic coverage comparison visualization
    coverage_2018 = {}
    coverage_2024 = {}
    
    for topic_id, details in curriculum_gaps['topic_coverage']['2018'].items():
        coverage_2018[topic_id] = details.get('coverage_percentage', 0)
    
    for topic_id, details in curriculum_gaps['topic_coverage']['2024'].items():
        coverage_2024[topic_id] = details.get('coverage_percentage', 0)
    
    if coverage_2018 and coverage_2024:
        # Create bar chart comparing topics in each year
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 2018 coverage
        sorted_topics_2018 = sorted(coverage_2018.items(), key=lambda x: x[1], reverse=True)
        topic_ids_2018 = [f"Topic {t[0]}" for t in sorted_topics_2018]
        coverage_vals_2018 = [t[1] for t in sorted_topics_2018]
        
        ax1.barh(topic_ids_2018, coverage_vals_2018)
        ax1.set_title('2018 Curriculum Topic Coverage')
        ax1.set_xlabel('Coverage Percentage')
        ax1.set_ylabel('Topic')
        
        # 2024 coverage
        sorted_topics_2024 = sorted(coverage_2024.items(), key=lambda x: x[1], reverse=True)
        topic_ids_2024 = [f"Topic {t[0]}" for t in sorted_topics_2024]
        coverage_vals_2024 = [t[1] for t in sorted_topics_2024]
        
        ax2.barh(topic_ids_2024, coverage_vals_2024)
        ax2.set_title('2024 Curriculum Topic Coverage')
        ax2.set_xlabel('Coverage Percentage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'topic_coverage_comparison.png'), dpi=300)
        plt.close()
    
    # Missing topics visualization (topics present in 2018 but missing in 2024)
    missing_topics = curriculum_gaps['missing_topics']['2018']
    if missing_topics:
        plt.figure(figsize=(12, len(missing_topics) * 0.5 + 2))
        
        # Create labeled bars for each missing topic
        topic_labels = []
        coverage_values = []
        
        for topic in missing_topics:
            # Get top 3 keywords as label
            keywords = ', '.join(topic.get('keywords', [])[:3])
            topic_labels.append(f"Topic {topic['topic_id']}: {keywords}")
            coverage_values.append(topic.get('coverage', 0))
        
        # Sort by coverage
        sorted_indices = np.argsort(coverage_values)[::-1]
        topic_labels = [topic_labels[i] for i in sorted_indices]
        coverage_values = [coverage_values[i] for i in sorted_indices]
        
        plt.barh(topic_labels, coverage_values)
        plt.title('Topics Present in 2018 but Missing/Reduced in 2024 Curriculum')
        plt.xlabel('Coverage Percentage in 2018')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'missing_topics_2018.png'), dpi=300)
        plt.close()
    
    # Emerging topics visualization (topics present in 2024 but not in 2018)
    emerging_topics = curriculum_gaps['missing_topics']['2024']
    if emerging_topics:
        plt.figure(figsize=(12, len(emerging_topics) * 0.5 + 2))
        
        # Create labeled bars for each emerging topic
        topic_labels = []
        coverage_values = []
        
        for topic in emerging_topics:
            # Get top 3 keywords as label
            keywords = ', '.join(topic.get('keywords', [])[:3])
            topic_labels.append(f"Topic {topic['topic_id']}: {keywords}")
            coverage_values.append(topic.get('coverage', 0))
        
        # Sort by coverage
        sorted_indices = np.argsort(coverage_values)[::-1]
        topic_labels = [topic_labels[i] for i in sorted_indices]
        coverage_values = [coverage_values[i] for i in sorted_indices]
        
        plt.barh(topic_labels, coverage_values)
        plt.title('New/Emerging Topics in 2024 Curriculum (Not Present in 2018)')
        plt.xlabel('Coverage Percentage in 2024')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'emerging_topics_2024.png'), dpi=300)
        plt.close()

def analyze_cognitive_complexity_gaps(processed_data):
    """Analyze gaps in cognitive complexity levels between curricula."""
    # Define cognitive complexity levels (based on Bloom's taxonomy)
    cognitive_levels = {
        'remember': 1,
        'understand': 2,
        'apply': 3,
        'analyze': 4,
        'evaluate': 5,
        'create': 6
    }
    
    # Bloom's taxonomy verbs in Turkish
    bloom_verbs = {
        'remember': ['tanımla', 'listele', 'belirle', 'hatırla', 'bilir', 'tanır', 'tekrarlar', 'isimlendir'],
        'understand': ['açıkla', 'yorumla', 'özetle', 'karşılaştır', 'anlar', 'kavrar', 'örneklendir', 'ifade et'],
        'apply': ['uygula', 'hesapla', 'çöz', 'göster', 'kullan', 'yapar', 'uygular', 'gerçekleştir'],
        'analyze': ['analiz', 'incele', 'ayırt', 'sorgula', 'ayrıştır', 'sınıflandır', 'karşılaştır', 'kategorize et'],
        'evaluate': ['değerlendir', 'savun', 'yargıla', 'eleştir', 'karar ver', 'öner', 'ölç', 'seç'],
        'create': ['geliştir', 'tasarla', 'oluştur', 'üret', 'kur', 'icat et', 'yarat', 'planla']
    }
    
    # Analyze cognitive complexity in curricula
    results = {'2018': {}, '2024': {}}
    objective_complexity = {'2018': [], '2024': []}
    
    for year in ['2018', '2024']:
        if year not in processed_data or not processed_data[year]:
            continue
            
        if 'processed_objectives' not in processed_data[year]:
            continue
            
        # Initialize level counts
        results[year]['level_counts'] = {level: 0 for level in cognitive_levels.keys()}
        
        # Extract verb complexity from objectives
        for obj in processed_data[year]['processed_objectives']:
            if 'spacy_tokens' not in obj:
                continue
                
            # Find verbs
            verbs = [token['lemma'].lower() for token in obj['spacy_tokens'] if token['pos'] == 'VERB']
            
            # Map verbs to complexity levels
            obj_complexity = 0
            matched_level = None
            
            for verb in verbs:
                for level, level_verbs in bloom_verbs.items():
                    if any(bloom_verb in verb or verb in bloom_verb for bloom_verb in level_verbs):
                        level_value = cognitive_levels[level]
                        if level_value > obj_complexity:
                            obj_complexity = level_value
                            matched_level = level
            
            # If we found a match, count it
            if obj_complexity > 0:
                results[year]['level_counts'][matched_level] += 1
                
                # Store objective details
                objective_complexity[year].append({
                    'text': obj.get('cleaned_text', '')[:50] + '...',
                    'complexity': obj_complexity,
                    'complexity_level': matched_level,
                    'section': obj.get('section', '')
                })
        
        # Calculate percentages
        total_matched = sum(results[year]['level_counts'].values())
        if total_matched > 0:
            results[year]['level_percentages'] = {
                level: (count / total_matched * 100) 
                for level, count in results[year]['level_counts'].items()
            }
            
            # Calculate average complexity
            complexity_values = [obj['complexity'] for obj in objective_complexity[year]]
            results[year]['avg_complexity'] = sum(complexity_values) / len(complexity_values) if complexity_values else 0
        else:
            results[year]['level_percentages'] = {level: 0 for level in cognitive_levels.keys()}
            results[year]['avg_complexity'] = 0
    
    # Calculate complexity gaps
    if results['2018'] and results['2024']:
        results['complexity_gaps'] = {}
        
        for level in cognitive_levels.keys():
            pct_2018 = results['2018']['level_percentages'].get(level, 0)
            pct_2024 = results['2024']['level_percentages'].get(level, 0)
            
            results['complexity_gaps'][level] = {
                'percentage_2018': pct_2018,
                'percentage_2024': pct_2024,
                'percentage_change': pct_2024 - pct_2018,
                'relative_change': (pct_2024 / pct_2018 - 1) * 100 if pct_2018 > 0 else float('inf')
            }
    
    # Create visualizations
    visualize_cognitive_complexity(results, cognitive_levels)
    
    # Save results
    with open(os.path.join(PROCESSED_DIR, 'cognitive_complexity_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def visualize_cognitive_complexity(results, cognitive_levels):
    """Create visualizations for cognitive complexity analysis."""
    # Comparison bar chart
    plt.figure(figsize=(14, 8))
    
    # Order levels by cognitive complexity
    ordered_levels = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    x = np.arange(len(ordered_levels))
    width = 0.35
    
    # Get percentages for each year
    if 'level_percentages' in results['2018'] and 'level_percentages' in results['2024']:
        values_2018 = [results['2018']['level_percentages'].get(level, 0) for level in ordered_levels]
        values_2024 = [results['2024']['level_percentages'].get(level, 0) for level in ordered_levels]
        
        plt.bar(x - width/2, values_2018, width, label='2018 Curriculum')
        plt.bar(x + width/2, values_2024, width, label='2024 Curriculum')
        
        plt.ylabel('Percentage of Objectives')
        plt.title('Cognitive Complexity Distribution by Curriculum')
        plt.xticks(x, ordered_levels)
        plt.legend()
        
        # Add value labels
        for i, v in enumerate(values_2018):
            plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center')
        for i, v in enumerate(values_2024):
            plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add complexity level labels
        for i, level in enumerate(['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']):
            plt.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'cognitive_complexity_comparison.png'), dpi=300)
        plt.close()
    
    # Complexity change visualization
    if 'complexity_gaps' in results:
        # Create visualization of percentage changes
        plt.figure(figsize=(12, 8))
        
        percentage_changes = [results['complexity_gaps'][level]['percentage_change'] for level in ordered_levels]
        
        bars = plt.bar(ordered_levels, percentage_changes)
        
        # Color bars based on positive/negative values
        for i, bar in enumerate(bars):
            if percentage_changes[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.ylabel('Percentage Point Change (2024 - 2018)')
        plt.title('Change in Cognitive Complexity Distribution (2018 to 2024)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(percentage_changes):
            plt.text(i, v + 0.5 if v >= 0 else v - 2, f"{v:+.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'cognitive_complexity_changes.png'), dpi=300)
        plt.close()
        
        # Create average complexity comparison
        if 'avg_complexity' in results['2018'] and 'avg_complexity' in results['2024']:
            plt.figure(figsize=(8, 6))
            
            avg_values = [results['2018']['avg_complexity'], results['2024']['avg_complexity']]
            plt.bar(['2018 Curriculum', '2024 Curriculum'], avg_values)
            
            plt.ylabel('Average Cognitive Complexity (1-6)')
            plt.title('Average Cognitive Complexity Comparison')
            
            # Add value labels
            for i, v in enumerate(avg_values):
                plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
            
            # Add complexity level labels
            for i, level in enumerate(['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']):
                plt.axhline(y=i+1, color='gray', linestyle=':', alpha=0.5)
                plt.text(-0.3, i+1.05, level, fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'average_cognitive_complexity.png'), dpi=300)
            plt.close()

def visualize_curriculum_evolution_network(processed_data):
    """Create a network visualization showing the evolution of curriculum topics."""
    # Load topic similarity data if it exists
    topics_comparison_path = os.path.join(PROCESSED_DIR, 'topic_keywords.json')
    similarity_matrix_path = os.path.join(PROCESSED_DIR, 'topic_similarity_matrix.npy')
    
    if not os.path.exists(topics_comparison_path) or not os.path.exists(similarity_matrix_path):
        print("Topic comparison data not found. Skipping curriculum evolution network visualization.")
        return
    
    with open(topics_comparison_path, 'r', encoding='utf-8') as f:
        topics_comparison = json.load(f)
    
    similarity_matrix = np.load(similarity_matrix_path)
    
    # Create a network graph
    G = nx.Graph()
    
    # Add nodes for each topic in both curricula
    for topic_id, keywords in topics_comparison['2018'].items():
        G.add_node(f"2018 Topic {topic_id}", year='2018', keywords=keywords[:5], size=10)
    
    for topic_id, keywords in topics_comparison['2024'].items():
        G.add_node(f"2024 Topic {topic_id}", year='2024', keywords=keywords[:5], size=10)
    
    # Add edges between similar topics (with threshold)
    similarity_threshold = 0.2
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            similarity = similarity_matrix[i, j]
            if similarity >= similarity_threshold:
                G.add_edge(f"2018 Topic {i}", f"2024 Topic {j}", weight=similarity)
    
    # Calculate node positions (using spring layout for better visualization)
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Get node attributes
    node_colors = ['blue' if G.nodes[node]['year'] == '2018' else 'red' for node in G.nodes]
    node_sizes = [G.nodes[node]['size'] * 100 for node in G.nodes]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges with varying thickness based on similarity
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    
    # Create custom labels showing top keywords
    custom_labels = {}
    for node in G.nodes:
        keywords = ', '.join(G.nodes[node]['keywords'][:3])
        custom_labels[node] = f"{node}\n({keywords})"
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=custom_labels, font_size=8, font_family='sans-serif')
    
    # Add title and legend
    plt.title("Curriculum Topic Evolution Network (2018 to 2024)", fontsize=16)
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='2018 Topics'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='2024 Topics'),
        plt.Line2D([0], [0], color='gray', lw=2, label='Topic Similarity')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'curriculum_evolution_network.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save network data
    with open(os.path.join(PROCESSED_DIR, 'curriculum_evolution_network.pkl'), 'wb') as f:
        pickle.dump((G, pos), f)

def generate_analysis_summary(topic_gaps, cognitive_gaps):
    """Generate a comprehensive text summary of the curriculum analysis results."""
    summary = []
    
    # Create a header
    summary.append("# Turkish Mathematics Curriculum Analysis (2018 vs 2024)")
    summary.append("=" * 60)
    summary.append("")
    
    # Topic coverage summary
    summary.append("## Topic Coverage Analysis")
    summary.append("-" * 30)
    
    # Summarize emerging topics (new in 2024)
    if 'missing_topics' in topic_gaps and '2024' in topic_gaps['missing_topics']:
        emerging_topics = topic_gaps['missing_topics']['2024']
        summary.append(f"\n### Emerging Topics in 2024 Curriculum ({len(emerging_topics)} topics)")
        
        if emerging_topics:
            for i, topic in enumerate(emerging_topics):
                keywords = ', '.join(topic.get('keywords', [])[:5])
                coverage = topic.get('coverage', 0)
                summary.append(f"\n{i+1}. **Topic {topic['topic_id']}** (Coverage: {coverage:.1f}%)")
                summary.append(f"   Keywords: {keywords}")
                summary.append(f"   Similarity to best matching 2018 topic: {topic['similarity']:.2f}")
        else:
            summary.append("\nNo significant emerging topics identified.")
    
    # Summarize fading topics (present in 2018 but missing in 2024)
    if 'missing_topics' in topic_gaps and '2018' in topic_gaps['missing_topics']:
        fading_topics = topic_gaps['missing_topics']['2018']
        summary.append(f"\n### Fading Topics from 2018 Curriculum ({len(fading_topics)} topics)")
        
        if fading_topics:
            for i, topic in enumerate(fading_topics):
                keywords = ', '.join(topic.get('keywords', [])[:5])
                coverage = topic.get('coverage', 0)
                summary.append(f"\n{i+1}. **Topic {topic['topic_id']}** (Coverage: {coverage:.1f}%)")
                summary.append(f"   Keywords: {keywords}")
                summary.append(f"   Similarity to best matching 2024 topic: {topic['similarity']:.2f}")
        else:
            summary.append("\nNo significant fading topics identified.")
    
    # Cognitive complexity summary
    summary.append("\n\n## Cognitive Complexity Analysis")
    summary.append("-" * 30)
    
    # Compare average cognitive complexity
    if 'avg_complexity' in cognitive_gaps.get('2018', {}) and 'avg_complexity' in cognitive_gaps.get('2024', {}):
        avg_2018 = cognitive_gaps['2018']['avg_complexity']
        avg_2024 = cognitive_gaps['2024']['avg_complexity']
        diff = avg_2024 - avg_2018
        
        summary.append(f"\nAverage Cognitive Complexity:")
        summary.append(f"- 2018 Curriculum: {avg_2018:.2f}/6.0")
        summary.append(f"- 2024 Curriculum: {avg_2024:.2f}/6.0")
        summary.append(f"- Change: {diff:+.2f} ({(diff/avg_2018)*100:+.1f}%)")
        
        if diff > 0:
            summary.append("\nThe 2024 curriculum shows a higher average cognitive complexity, indicating a shift toward higher-order thinking skills.")
        elif diff < 0:
            summary.append("\nThe 2024 curriculum shows a lower average cognitive complexity, potentially focusing more on fundamental concepts.")
        else:
            summary.append("\nThe average cognitive complexity remains similar between the two curricula.")
    
    # Cognitive level distribution
    if 'complexity_gaps' in cognitive_gaps:
        summary.append("\n### Cognitive Level Distribution (Bloom's Taxonomy)")
        summary.append("\n| Cognitive Level | 2018 Curriculum | 2024 Curriculum | Change |")
        summary.append("| -------------- | --------------- | --------------- | ------ |")
        
        for level in ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']:
            if level in cognitive_gaps['complexity_gaps']:
                pct_2018 = cognitive_gaps['complexity_gaps'][level]['percentage_2018']
                pct_2024 = cognitive_gaps['complexity_gaps'][level]['percentage_2024']
                change = cognitive_gaps['complexity_gaps'][level]['percentage_change']
                
                summary.append(f"| {level.capitalize()} | {pct_2018:.1f}% | {pct_2024:.1f}% | {change:+.1f}% |")
        
        # Interpret the changes
        summary.append("\n#### Key Changes in Cognitive Demand:")
        
        # Find biggest increases and decreases
        changes = [(level, cognitive_gaps['complexity_gaps'][level]['percentage_change']) 
                 for level in cognitive_gaps['complexity_gaps']]
        increases = sorted([c for c in changes if c[1] > 0], key=lambda x: x[1], reverse=True)
        decreases = sorted([c for c in changes if c[1] < 0], key=lambda x: x[1])
        
        if increases:
            summary.append("\nSignificant increases in:")
            for level, change in increases[:2]:  # Top 2 increases
                summary.append(f"- **{level.capitalize()}** level objectives: {change:+.1f}%")
        
        if decreases:
            summary.append("\nSignificant decreases in:")
            for level, change in decreases[:2]:  # Top 2 decreases
                summary.append(f"- **{level.capitalize()}** level objectives: {change:.1f}%")
    
    # Overall interpretation
    summary.append("\n\n## Summary of Curriculum Evolution")
    summary.append("-" * 30)
    
    # Interpret emerging topic themes
    if 'missing_topics' in topic_gaps and '2024' in topic_gaps['missing_topics'] and topic_gaps['missing_topics']['2024']:
        # Collect keywords from emerging topics
        all_keywords = []
        for topic in topic_gaps['missing_topics']['2024']:
            all_keywords.extend(topic.get('keywords', [])[:5])
        
        # Count keyword frequency
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(10)
        
        summary.append("\n### Key Themes in New Content (2024)")
        summary.append("\nBased on keyword analysis, the following themes appear in the new curriculum content:")
        for keyword, count in top_keywords:
            summary.append(f"- **{keyword}** (appears {count} times)")
    
    # Interpret cognitive shift
    if 'complexity_gaps' in cognitive_gaps:
        higher_order = sum(cognitive_gaps['complexity_gaps'][level]['percentage_change'] 
                          for level in ['analyze', 'evaluate', 'create'])
        lower_order = sum(cognitive_gaps['complexity_gaps'][level]['percentage_change'] 
                         for level in ['remember', 'understand', 'apply'])
        
        summary.append("\n### Shift in Cognitive Demands")
        if higher_order > 0 and lower_order < 0:
            summary.append("\nThe 2024 curriculum shows a clear shift toward higher-order thinking skills (analyze, evaluate, create) and away from lower-order skills (remember, understand, apply).")
        elif higher_order < 0 and lower_order > 0:
            summary.append("\nThe 2024 curriculum emphasizes fundamental skills (remember, understand, apply) more than the 2018 curriculum, with less focus on higher-order thinking.")
        else:
            summary.append("\nThe distribution of cognitive demands shows a mixed pattern of changes, without a clear direction toward higher or lower-order thinking skills.")
    
    # Conclusion with observations about overall curriculum evolution
    summary.append("\n## Conclusion")
    summary.append("-" * 30)
    summary.append("\nThe analysis reveals several important shifts in the Turkish Mathematics Curriculum between 2018 and 2024:")
    
    # Generate 3-4 key observations based on the analysis
    observations = []
    
    # Topic-related observations
    if 'missing_topics' in topic_gaps:
        if len(topic_gaps['missing_topics'].get('2024', [])) > len(topic_gaps['missing_topics'].get('2018', [])):
            observations.append("The 2024 curriculum introduces more new topics than it removes from the 2018 version, suggesting curriculum expansion.")
        elif len(topic_gaps['missing_topics'].get('2024', [])) < len(topic_gaps['missing_topics'].get('2018', [])):
            observations.append("The 2024 curriculum removes more topics than it adds compared to the 2018 version, suggesting curriculum streamlining.")
    
    # Cognitive complexity observations
    if 'avg_complexity' in cognitive_gaps.get('2018', {}) and 'avg_complexity' in cognitive_gaps.get('2024', {}):
        avg_2018 = cognitive_gaps['2018']['avg_complexity']
        avg_2024 = cognitive_gaps['2024']['avg_complexity']
        diff = avg_2024 - avg_2018
        
        if diff > 0.2:
            observations.append(f"There is a substantial increase ({diff:+.2f}) in cognitive complexity, indicating a more challenging curriculum that emphasizes higher-order thinking skills.")
        elif diff < -0.2:
            observations.append(f"There is a notable decrease ({diff:.2f}) in cognitive complexity, suggesting a focus on making the curriculum more accessible or foundational.")
    
    # Add observations to the summary
    for i, obs in enumerate(observations):
        summary.append(f"\n{i+1}. {obs}")
    
    # Final note
    summary.append("\n\nThis analysis was conducted using unsupervised topic modeling and cognitive complexity assessment techniques, allowing for an objective comparison of the curriculum evolution.")
    
    return "\n".join(summary)


def generate_gap_analysis_structured_output(topic_gaps, cognitive_gaps):
    """
    Generate a comprehensive structured output (JSON-compatible) with all curriculum gap analysis results
    for easy parsing by language models or other applications.
    """
    output = {
        "analysis_type": "Turkish Mathematics Curriculum Gap Analysis",
        "comparison": "2018 vs 2024",
        "topic_coverage_analysis": {},
        "cognitive_complexity_analysis": {},
        "curriculum_evolution": {},
        "key_insights": []
    }
    
    # Topic Coverage Analysis
    if topic_gaps:
        # Extract topic coverage by year
        topic_coverage = {}
        for year in ['2018', '2024']:
            if year in topic_gaps.get('topic_coverage', {}):
                year_coverage = {}
                for topic_id, details in topic_gaps['topic_coverage'][year].items():
                    # Convert to appropriate types for JSON
                    year_coverage[str(topic_id)] = {
                        "count": details.get('count', 0),
                        "coverage_percentage": float(details.get('coverage_percentage', 0)),
                        "keywords": details.get('keywords', [])
                    }
                topic_coverage[year] = year_coverage
        
        # Extract topic gaps (missing/emerging topics)
        missing_topics = {}
        for year in ['2018', '2024']:
            if year in topic_gaps.get('missing_topics', {}):
                # Normalize the data for JSON
                missing_topics[year] = []
                for topic in topic_gaps['missing_topics'][year]:
                    missing_topics[year].append({
                        "topic_id": int(topic.get('topic_id', 0)),
                        "keywords": topic.get('keywords', []),
                        "coverage": float(topic.get('coverage', 0)),
                        f"best_match_{2024 if year == '2018' else 2018}": int(topic.get(f'best_match_{2024 if year == "2018" else 2018}', 0)),
                        "similarity": float(topic.get('similarity', 0))
                    })
        
        # Calculate overall topic shift metrics
        topic_shift_metrics = {}
        if '2018' in missing_topics and '2024' in missing_topics:
            # Topics removed vs. added
            topics_removed = len(missing_topics['2018'])
            topics_added = len(missing_topics['2024'])
            
            # Quantify the overall topic turnover
            total_topics = len(topic_gaps.get('topic_keywords', {}).get('2018', {})) + len(topic_gaps.get('topic_keywords', {}).get('2024', {}))
            topic_turnover_pct = ((topics_removed + topics_added) / total_topics * 100) if total_topics > 0 else 0
            
            topic_shift_metrics = {
                "topics_removed": topics_removed,
                "topics_added": topics_added,
                "net_topic_change": topics_added - topics_removed,
                "topic_turnover_percentage": float(topic_turnover_pct)
            }
        
        output["topic_coverage_analysis"] = {
            "topic_coverage_by_year": topic_coverage,
            "missing_topics": missing_topics,
            "topic_shift_metrics": topic_shift_metrics
        }
    
    # Cognitive Complexity Analysis
    if cognitive_gaps:
        cognitive_analysis = {}
        
        # Extract level counts and percentages for each year
        for year in ['2018', '2024']:
            if year in cognitive_gaps:
                cognitive_analysis[year] = {}
                
                # Copy level counts
                if 'level_counts' in cognitive_gaps[year]:
                    cognitive_analysis[year]['level_counts'] = cognitive_gaps[year]['level_counts']
                
                # Copy level percentages
                if 'level_percentages' in cognitive_gaps[year]:
                    cognitive_analysis[year]['level_percentages'] = {
                        level: float(pct) for level, pct in cognitive_gaps[year]['level_percentages'].items()
                    }
                
                # Copy average complexity
                if 'avg_complexity' in cognitive_gaps[year]:
                    cognitive_analysis[year]['avg_complexity'] = float(cognitive_gaps[year]['avg_complexity'])
        
        # Extract complexity gaps
        complexity_gaps = {}
        if 'complexity_gaps' in cognitive_gaps:
            for level, details in cognitive_gaps['complexity_gaps'].items():
                complexity_gaps[level] = {
                    "percentage_2018": float(details.get('percentage_2018', 0)),
                    "percentage_2024": float(details.get('percentage_2024', 0)),
                    "percentage_change": float(details.get('percentage_change', 0)),
                    "relative_change": float(details.get('relative_change', 0)) if not isinstance(details.get('relative_change'), float) or not math.isinf(details.get('relative_change', 0)) else None
                }
        
        # Calculate higher-order vs. lower-order thinking shifts
        cognitive_shift_metrics = {}
        if complexity_gaps:
            # Higher-order thinking (analyze, evaluate, create)
            hot_2018 = sum(complexity_gaps[level]['percentage_2018'] for level in ['analyze', 'evaluate', 'create'] if level in complexity_gaps)
            hot_2024 = sum(complexity_gaps[level]['percentage_2024'] for level in ['analyze', 'evaluate', 'create'] if level in complexity_gaps)
            hot_change = hot_2024 - hot_2018
            
            # Lower-order thinking (remember, understand, apply)
            lot_2018 = sum(complexity_gaps[level]['percentage_2018'] for level in ['remember', 'understand', 'apply'] if level in complexity_gaps)
            lot_2024 = sum(complexity_gaps[level]['percentage_2024'] for level in ['remember', 'understand', 'apply'] if level in complexity_gaps)
            lot_change = lot_2024 - lot_2018
            
            cognitive_shift_metrics = {
                "higher_order_thinking": {
                    "percentage_2018": float(hot_2018),
                    "percentage_2024": float(hot_2024),
                    "change": float(hot_change)
                },
                "lower_order_thinking": {
                    "percentage_2018": float(lot_2018),
                    "percentage_2024": float(lot_2024),
                    "change": float(lot_change)
                },
                "cognitive_shift_direction": "higher" if hot_change > 0 and lot_change < 0 else 
                                           "lower" if hot_change < 0 and lot_change > 0 else 
                                           "mixed"
            }
            
            # Add average complexity change if available
            if 'avg_complexity' in cognitive_gaps.get('2018', {}) and 'avg_complexity' in cognitive_gaps.get('2024', {}):
                avg_2018 = cognitive_gaps['2018']['avg_complexity']
                avg_2024 = cognitive_gaps['2024']['avg_complexity']
                avg_change = avg_2024 - avg_2018
                avg_change_pct = (avg_change / avg_2018 * 100) if avg_2018 > 0 else None
                
                cognitive_shift_metrics["average_complexity"] = {
                    "value_2018": float(avg_2018),
                    "value_2024": float(avg_2024),
                    "absolute_change": float(avg_change),
                    "percentage_change": float(avg_change_pct) if avg_change_pct is not None else None
                }
        
        output["cognitive_complexity_analysis"] = {
            "by_year": cognitive_analysis,
            "complexity_gaps": complexity_gaps,
            "cognitive_shift_metrics": cognitive_shift_metrics
        }
    
    # Generate key insights
    insights = []
    
    # Topic coverage insights
    if topic_gaps and 'missing_topics' in topic_gaps:
        # Insight about emerging topics
        if '2024' in topic_gaps['missing_topics'] and topic_gaps['missing_topics']['2024']:
            emerging_topics = topic_gaps['missing_topics']['2024']
            top_keywords = []
            
            # Collect keywords from emerging topics
            for topic in emerging_topics:
                top_keywords.extend(topic.get('keywords', [])[:3])
            
            # Count keyword frequency to identify themes
            from collections import Counter
            keyword_counts = Counter(top_keywords)
            top_keywords = [kw for kw, _ in keyword_counts.most_common(5)]
            
            insight = {
                "type": "emerging_topics",
                "count": len(emerging_topics),
                "top_themes": top_keywords,
                "description": f"The 2024 curriculum introduces {len(emerging_topics)} new topics not present in the 2018 version, with key themes including: {', '.join(top_keywords)}"
            }
            insights.append(insight)
        
        # Insight about fading topics
        if '2018' in topic_gaps['missing_topics'] and topic_gaps['missing_topics']['2018']:
            fading_topics = topic_gaps['missing_topics']['2018']
            top_keywords = []
            
            # Collect keywords from fading topics
            for topic in fading_topics:
                top_keywords.extend(topic.get('keywords', [])[:3])
            
            # Count keyword frequency to identify themes
            from collections import Counter
            keyword_counts = Counter(top_keywords)
            top_keywords = [kw for kw, _ in keyword_counts.most_common(5)]
            
            insight = {
                "type": "fading_topics",
                "count": len(fading_topics),
                "top_themes": top_keywords,
                "description": f"The 2024 curriculum removes or significantly reduces {len(fading_topics)} topics from the 2018 version, with key themes including: {', '.join(top_keywords)}"
            }
            insights.append(insight)
    
    # Cognitive complexity insights
    if cognitive_gaps and 'complexity_gaps' in cognitive_gaps:
        # Find biggest changes in cognitive levels
        changes = [(level, cognitive_gaps['complexity_gaps'][level]['percentage_change']) 
                  for level in cognitive_gaps['complexity_gaps']]
        
        # Identify significant increases
        increases = sorted([c for c in changes if c[1] > 3], key=lambda x: x[1], reverse=True)
        if increases:
            top_increases = increases[:2]  # Top 2 increases
            insight = {
                "type": "cognitive_level_increase",
                "levels": [level for level, _ in top_increases],
                "changes": [float(change) for _, change in top_increases],
                "description": f"The 2024 curriculum shows a significant increase in {'and '.join([f'{level} ({change:+.1f}%)' for level, change in top_increases])} level objectives."
            }
            insights.append(insight)
        
        # Identify significant decreases
        decreases = sorted([c for c in changes if c[1] < -3], key=lambda x: x[1])
        if decreases:
            top_decreases = decreases[:2]  # Top 2 decreases
            insight = {
                "type": "cognitive_level_decrease",
                "levels": [level for level, _ in top_decreases],
                "changes": [float(change) for _, change in top_decreases],
                "description": f"The 2024 curriculum shows a significant decrease in {'and '.join([f'{level} ({change:.1f}%)' for level, change in top_decreases])} level objectives."
            }
            insights.append(insight)
        
        # Insight about overall cognitive direction
        if 'cognitive_shift_metrics' in output["cognitive_complexity_analysis"]:
            shift_metrics = output["cognitive_complexity_analysis"]["cognitive_shift_metrics"]
            
            if shift_metrics.get("cognitive_shift_direction") == "higher":
                insight = {
                    "type": "cognitive_shift",
                    "direction": "higher",
                    "hot_change": float(shift_metrics["higher_order_thinking"]["change"]),
                    "lot_change": float(shift_metrics["lower_order_thinking"]["change"]),
                    "description": "The 2024 curriculum shows a clear shift toward higher-order thinking skills (analyze, evaluate, create) and away from lower-order skills (remember, understand, apply)."
                }
                insights.append(insight)
            elif shift_metrics.get("cognitive_shift_direction") == "lower":
                insight = {
                    "type": "cognitive_shift",
                    "direction": "lower",
                    "hot_change": float(shift_metrics["higher_order_thinking"]["change"]),
                    "lot_change": float(shift_metrics["lower_order_thinking"]["change"]),
                    "description": "The 2024 curriculum emphasizes fundamental skills (remember, understand, apply) more than the 2018 curriculum, with less focus on higher-order thinking."
                }
                insights.append(insight)
    
    # Add overall curriculum evolution insight
    if topic_gaps and cognitive_gaps:
        # Determine if there's expansion, reduction, or transformation
        topics_removed = len(topic_gaps['missing_topics'].get('2018', []))
        topics_added = len(topic_gaps['missing_topics'].get('2024', []))
        
        # Get cognitive complexity change if available
        complexity_change = 0
        if ('avg_complexity' in cognitive_gaps.get('2018', {}) and 
            'avg_complexity' in cognitive_gaps.get('2024', {})):
            complexity_change = cognitive_gaps['2024']['avg_complexity'] - cognitive_gaps['2018']['avg_complexity']
        
        # Determine curriculum evolution pattern
        if topics_added > topics_removed and complexity_change > 0.1:
            evolution = "expansion with increased complexity"
        elif topics_added > topics_removed and complexity_change < -0.1:
            evolution = "expansion with simplified demands"
        elif topics_added < topics_removed and complexity_change > 0.1:
            evolution = "streamlined but more cognitively demanding"
        elif topics_added < topics_removed and complexity_change < -0.1:
            evolution = "streamlined and simplified"
        elif abs(topics_added - topics_removed) <= 2 and abs(complexity_change) <= 0.1:
            evolution = "refined with minimal structural changes"
        else:
            evolution = "transformed with mixed patterns of change"
        
        insight = {
            "type": "curriculum_evolution",
            "topics_added": topics_added,
            "topics_removed": topics_removed,
            "complexity_change": float(complexity_change),
            "evolution_pattern": evolution,
            "description": f"The Turkish Mathematics Curriculum has undergone {evolution} from 2018 to 2024, with {topics_added} new topics added, {topics_removed} topics removed, and a {complexity_change:+.2f} change in average cognitive complexity."
        }
        insights.append(insight)
    
    output["key_insights"] = insights
    
    return output

def export_gap_analysis_structured_data(topic_gaps, cognitive_gaps):
    """Export all gap analysis data in a structured format (JSON)."""
    print("Generating structured gap analysis data export...")
    
    import math  # Required for handling infinity values in JSON
    
    structured_data = generate_gap_analysis_structured_output(
        topic_gaps,
        cognitive_gaps
    )
    
    # Save to file
    import json
    structured_data_path = os.path.join(PROCESSED_DIR, 'curriculum_gap_analysis_data.json')
    with open(structured_data_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    
    print(f"Structured gap analysis data export saved to: {structured_data_path}")
    return structured_data_path


def main():
    """Main function to execute the curriculum gap analysis."""
    print("Loading processed data...")
    processed_data = load_processed_data()
    
    if not processed_data:
        print("Error: Could not load processed data.")
        return
    
    print("\nAnalyzing topic coverage gaps...")
    topic_gaps = analyze_topic_coverage(processed_data)
    
    print("Analyzing cognitive complexity gaps...")
    cognitive_gaps = analyze_cognitive_complexity_gaps(processed_data)
    
    print("Creating curriculum evolution network visualization...")
    visualize_curriculum_evolution_network(processed_data)
    
    print("Generating text summary of analysis results...")
    summary_text = generate_analysis_summary(topic_gaps, cognitive_gaps)
    
    # Save the summary to a text file
    summary_path = os.path.join(PROCESSED_DIR, 'curriculum_analysis_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"Analysis summary saved to: {summary_path}")
    
    # Generate and export structured data for LLM processing
    print("\nGenerating structured data for LLM processing...")
    structured_data_path = export_gap_analysis_structured_data(
        topic_gaps,
        cognitive_gaps
    )
    
    print("\nGap analysis complete. Results saved to processed_data and figures directories.")
    print(f"Files created in {PROCESSED_DIR}:")
    print("  - topic_coverage_gaps.json")
    print("  - cognitive_complexity_analysis.json")
    print("  - curriculum_evolution_network.pkl")
    print("  - curriculum_analysis_summary.txt")
    print(f"  - {os.path.basename(structured_data_path)} (Structured data for LLM processing)")
    
    print(f"\nVisualization files created in {FIGURES_DIR}:")
    print("  - topic_coverage_comparison.png")
    print("  - missing_topics_2018.png")
    print("  - emerging_topics_2024.png")
    print("  - cognitive_complexity_comparison.png")
    print("  - cognitive_complexity_changes.png")
    print("  - average_cognitive_complexity.png")
    print("  - curriculum_evolution_network.png")
    
    

if __name__ == "__main__":
    main()