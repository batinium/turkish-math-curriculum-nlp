#curriculum-topic-modeling.py
"""
Turkish Mathematics Curriculum NLP Analysis - Topic Modeling and Semantic Analysis
===============================================================================
This script performs advanced NLP analysis on the preprocessed curriculum data,
including:
1. Topic modeling to identify hidden themes
2. Semantic similarity analysis to compare curricula
3. Network analysis of concept relationships
4. Classification of curriculum elements
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# For NLP processing
import nltk
from nltk.corpus import stopwords
import spacy

# Set GPU memory growth to avoid memory issues
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# For topic modeling
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess

# For visualization
import pyLDAvis
import pyLDAvis.gensim_models
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap


# Import at the beginning of your topic-modeling.py
from custom_stopwords import get_turkish_curriculum_stopwords, get_extended_stopwords


# Configure visualizations
plt.style.use('fivethirtyeight')
sns.set(style="whitegrid")

# Define paths
PROCESSED_DIR = "processed_data"
FIGURES_DIR = "figures"
MODELS_DIR = "models"

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Check GPU availability
gpu_available = spacy.prefer_gpu()

# Load processed data
def load_processed_data():
    """Load the preprocessed curriculum data."""
    try:
        pickle_path = os.path.join(PROCESSED_DIR, 'processed_curriculum_data.pkl')
        print(f"Loading pickle file from {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Perform some diagnostics on the loaded data
        print(f"Successfully loaded processed data with {len(data)} curricula")
        
        # Check for curricula by year
        curricula_2018 = [name for name, curr in data.items() 
                        if curr and ('2018' in curr.get('detected_type', '') or 
                                    'numeric-based' in curr.get('detected_type', '') or
                                    'chapter-based' in curr.get('detected_type', ''))]
        curricula_2024 = [name for name, curr in data.items() 
                          if curr and '2024' in curr.get('detected_type', '')]
        
        print(f"Found {len(curricula_2018)} curricula from 2018: {curricula_2018}")
        print(f"Found {len(curricula_2024)} curricula from 2024: {curricula_2024}")
        
        # Check for lemmatized text in objectives
        for name, curr in data.items():
            if not curr or 'processed_objectives' not in curr:
                continue
                
            objectives = curr['processed_objectives']
            with_lemmatized = sum(1 for obj in objectives if 'lemmatized_text' in obj)
            total_objectives = len(objectives)
            
            print(f"{name}: {with_lemmatized}/{total_objectives} objectives have lemmatized_text")
        
        return data
    except FileNotFoundError:
        print(f"Processed data file not found. Run preprocessing script first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Prepare data for topic modeling
def prepare_topic_modeling_data(processed_data):
    """Prepare the data for topic modeling by extracting learning objectives."""
    documents = {
        '2018': [],
        '2024': []
    }
    
    metadata = {
        '2018': [],
        '2024': []
    }
    
    # Track skipped items for reporting
    skipped_items = 0
    total_objectives = 0
    
    # First, determine which curriculum corresponds to which year
    curricula_by_year = {'2018': [], '2024': []}
    
    for name, curriculum in processed_data.items():
        if not curriculum:
            continue
            
        # Check if we can determine year from detected_type
        detected_type = curriculum.get('detected_type', '')
        print(f"Detected type for {name}: {detected_type}")
        
        if '2018' in detected_type or detected_type == '2018-style':
            curricula_by_year['2018'].append((name, curriculum))
        elif '2024' in detected_type or detected_type == '2024-style':
            curricula_by_year['2024'].append((name, curriculum))
        else:
            # Try to guess from the name
            if '2018' in name:
                curricula_by_year['2018'].append((name, curriculum))
            elif '2024' in name:
                curricula_by_year['2024'].append((name, curriculum))
            else:
                print(f"Warning: Could not determine year for curriculum '{name}'. Skipping.")
    
    # Process each year's curricula
    for year in ['2018', '2024']:
        if not curricula_by_year[year]:
            print(f"No curricula found for year {year}")
            continue
            
        print(f"Processing {len(curricula_by_year[year])} curricula for year {year}")
        
        for name, curriculum in curricula_by_year[year]:
            if 'processed_objectives' not in curriculum:
                print(f"Warning: No processed objectives found for curriculum '{name}'. Skipping.")
                continue
                
            for obj in curriculum['processed_objectives']:
                total_objectives += 1
                
                # Skip empty or math-symbol-only objectives
                if (obj.get('original_text', '').strip() in ['±', '∈', '∉', '∋', '∌', '∩', '∪', '⊂', '⊃', '⊆', '⊇', 
                                                           '⊄', '⊅', '∧', '∨', '¬', '→', '↔', '∀', '∃', '∄', '∑', 
                                                           '∏', '∫', '∮', '≤', '≥', '≠', '≈', '←', '↑', '↓', '↔'] or
                    (not obj.get('cleaned_text', '').strip() and len(obj.get('original_text', '').strip()) <= 2)):
                    
                    skipped_items += 1
                    continue
                
                # Extract lemmatized text if available
                text = obj.get('lemmatized_text', '')
                
                # Fallback to cleaned_text if lemmatized_text is not available or empty
                if not text.strip():
                    text = obj.get('cleaned_text', '')
                    if text.strip():  # Only print warning if cleaned_text has content
                        print(f"Warning: No lemmatized_text found for an objective in {name}. For object {obj} Using cleaned_text instead.")
                    
                # Skip if still no text
                if not text.strip():
                    skipped_items += 1
                    continue
                
                # Add to documents
                documents[year].append(text)
                
                # Store metadata
                metadata[year].append({
                    'section': obj.get('section', ''),
                    'item': obj.get('item', ''),
                    'ai_relevance_score': obj.get('ai_relevance_score', 0)
                })
    
    print(f"Total objectives: {total_objectives}")
    print(f"Skipped {skipped_items} empty or math-symbol-only objectives ({skipped_items/total_objectives:.1%})")
    
    return documents, metadata

# Define Turkish stopwords
def get_stopwords():
    """Get Turkish stopwords and additional custom stopwords."""
    try:
        turkish_stopwords = set(stopwords.words('turkish'))
    except:
        print("Turkish stopwords not available in NLTK. Using custom list.")
        # Use the basic Turkish stopwords from your module
        turkish_stopwords = get_turkish_curriculum_stopwords()
    
    # Add the custom stopwords from your module
    custom_stopwords = get_extended_stopwords()
    
    return turkish_stopwords.union(custom_stopwords)

# Preprocess text for topic modeling
def preprocess_for_topics(text, stopwords_set, nlp=None):
    """Preprocess text for topic modeling with lemmatization."""
    # Simply split the already-lemmatized text and filter
    tokens = text.split()
    
    # Filter out stopwords and short words
    filtered_tokens = [token.lower() for token in tokens 
                      if token.lower() not in stopwords_set
                      and len(token) > 3]
    
    return filtered_tokens

# Topic modeling function
def run_topic_modeling(documents, year, stopwords_set, num_topics=5):
    """Run LDA topic modeling on the documents."""
    
    model_path = os.path.join(MODELS_DIR, f'lda_model_{year}_{num_topics}.pkl')
    if os.path.exists(model_path):
        print(f"Loading existing topic model for {year} from {model_path}")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    try:
        # Preprocess documents
        processed_docs = []
        for doc in documents:
            processed_docs.append(preprocess_for_topics(doc, stopwords_set))

        
        # Create dictionary
        dictionary = corpora.Dictionary(processed_docs)
        
        # Filter out extreme values
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        
        # Create corpus
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        
        # Build LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            update_every=1,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Calculate model coherence
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=processed_docs, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        print(f"\nCoherence Score for {year} curriculum: {coherence_score}")
        
        # Save model and related data
        model_data = {
            'model': lda_model,
            'corpus': corpus,
            'dictionary': dictionary,
            'processed_docs': processed_docs,
            'coherence_score': coherence_score
        }
        
        # Save visualization
        try:
            vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
            pyLDAvis.save_html(vis_data, os.path.join(FIGURES_DIR, f'topic_model_vis_{year}.html'))
            print(f"Topic model visualization saved to {os.path.join(FIGURES_DIR, f'topic_model_vis_{year}.html')}")
        except Exception as e:
            print(f"Error generating visualization: {e}")
        
        # Save model at the end
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
    except Exception as e:
        print(f"Error running topic modeling: {e}")
        model_data = None
        
    return model_data

# Find optimal number of topics
def find_optimal_topics(documents, year, stopwords_set, start=2, limit=10, step=1):
    """Find the optimal number of topics by coherence score."""
    
    # Check if we've already computed this
    results_path = os.path.join(PROCESSED_DIR, f'optimal_topics_{year}.json')
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                saved_results = json.load(f)
                print(f"Loading pre-computed optimal topics for {year}: {saved_results['optimal_num_topics']}")
                return saved_results['optimal_num_topics'], None
        except Exception as e:
            print(f"Error loading pre-computed optimal topics: {e}")
            # Continue with computation if loading fails
    
    coherence_values = []
    model_list = []
    
    # Preprocess documents - No need for nlp parameter anymore
    processed_docs = [preprocess_for_topics(doc, stopwords_set) for doc in documents]
    
    # Create dictionary
    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    
    # Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    for num_topics in range(start, limit, step):
        print(f"Testing model with {num_topics} topics...")
        
        # Build LDA model
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto'
        )
        
        model_list.append(model)
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=model, 
            texts=processed_docs, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        
        coherence_values.append(coherence_model.get_coherence())
    
    # Find optimal number of topics
    optimal_idx = coherence_values.index(max(coherence_values))
    optimal_num_topics = range(start, limit, step)[optimal_idx]
    
    print(f"Optimal number of topics for {year} curriculum: {optimal_num_topics}")
    
    # Plot coherence scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(start, limit, step), coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title(f"Coherence Scores for Different Topic Numbers ({year})")
    plt.axvline(x=optimal_num_topics, color='r', linestyle='--', label=f'Optimal: {optimal_num_topics}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'topic_coherence_{year}.png'), dpi=300)
    plt.close()
    
    # Save results for future runs
    with open(results_path, 'w') as f:
        json.dump({'optimal_num_topics': optimal_num_topics}, f)
    
    return optimal_num_topics, model_list[optimal_idx]

def analyze_topic_distribution(model_data, documents, metadata, year):
    """Analyze how topics are distributed across curriculum documents."""
    
    # Check if we've already analyzed this
    topic_df_path = os.path.join(PROCESSED_DIR, f'topic_distribution_{year}.csv')
    if os.path.exists(topic_df_path):
        try:
            print(f"Loading pre-computed topic distribution for {year}")
            return pd.read_csv(topic_df_path)
        except Exception as e:
            print(f"Error loading pre-computed topic distribution: {e}")
            # Continue with computation if loading fails
    
    # Get model, corpus, and dictionary
    lda_model = model_data['model']
    corpus = model_data['corpus']
    
    # Get topic distribution for each document
    topic_distribution = []
    for i, doc in enumerate(corpus):
        # Get dominant topic
        topic_percs = lda_model.get_document_topics(doc)
        dominant_topic = sorted(topic_percs, key=lambda x: x[1], reverse=True)[0]
        
        # Store document data with its dominant topic
        doc_data = {
            'text': documents[i],
            'dominant_topic': dominant_topic[0],
            'topic_percent': dominant_topic[1],
            'section': metadata[i]['section'] if i < len(metadata) else 'Unknown',
            'item': metadata[i]['item'] if i < len(metadata) else 'Unknown'
        }
        
        topic_distribution.append(doc_data)
    
    # Convert to DataFrame
    topic_df = pd.DataFrame(topic_distribution)
    
    # Save distribution
    topic_df.to_csv(topic_df_path, index=False)
    
    # Check if visualization already exists
    dist_plot_path = os.path.join(FIGURES_DIR, f'topic_distribution_{year}.png')
    section_plot_path = os.path.join(FIGURES_DIR, f'topics_by_section_{year}.png')
    
    if not os.path.exists(dist_plot_path):
        # Create topic distribution chart
        topic_counts = topic_df['dominant_topic'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=topic_counts.index, y=topic_counts.values)
        plt.xlabel("Topic Number")
        plt.ylabel("Document Count")
        plt.title(f"Document Distribution Across Topics ({year})")
        plt.tight_layout()
        plt.savefig(dist_plot_path, dpi=300)
        plt.close()
    
    # New visualization: Topics by curriculum section
    if not os.path.exists(section_plot_path) and 'section' in topic_df.columns:
        # Get top sections for clarity
        top_sections = topic_df['section'].value_counts().head(10).index
        filtered_df = topic_df[topic_df['section'].isin(top_sections)]
        
        plt.figure(figsize=(12, 8))
        section_topic_counts = pd.crosstab(filtered_df['section'], filtered_df['dominant_topic'])
        
        # Normalize to show percentage distribution
        section_topic_pcts = section_topic_counts.div(section_topic_counts.sum(axis=1), axis=0)
        
        # Plot heatmap
        sns.heatmap(section_topic_pcts, cmap='YlGnBu', annot=True, fmt='.0%')
        plt.title(f"Topic Distribution by Curriculum Section ({year})")
        plt.xlabel("Topic Number")
        plt.ylabel("Curriculum Section")
        plt.tight_layout()
        plt.savefig(section_plot_path, dpi=300)
        plt.close()
    
    return topic_df



# Compare topics between curricula
def compare_topics(model_data_2018, model_data_2024):
    """Compare topics between 2018 and 2024 curricula."""
    
    # Check if we've already compared these
    similarity_matrix_path = os.path.join(PROCESSED_DIR, 'topic_similarity_matrix.npy')
    topics_comparison_path = os.path.join(PROCESSED_DIR, 'topic_keywords.json')
    matrix_plot_path = os.path.join(FIGURES_DIR, 'topic_similarity_matrix.png')
    
    if os.path.exists(similarity_matrix_path) and os.path.exists(topics_comparison_path):
        try:
            print("Loading pre-computed topic comparison results")
            with open(topics_comparison_path, 'r', encoding='utf-8') as f:
                topics_comparison = json.load(f)
            
            similarity_matrix = np.load(similarity_matrix_path)
            
            # Check if we need to recreate the visualization
            if not os.path.exists(matrix_plot_path):
                # Create heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(similarity_matrix, annot=True, cmap='YlGnBu', 
                            xticklabels=[f"2024 Topic {i}" for i in range(similarity_matrix.shape[1])],
                            yticklabels=[f"2018 Topic {i}" for i in range(similarity_matrix.shape[0])])
                plt.title("Topic Similarity Between 2018 and 2024 Curricula")
                plt.tight_layout()
                plt.savefig(matrix_plot_path, dpi=300)
                plt.close()
            
            return similarity_matrix, topics_comparison
        except Exception as e:
            print(f"Error loading pre-computed topic comparison: {e}")
            # Continue with computation if loading fails
    
    # Extract top words for each topic from both models
    topics_2018 = {}
    topics_2024 = {}
    
    if model_data_2018:
        model_2018 = model_data_2018['model']
        for topic_id in range(model_2018.num_topics):
            topics_2018[topic_id] = [word for word, _ in model_2018.show_topic(topic_id, topn=10)]
    
    if model_data_2024:
        model_2024 = model_data_2024['model']
        for topic_id in range(model_2024.num_topics):
            topics_2024[topic_id] = [word for word, _ in model_2024.show_topic(topic_id, topn=10)]
    
    # Calculate topic similarity matrix
    similarity_matrix = np.zeros((len(topics_2018), len(topics_2024)))
    
    for i, topic_2018 in topics_2018.items():
        for j, topic_2024 in topics_2024.items():
            # Calculate Jaccard similarity
            intersection = len(set(topic_2018) & set(topic_2024))
            union = len(set(topic_2018) | set(topic_2024))
            similarity = intersection / union if union > 0 else 0
            similarity_matrix[i, j] = similarity
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='YlGnBu', 
                xticklabels=[f"2024 Topic {i}" for i in range(len(topics_2024))],
                yticklabels=[f"2018 Topic {i}" for i in range(len(topics_2018))])
    plt.title("Topic Similarity Between 2018 and 2024 Curricula")
    plt.tight_layout()
    plt.savefig(matrix_plot_path, dpi=300)
    plt.close()
    
    # Save topic words
    topics_comparison = {
        '2018': {str(k): v for k, v in topics_2018.items()},
        '2024': {str(k): v for k, v in topics_2024.items()}
    }
    
    with open(topics_comparison_path, 'w', encoding='utf-8') as f:
        json.dump(topics_comparison, f, indent=2, ensure_ascii=False)
    
    # Save similarity matrix
    np.save(similarity_matrix_path, similarity_matrix)
    
    return similarity_matrix, topics_comparison


# Create concept network
def create_concept_network(topics_comparison, similarity_matrix):
    """Create a network visualization of related concepts across curricula."""
    # Create a graph
    G = nx.Graph()
    
    # Add nodes for each topic
    for year in ['2018', '2024']:
        for topic_id, keywords in topics_comparison[year].items():
            topic_name = f"{year} Topic {topic_id}"
            G.add_node(topic_name, year=year, keywords=keywords)
    
    # Add edges between similar topics
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            similarity = similarity_matrix[i, j]
            if similarity > 0.2:  # Only connect topics with meaningful similarity
                G.add_edge(f"2018 Topic {i}", f"2024 Topic {j}", weight=similarity)
    
    # Calculate node positions (spring layout works well for showing clusters)
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the network
    plt.figure(figsize=(16, 12))
    
    # Draw nodes
    node_colors = ['blue' if G.nodes[node]['year'] == '2018' else 'red' for node in G.nodes]
    node_sizes = [1000 for _ in G.nodes]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges with varying thickness based on similarity
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Add title and legend
    plt.title("Topic Relationship Network Between Curricula", fontsize=16)
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='2018 Topics'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='2024 Topics')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'topic_network.png'), dpi=300)
    plt.close()
    
    # Save graph data
    with open(os.path.join(PROCESSED_DIR, 'topic_network.pkl'), 'wb') as f:
        pickle.dump((G, pos), f)
    
    return G, pos


def visualize_topic_evolution(evolution_results):
    """Create visualizations for topic evolution analysis."""
    # Visualize emerging topics
    if evolution_results['emerging_topics']:
        emerging_df = pd.DataFrame(evolution_results['emerging_topics'])
        plt.figure(figsize=(12, len(emerging_df) * 0.5 + 2))
        ax = plt.subplot(111)
        
        # Display top keywords for each emerging topic
        topic_labels = [', '.join(kw[:5]) for kw in emerging_df['keywords']]
        y_pos = np.arange(len(topic_labels))
        
        # Plot horizontal bars for emerging topics
        ax.barh(y_pos, emerging_df['max_similarity'], color='green', alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic_labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Maximum Similarity to Any 2018 Topic')
        ax.set_title('Emerging Topics in 2024 Curriculum')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'emerging_topics.png'), dpi=300)
        plt.close()
    
    # Visualize fading topics
    if evolution_results['fading_topics']:
        fading_df = pd.DataFrame(evolution_results['fading_topics'])
        plt.figure(figsize=(12, len(fading_df) * 0.5 + 2))
        ax = plt.subplot(111)
        
        # Display top keywords for each fading topic
        topic_labels = [', '.join(kw[:5]) for kw in fading_df['keywords']]
        y_pos = np.arange(len(topic_labels))
        
        # Plot horizontal bars for fading topics
        ax.barh(y_pos, fading_df['max_similarity'], color='red', alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic_labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Maximum Similarity to Any 2024 Topic')
        ax.set_title('Fading Topics from 2018 Curriculum')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fading_topics.png'), dpi=300)
        plt.close()
    
    # Visualize evolving topic pairs
    if evolution_results['evolving_topic_pairs']:
        # Only take top 10 evolving pairs for visualization clarity
        top_pairs = evolution_results['evolving_topic_pairs'][:10]
        evolving_df = pd.DataFrame(top_pairs)
        
        plt.figure(figsize=(12, len(evolving_df) * 0.6 + 2))
        ax = plt.subplot(111)
        
        # Display topic pairs with their keywords
        pair_labels = []
        for _, row in evolving_df.iterrows():
            kw_2018 = ', '.join(row['keywords_2018'][:3])
            kw_2024 = ', '.join(row['keywords_2024'][:3])
            pair_labels.append(f"Topic {row['topic_2018']} ({kw_2018}) → Topic {row['topic_2024']} ({kw_2024})")
        
        y_pos = np.arange(len(pair_labels))
        
        # Plot horizontal bars for evolving topics
        ax.barh(y_pos, evolving_df['similarity'], color='blue', alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pair_labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Topic Similarity')
        ax.set_title('Evolving Topics Between 2018 and 2024 Curricula')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'evolving_topics.png'), dpi=300)
        plt.close()


# Analyze topic evolution
def analyze_topic_evolution(topics_comparison, similarity_matrix, threshold=0.2):
    """
    Analyze how curriculum topics have evolved from 2018 to 2024.
    Identifies emerging and fading topics based on similarity analysis.
    """
    # Check if we've already analyzed this
    evolution_path = os.path.join(PROCESSED_DIR, 'topic_evolution.json')
    if os.path.exists(evolution_path):
        try:
            print("Loading pre-computed topic evolution analysis")
            with open(evolution_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading topic evolution analysis: {e}")
    
    # Identify emerging topics (topics in 2024 with low similarity to any 2018 topic)
    emerging_topics = []
    for j in range(similarity_matrix.shape[1]):  # For each 2024 topic
        max_similarity = np.max(similarity_matrix[:, j])
        if max_similarity < threshold:
            # This 2024 topic is significantly different from all 2018 topics
            emerging_topics.append({
                'topic_id': j,
                'keywords': topics_comparison['2024'][str(j)],
                'max_similarity': float(max_similarity),
                'most_similar_2018_topic': int(np.argmax(similarity_matrix[:, j]))
            })
    
    # Identify fading topics (topics in 2018 with low similarity to any 2024 topic)
    fading_topics = []
    for i in range(similarity_matrix.shape[0]):  # For each 2018 topic
        max_similarity = np.max(similarity_matrix[i, :])
        if max_similarity < threshold:
            # This 2018 topic doesn't have a strong counterpart in 2024
            fading_topics.append({
                'topic_id': i,
                'keywords': topics_comparison['2018'][str(i)],
                'max_similarity': float(max_similarity),
                'most_similar_2024_topic': int(np.argmax(similarity_matrix[i, :]))
            })
    
    # Identify evolving topics (topics with high similarity between years)
    evolving_topic_pairs = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if similarity_matrix[i, j] >= threshold:
                evolving_topic_pairs.append({
                    'topic_2018': i,
                    'topic_2024': j,
                    'similarity': float(similarity_matrix[i, j]),
                    'keywords_2018': topics_comparison['2018'][str(i)],
                    'keywords_2024': topics_comparison['2024'][str(j)]
                })
    
    # Sort by similarity
    evolving_topic_pairs = sorted(evolving_topic_pairs, key=lambda x: x['similarity'], reverse=True)
    
    results = {
        'emerging_topics': emerging_topics,
        'fading_topics': fading_topics,
        'evolving_topic_pairs': evolving_topic_pairs
    }
    
    # Save results
    with open(evolution_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create visualizations for topic evolution
    visualize_topic_evolution(results)
    
    return results

# Main function
def main():
    """Main function to execute the topic modeling and semantic analysis."""
    print("Loading processed data...")
    processed_data = load_processed_data()
    
    if not processed_data:
        print("Error: Could not load processed data.")
        return
    
    print("Preparing data for topic modeling...")
    documents, metadata = prepare_topic_modeling_data(processed_data)
    
    # Get stopwords
    stopwords_set = get_stopwords()
    
    # Initialize results containers
    model_data = {
        '2018': None,
        '2024': None
    }
    
    topic_distributions = {
        '2018': None,
        '2024': None
    }
    
    # Run topic modeling for each curriculum
    for year in ['2018', '2024']:
        if not documents[year]:
            print(f"No documents available for {year} curriculum.")
            continue
        
        print(f"\nPerforming topic modeling for {year} curriculum...")
        
        # Check if we already have a model for this year
        existing_models = [f for f in os.listdir(MODELS_DIR) if f.startswith(f'lda_model_{year}_')]
        if existing_models:
            print(f"Found existing models for {year}: {existing_models}")
            # Get the model with the highest number of topics
            model_nums = [int(f.split('_')[-1].split('.')[0]) for f in existing_models]
            existing_model_path = os.path.join(MODELS_DIR, f'lda_model_{year}_{max(model_nums)}.pkl')
            with open(existing_model_path, 'rb') as f:
                model_data[year] = pickle.load(f)
            print(f"Loaded existing model with {model_data[year]['model'].num_topics} topics")
        else:
            # Find optimal number of topics
            print(f"Finding optimal number of topics for {year} curriculum...")
            optimal_num_topics, _ = find_optimal_topics(
                documents[year], 
                year, 
                stopwords_set,
                start=3,  # Start with 3 topics minimum
                limit=12 if len(documents[year]) > 50 else 8,  # Allow more topics
                step=1
            )
            
            # Run topic modeling with optimal number of topics
            print(f"Running topic modeling with {optimal_num_topics} topics...")
            model_data[year] = run_topic_modeling(
                documents[year],
                year,
                stopwords_set,
                num_topics=optimal_num_topics
            )
        
        # Analyze topic distribution
        print(f"Analyzing topic distribution for {year} curriculum...")
        topic_distributions[year] = analyze_topic_distribution(
            model_data[year],
            documents[year],
            metadata[year],
            year
        )
    
    # Compare topics between curricula if both are available
    if model_data['2018'] and model_data['2024']:
        print("\nComparing topics between curricula...")
        similarity_matrix, topics_comparison = compare_topics(
            model_data['2018'],
            model_data['2024']
        )
        
        # Create concept network visualization
        print("Creating concept network visualization...")
        create_concept_network(topics_comparison, similarity_matrix)
        
        # Analyze topic evolution (our new unsupervised approach)
        print("Analyzing topic evolution between 2018 and 2024 curricula...")
        evolution_results = analyze_topic_evolution(topics_comparison, similarity_matrix)
        
        print("\nTopic evolution analysis:")
        print(f"- Found {len(evolution_results['emerging_topics'])} emerging topics in 2024")
        print(f"- Found {len(evolution_results['fading_topics'])} fading topics from 2018")
        print(f"- Found {len(evolution_results['evolving_topic_pairs'])} evolving topic pairs")
    
    print("\nAnalysis complete. Results saved to the processed_data and figures directories.")
    print("The following files have been created:")
    
    for year in ['2018', '2024']:
        if model_data[year]:
            vis_path = os.path.join(FIGURES_DIR, f'topic_model_vis_{year}.html')
            coh_path = os.path.join(FIGURES_DIR, f'topic_coherence_{year}.png')
            dist_img_path = os.path.join(FIGURES_DIR, f'topic_distribution_{year}.png')
            section_plot_path = os.path.join(FIGURES_DIR, f'topics_by_section_{year}.png')
            dist_csv_path = os.path.join(PROCESSED_DIR, f'topic_distribution_{year}.csv')
            
            if os.path.exists(vis_path):
                print(f"  - topic_model_vis_{year}.html: Interactive topic model visualization")
            if os.path.exists(coh_path):
                print(f"  - topic_coherence_{year}.png: Coherence score plot")
            if os.path.exists(dist_img_path):
                print(f"  - topic_distribution_{year}.png: Document distribution across topics")
            if os.path.exists(section_plot_path):
                print(f"  - topics_by_section_{year}.png: Topic distribution by curriculum section")
            if os.path.exists(dist_csv_path):
                print(f"  - topic_distribution_{year}.csv: Detailed topic distribution data")
    
    # Files for topic comparison and evolution
    topic_files = [
        ('topic_similarity_matrix.png', 'Topic similarity heatmap'),
        ('topic_keywords.json', 'Top keywords for each topic'),
        ('topic_network.png', 'Network visualization of topic relationships'),
        ('topic_network.pkl', 'Network data for further analysis'),
        ('emerging_topics.png', 'Emerging topics in 2024 curriculum'),
        ('fading_topics.png', 'Fading topics from 2018 curriculum'),
        ('evolving_topics.png', 'Evolving topic pairs between curricula'),
        ('topic_evolution.json', 'Topic evolution analysis data')
    ]
    
    for filename, description in topic_files:
        if os.path.exists(os.path.join(FIGURES_DIR, filename)) or os.path.exists(os.path.join(PROCESSED_DIR, filename)):
            print(f"  - {filename}: {description}")

if __name__ == "__main__":
    main()