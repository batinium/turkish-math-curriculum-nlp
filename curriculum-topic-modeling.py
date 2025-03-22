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

# For topic modeling
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess

# For visualization
import pyLDAvis
import pyLDAvis.gensim_models
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

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
    
    for year in ['2018', '2024']:
        if not processed_data[year]:
            continue
            
        for obj in processed_data[year]['processed_objectives']:
            # Extract cleaned text
            text = obj['cleaned_text']
            
            # Add to documents
            documents[year].append(text)
            
            # Store metadata
            metadata[year].append({
                'section': obj['section'],
                'item': obj['item'],
                'ai_relevance_score': obj['ai_relevance_score']
            })
    
    return documents, metadata

# Define Turkish stopwords
def get_stopwords():
    """Get Turkish stopwords and additional custom stopwords."""
    try:
        turkish_stopwords = set(stopwords.words('turkish'))
    except:
        print("Turkish stopwords not available in NLTK. Using custom list.")
        # Basic Turkish stopwords
        turkish_stopwords = {
            've', 'ile', 'bu', 'bir', 'için', 'ya', 'de', 'da', 'olarak', 'gibi',
            'kadar', 'değil', 'daha', 'çok', 'en', 'göre', 'her', 'mi', 'ne',
            'o', 'ama', 'ki', 'eğer', 'veya', 'hem', 'ise', 'ancak', 'şey'
        }
    
    # Add custom stopwords for curriculum analysis
    custom_stopwords = {
        'fonksiyon', 'ile', 'olarak', 'için', 'belirler', 'kullanır',
        'açıklanır', 'yapar', 'verilir', 'a', 'b', 'c', 'ç', 'd', 'e', 'f'
    }
    
    return turkish_stopwords.union(custom_stopwords)

# Preprocess text for topic modeling
def preprocess_for_topics(text, stopwords_set):
    """Preprocess text for topic modeling."""
    # Tokenize and clean
    tokens = simple_preprocess(text, deacc=True)  # Removes punctuation and converts to lowercase
    
    # Remove stopwords and short words
    tokens = [token for token in tokens if token not in stopwords_set and len(token) > 3]
    
    return tokens

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
        processed_docs = [preprocess_for_topics(doc, stopwords_set) for doc in documents]
        
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
    coherence_values = []
    model_list = []
    
    # Preprocess documents
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
    
    return optimal_num_topics, model_list[optimal_idx]

# Analyze topic distribution
def analyze_topic_distribution(model_data, documents, metadata, year):
    """Analyze how topics are distributed across curriculum documents."""
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
            'item': metadata[i]['item'] if i < len(metadata) else 'Unknown',
            'ai_relevance_score': metadata[i]['ai_relevance_score'] if i < len(metadata) else 0
        }
        
        topic_distribution.append(doc_data)
    
    # Convert to DataFrame
    topic_df = pd.DataFrame(topic_distribution)
    
    # Save distribution
    topic_df.to_csv(os.path.join(PROCESSED_DIR, f'topic_distribution_{year}.csv'), index=False)
    
    # Create topic distribution chart
    topic_counts = topic_df['dominant_topic'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=topic_counts.index, y=topic_counts.values)
    plt.xlabel("Topic Number")
    plt.ylabel("Document Count")
    plt.title(f"Document Distribution Across Topics ({year})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'topic_distribution_{year}.png'), dpi=300)
    plt.close()
    
    # Analyze AI relevance by topic
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dominant_topic', y='ai_relevance_score', data=topic_df)
    plt.xlabel("Topic Number")
    plt.ylabel("AI Relevance Score")
    plt.title(f"AI Relevance by Topic ({year})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'ai_relevance_by_topic_{year}.png'), dpi=300)
    plt.close()
    
    return topic_df

# Compare topics between curricula
def compare_topics(model_data_2018, model_data_2024):
    """Compare topics between 2018 and 2024 curricula."""
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
    plt.savefig(os.path.join(FIGURES_DIR, 'topic_similarity_matrix.png'), dpi=300)
    plt.close()
    
    # Save topic words
    topics_comparison = {
        '2018': {str(k): v for k, v in topics_2018.items()},
        '2024': {str(k): v for k, v in topics_2024.items()}
    }
    
    with open(os.path.join(PROCESSED_DIR, 'topic_keywords.json'), 'w', encoding='utf-8') as f:
        json.dump(topics_comparison, f, indent=2, ensure_ascii=False)
    
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

# Classify learning objectives by AI-relevance potential
def classify_objectives_by_ai_potential(processed_data):
    """Classify learning objectives by their potential AI relevance."""
    classifications = {
        '2018': [],
        '2024': []
    }
    
    # Define a simple model for classification
    # In a real implementation, you would train a classifier
    def classify_objective(obj):
        # Get AI relevance score
        ai_score = obj['ai_relevance_score']
        
        # Define thresholds for classification
        if ai_score >= 3:
            return "High AI Relevance"
        elif ai_score >= 1:
            return "Medium AI Relevance"
        else:
            return "Low AI Relevance"
    
    # Classify objectives for both curricula
    for year in ['2018', '2024']:
        if not processed_data[year]:
            continue
            
        for obj in processed_data[year]['processed_objectives']:
            classification = classify_objective(obj)
            
            classifications[year].append({
                'section': obj['section'],
                'item': obj['item'],
                'text': obj['cleaned_text'],
                'ai_relevance_score': obj['ai_relevance_score'],
                'classification': classification
            })
    
    # Convert to DataFrames
    for year in ['2018', '2024']:
        if classifications[year]:
            df = pd.DataFrame(classifications[year])
            df.to_csv(os.path.join(PROCESSED_DIR, f'ai_classification_{year}.csv'), index=False)
    
    # Create classification distribution charts
    for year in ['2018', '2024']:
        if classifications[year]:
            df = pd.DataFrame(classifications[year])
            
            plt.figure(figsize=(10, 6))
            sns.countplot(x='classification', data=df, order=['Low AI Relevance', 'Medium AI Relevance', 'High AI Relevance'])
            plt.title(f"AI Relevance Classification Distribution ({year})")
            plt.xlabel("AI Relevance Category")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'ai_classification_{year}.png'), dpi=300)
            plt.close()
    
    return classifications

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
        
        # Find optimal number of topics
        print(f"Finding optimal number of topics for {year} curriculum...")
        optimal_num_topics, _ = find_optimal_topics(
            documents[year], 
            year, 
            stopwords_set,
            start=2,
            limit=8 if len(documents[year]) > 10 else 5,
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
        
        print("Creating concept network visualization...")
        create_concept_network(topics_comparison, similarity_matrix)
    
    # Classify objectives by AI potential
    print("\nClassifying learning objectives by AI potential...")
    classifications = classify_objectives_by_ai_potential(processed_data)
    
    print("\nAnalysis complete. Results saved to the processed_data and figures directories.")
    print("The following files have been created:")
    
    for year in ['2018', '2024']:
        if model_data[year]:
            print(f"  - topic_model_vis_{year}.html: Interactive topic model visualization")
            print(f"  - topic_coherence_{year}.png: Coherence score plot")
            print(f"  - topic_distribution_{year}.png: Document distribution across topics")
            print(f"  - topic_distribution_{year}.csv: Detailed topic distribution data")
            print(f"  - ai_relevance_by_topic_{year}.png: AI relevance scores by topic")
            print(f"  - ai_classification_{year}.png: AI relevance classification distribution")
            print(f"  - ai_classification_{year}.csv: Detailed AI classification data")
    
    if model_data['2018'] and model_data['2024']:
        print("  - topic_similarity_matrix.png: Topic similarity heatmap")
        print("  - topic_keywords.json: Top keywords for each topic")
        print("  - topic_network.png: Network visualization of topic relationships")
        print("  - topic_network.pkl: Network data for further analysis")

if __name__ == "__main__":
    main()