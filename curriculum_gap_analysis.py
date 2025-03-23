"""
Turkish Mathematics Curriculum NLP Analysis - AI Skill Gap Analysis
===================================================================
This script analyzes gaps between mathematics curricula and AI skill requirements.
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
            if '2018' in key:
                normalized_data['2018'] = value
                print(f"Mapped '{key}' to '2018'")
            elif '2024' in key:
                normalized_data['2024'] = value
                print(f"Mapped '{key}' to '2024'")
        
        return normalized_data
        
    except FileNotFoundError:
        print(f"Processed data file not found. Run preprocessing script first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_ai_skills_coverage_matrix(processed_data):
    """Create a matrix showing coverage of AI skills in the curriculum."""
    # Define AI skill categories and subcategories
    ai_skills = {
        'computational_thinking': ['algorithmic_thinking', 'decomposition', 'abstraction', 'pattern_recognition'],
        'mathematical_reasoning': ['logical_operations', 'inference', 'proof', 'modeling'],
        'pattern_recognition': ['pattern_identification', 'regularities', 'sequences', 'relationships'],
        'data_concepts': ['data_analysis', 'statistics', 'probability', 'visualization'],
        'ai_specific': ['machine_learning', 'neural_networks', 'decision_making', 'automation'],
        'digital_literacy': ['technology_use', 'digital_tools', 'coding', 'information_processing'],
        'optimization': ['function_analysis', 'constraint_handling', 'minimization_maximization'],
        'computational_mathematics': ['numerical_analysis', 'computational_methods', 'approximation']
    }
    
    # Calculate coverage for each skill in both curricula
    coverage = {'2018': {}, '2024': {}}
    
    # Debug: Print keys in processed_data
    print(f"Keys in processed_data: {list(processed_data.keys())}")
    
    for year in ['2018', '2024']:
        if year not in processed_data or not processed_data[year]:
            print(f"No data available for year {year}")
            continue
            
        print(f"Processing year {year}")
        
        # Debug: Check AI relevance categories in the data
        if 'processed_objectives' in processed_data[year]:
            sample_obj = next(iter(processed_data[year]['processed_objectives']), None)
            if sample_obj and 'ai_relevance' in sample_obj:
                print(f"Sample AI relevance categories in {year}: {list(sample_obj['ai_relevance'].keys())}")
        
        for main_skill, subskills in ai_skills.items():
            coverage[year][main_skill] = {}
            
            # Calculate main skill coverage
            if 'processed_objectives' in processed_data[year]:
                # Count objectives with any AI relevance for this skill
                skill_found = False
                relevant_cat = None
                
                # First pass: check if the skill name matches exactly
                main_skill_count = sum(1 for obj in processed_data[year]['processed_objectives'] 
                                    if 'ai_relevance' in obj and 
                                    main_skill in obj['ai_relevance'] and 
                                    obj['ai_relevance'][main_skill] > 0)
                
                if main_skill_count > 0:
                    skill_found = True
                    relevant_cat = main_skill
                else:
                    # Second pass: try to find a close match in AI categories
                    for obj in processed_data[year]['processed_objectives']:
                        if 'ai_relevance' in obj:
                            # Check categories in this object
                            for category in obj['ai_relevance'].keys():
                                # Check if this category might match our skill
                                if main_skill.lower() in category.lower() or category.lower() in main_skill.lower():
                                    skill_found = True
                                    relevant_cat = category
                                    break
                            if skill_found:
                                break
                
                if skill_found:
                    print(f"Found match for {main_skill} in {year}: {relevant_cat}")
                    # Now count using the relevant category
                    main_skill_count = sum(1 for obj in processed_data[year]['processed_objectives'] 
                                        if 'ai_relevance' in obj and 
                                        relevant_cat in obj['ai_relevance'] and 
                                        obj['ai_relevance'][relevant_cat] > 0)
                    
                    total_objectives = len(processed_data[year]['processed_objectives'])
                    coverage[year][main_skill]['coverage'] = main_skill_count / total_objectives if total_objectives > 0 else 0
                    print(f"Coverage for {main_skill} in {year}: {coverage[year][main_skill]['coverage']:.2%}")
                else:
                    print(f"No match found for {main_skill} in {year}")
                    coverage[year][main_skill]['coverage'] = 0
                
                # Calculate subskill coverage (we'll simplify for now)
                for subskill in subskills:
                    # Just estimate as a fraction of the main skill for now
                    coverage[year][main_skill][subskill] = coverage[year][main_skill]['coverage'] / len(subskills)
    
    return coverage, ai_skills


def calculate_ai_gap_scores(coverage, ai_skills, ideal_coverage=0.15):
    """Calculate gap scores between current and ideal AI skill coverage."""
    gap_scores = {'2018': {}, '2024': {}}
    
    for year in ['2018', '2024']:
        for main_skill, details in coverage[year].items():
            main_coverage = details.get('coverage', 0)
            gap_scores[year][main_skill] = {
                'coverage': main_coverage,
                'gap': max(0, ideal_coverage - main_coverage),  # Positive values indicate gap
                'gap_percentage': (ideal_coverage - main_coverage) / ideal_coverage * 100 if ideal_coverage > 0 else 0
            }
            
    # Calculate improvement between years
    if coverage['2018'] and coverage['2024']:
        gap_scores['improvement'] = {}
        for main_skill in ai_skills.keys():
            if main_skill in coverage['2018'] and main_skill in coverage['2024']:
                old_coverage = coverage['2018'][main_skill].get('coverage', 0)
                new_coverage = coverage['2024'][main_skill].get('coverage', 0)
                gap_scores['improvement'][main_skill] = new_coverage - old_coverage
    
    return gap_scores

def visualize_ai_skill_gaps(gap_scores, ai_skills):
    """Create visualizations focused on AI skill gaps."""
    # Create gap heatmap for both years
    for year in ['2018', '2024']:
        if year not in gap_scores:
            print(f"No gap scores available for {year}. Skipping visualization.")
            continue
            
        # Check which skills actually have data for this year
        available_skills = []
        for skill in ai_skills.keys():
            if skill in gap_scores[year]:
                available_skills.append(skill)
        
        if not available_skills:
            print(f"No skills with gap data found for {year}. Skipping visualization.")
            continue
            
        # Print available skills for debugging
        print(f"Available skills for {year}: {available_skills}")
            
        # Prepare data for heatmap with only available skills
        gap_values = []
        skill_names = []
        
        for skill in available_skills:
            if 'gap_percentage' in gap_scores[year][skill]:
                gap_values.append(gap_scores[year][skill]['gap_percentage'])
                skill_names.append(skill)
        
        if not skill_names:
            print(f"No valid gap data found for {year}. Skipping visualization.")
            continue
            
        # Create heatmap
        plt.figure(figsize=(12, 8))
        gap_data = pd.DataFrame({'Skill': skill_names, 'Gap %': gap_values})
        gap_data = gap_data.sort_values('Gap %', ascending=False)
        
        # Custom colormap - darker colors for larger gaps
        cmap = LinearSegmentedColormap.from_list('gap_cmap', ['#f0f9e8', '#7bccc4', '#43a2ca', '#0868ac'])
        
        ax = sns.barplot(x='Gap %', y='Skill', data=gap_data, palette=cmap(gap_data['Gap %']/100))
        plt.title(f'AI Skill Gaps in {year} Mathematics Curriculum')
        plt.xlabel('Gap Percentage (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'ai_skill_gaps_{year}.png'), dpi=300)
        plt.close()
    
    # Create improvement visualization if both years are available
    if 'improvement' in gap_scores:
        # Find skills that have improvement data
        common_skills = []
        for skill in ai_skills.keys():
            if skill in gap_scores['improvement']:
                common_skills.append(skill)
        
        if not common_skills:
            print("No improvement data available. Skipping improvement visualization.")
            return
            
        # Create visualization for available skills
        improvement_values = [gap_scores['improvement'][skill] * 100 for skill in common_skills]
        
        plt.figure(figsize=(12, 8))
        imp_data = pd.DataFrame({'Skill': common_skills, 'Improvement': improvement_values})
        imp_data = imp_data.sort_values('Improvement')
        
        # Diverging colormap - red for negative, green for positive
        colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
        cmap = LinearSegmentedColormap.from_list('div_cmap', colors)
        
        ax = sns.barplot(x='Improvement', y='Skill', data=imp_data, 
                        palette=cmap(np.interp(imp_data['Improvement'], 
                                             [min(improvement_values), max(improvement_values)], 
                                             [0, 1])))
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        plt.title('Change in AI Skill Coverage (2018 to 2024)')
        plt.xlabel('Percentage Points Change')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'ai_skill_improvement.png'), dpi=300)
        plt.close()
        
        
def visualize_knowledge_network_gaps(processed_data, ai_skills):
    """Create a network visualization highlighting gaps in knowledge connections."""
    # Extract concepts and their relationships
    G = nx.Graph()
    
    # Add nodes for each AI skill
    for main_skill, subskills in ai_skills.items():
        G.add_node(main_skill, type='main_skill', size=15)
        for subskill in subskills:
            G.add_node(subskill, type='subskill', size=10)
            G.add_edge(main_skill, subskill, weight=1)
    
    # Add curriculum topics as nodes
    topics_2024 = {}
    if '2024' in processed_data and processed_data['2024']:
        topic_model_path = os.path.join(MODELS_DIR, f'lda_model_2024_5.pkl')
        if os.path.exists(topic_model_path):
            with open(topic_model_path, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                for topic_id in range(model.num_topics):
                    top_words = [word for word, _ in model.show_topic(topic_id, topn=5)]
                    topic_name = f"Topic {topic_id}: {', '.join(top_words[:3])}"
                    topics_2024[topic_id] = topic_name
                    G.add_node(topic_name, type='topic', size=8)
    
    # Connect topics to skills based on AI relevance
    if topics_2024 and '2024' in processed_data and processed_data['2024']:
        topic_dist_path = os.path.join(PROCESSED_DIR, 'topic_distribution_2024.csv')
        if os.path.exists(topic_dist_path):
            topic_df = pd.read_csv(topic_dist_path)
            
            # Calculate AI relevance for each topic
            for topic_id in topics_2024.keys():
                topic_objectives = topic_df[topic_df['dominant_topic'] == topic_id]
                
                for main_skill in ai_skills.keys():
                    # Check if any objectives with this topic have this skill
                    skill_match = False
                    for _, row in topic_objectives.iterrows():
                        ai_relevance = eval(row['ai_relevance']) if isinstance(row.get('ai_relevance'), str) else {}
                        if ai_relevance.get(main_skill, 0) > 0:
                            skill_match = True
                            break
                    
                    if skill_match:
                        G.add_edge(topics_2024[topic_id], main_skill, weight=0.5, edge_type='present')
                    else:
                        # Add dashed edge to represent a missing connection (gap)
                        G.add_edge(topics_2024[topic_id], main_skill, weight=0.2, edge_type='missing')
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # Draw nodes with different styles
    node_types = nx.get_node_attributes(G, 'type')
    node_sizes = nx.get_node_attributes(G, 'size')
    
    main_skill_nodes = [node for node, type_val in node_types.items() if type_val == 'main_skill']
    subskill_nodes = [node for node, type_val in node_types.items() if type_val == 'subskill']
    topic_nodes = [node for node, type_val in node_types.items() if type_val == 'topic']
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=main_skill_nodes, node_size=[node_sizes[node]*100 for node in main_skill_nodes], 
                         node_color='red', alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=subskill_nodes, node_size=[node_sizes[node]*80 for node in subskill_nodes], 
                         node_color='orange', alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_size=[node_sizes[node]*60 for node in topic_nodes], 
                         node_color='blue', alpha=0.5)
    
    # Draw edges with different styles
    edge_types = nx.get_edge_attributes(G, 'edge_type')
    present_edges = [(u, v) for (u, v), edge_type in edge_types.items() if edge_type == 'present']
    missing_edges = [(u, v) for (u, v), edge_type in edge_types.items() if edge_type == 'missing']
    other_edges = [(u, v) for (u, v) in G.edges() if (u, v) not in edge_types]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=present_edges, width=2, edge_color='green', alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=missing_edges, width=1, edge_color='red', style='dashed', alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, width=1, edge_color='gray', alpha=0.4)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif')
    
    plt.title('AI Knowledge Network Gap Analysis', fontsize=16)
    plt.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Main AI Skills'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='AI Subskills'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Curriculum Topics'),
        plt.Line2D([0], [0], color='green', lw=2, label='Present Connection'),
        plt.Line2D([0], [0], color='red', lw=1, linestyle='--', label='Missing Connection (Gap)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'ai_knowledge_network_gaps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def analyze_cognitive_complexity_gaps(processed_data):
    """Analyze gaps in cognitive complexity levels required for AI vs. current curriculum."""
    # Define cognitive complexity levels (based on Bloom's taxonomy)
    cognitive_levels = {
        'remember': 1,
        'understand': 2,
        'apply': 3,
        'analyze': 4,
        'evaluate': 5,
        'create': 6
    }
    
    # Define expected/ideal cognitive complexity for different AI skills
    ai_skill_complexity = {
        'computational_thinking': 4.5,  # Requires high analyze/evaluate skills
        'mathematical_reasoning': 5.0,  # Requires high evaluate/create skills
        'pattern_recognition': 4.0,     # Requires analyze skills
        'data_concepts': 3.8,           # Requires apply/analyze skills
        'ai_specific': 3.5,             # Requires apply/analyze skills
        'digital_literacy': 3.0         # Requires apply skills
    }
    
    # Analyze cognitive complexity in curricula
    results = {'2018': {}, '2024': {}}
    
    for year in ['2018', '2024']:
        if year not in processed_data or not processed_data[year]:
            continue
            
        if 'processed_objectives' not in processed_data[year]:
            continue
            
        # Extract verb complexity from objectives
        verb_complexity = []
        for obj in processed_data[year]['processed_objectives']:
            if 'spacy_tokens' not in obj:
                continue
                
            # Find verbs
            verbs = [token['lemma'].lower() for token in obj['spacy_tokens'] if token['pos'] == 'VERB']
            
            # Map verbs to complexity levels (simplified)
            obj_complexity = 0
            for verb in verbs:
                # This is a simplified example - you would need a more comprehensive verb mapping
                if verb in ['tanımla', 'listele', 'belirle', 'hatırla']:
                    obj_complexity = max(obj_complexity, cognitive_levels['remember'])
                elif verb in ['açıkla', 'yorumla', 'özetle', 'karşılaştır']:
                    obj_complexity = max(obj_complexity, cognitive_levels['understand'])
                elif verb in ['uygula', 'hesapla', 'çöz', 'göster']:
                    obj_complexity = max(obj_complexity, cognitive_levels['apply'])
                elif verb in ['analiz', 'incele', 'ayırt', 'sorgula']:
                    obj_complexity = max(obj_complexity, cognitive_levels['analyze'])
                elif verb in ['değerlendir', 'savun', 'yargıla', 'eleştir']:
                    obj_complexity = max(obj_complexity, cognitive_levels['evaluate'])
                elif verb in ['geliştir', 'tasarla', 'oluştur', 'üret']:
                    obj_complexity = max(obj_complexity, cognitive_levels['create'])
            
            if obj_complexity > 0:
                # Get AI relevance categories for this objective
                ai_categories = []
                if 'ai_relevance' in obj:
                    ai_categories = [cat for cat, score in obj['ai_relevance'].items() if score > 0]
                
                verb_complexity.append({
                    'objective': obj.get('cleaned_text', '')[:50] + '...',
                    'complexity': obj_complexity,
                    'ai_categories': ai_categories
                })
        
        # Calculate average complexity by AI category
        category_complexity = {category: [] for category in ai_skill_complexity.keys()}
        
        for obj_data in verb_complexity:
            for category in obj_data['ai_categories']:
                if category in category_complexity:
                    category_complexity[category].append(obj_data['complexity'])
        
        # Calculate averages and gaps
        for category, complexity_scores in category_complexity.items():
            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
            ideal_complexity = ai_skill_complexity.get(category, 0)
            gap = ideal_complexity - avg_complexity if ideal_complexity > 0 else 0
            
            results[year][category] = {
                'avg_complexity': avg_complexity,
                'ideal_complexity': ideal_complexity,
                'gap': gap,
                'gap_percentage': (gap / ideal_complexity * 100) if ideal_complexity > 0 else 0
            }
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    categories = list(ai_skill_complexity.keys())
    x = np.arange(len(categories))
    width = 0.2
    
    # Plot ideal complexity
    ideal_values = [ai_skill_complexity[cat] for cat in categories]
    plt.bar(x, ideal_values, width, label='Ideal for AI', color='darkgreen', alpha=0.7)
    
    # Plot 2018 and 2024 values if available
    if results['2018']:
        values_2018 = [results['2018'].get(cat, {}).get('avg_complexity', 0) for cat in categories]
        plt.bar(x - width, values_2018, width, label='2018 Curriculum', color='blue', alpha=0.6)
    
    if results['2024']:
        values_2024 = [results['2024'].get(cat, {}).get('avg_complexity', 0) for cat in categories]
        plt.bar(x + width, values_2024, width, label='2024 Curriculum', color='orange', alpha=0.6)
    
    plt.ylabel('Cognitive Complexity Level (1-6)')
    plt.title('Cognitive Complexity Gap Analysis for AI Skills')
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add complexity level labels
    for i, level in enumerate(['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']):
        plt.axhline(y=i+1, color='gray', linestyle=':', alpha=0.5)
        plt.text(-0.5, i+1.05, level, fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cognitive_complexity_gaps.png'), dpi=300)
    plt.close()
    
    return results

def main():
    """Main function to execute the gap analysis."""
    print("Loading processed data...")
    processed_data = load_processed_data()
    
    if not processed_data:
        print("Error: Could not load processed data.")
        return
    
    print("\nPerforming AI skill gap analysis...")
    coverage, ai_skills = create_ai_skills_coverage_matrix(processed_data)
    
    print("Calculating gap scores...")
    gap_scores = calculate_ai_gap_scores(coverage, ai_skills)
    
    print("Visualizing AI skill gaps...")
    visualize_ai_skill_gaps(gap_scores, ai_skills)
    
    print("Creating knowledge network gap visualization...")
    visualize_knowledge_network_gaps(processed_data, ai_skills)
    
    print("Analyzing cognitive complexity gaps...")
    cognitive_gaps = analyze_cognitive_complexity_gaps(processed_data)
    
    # Save gap analysis results
    with open(os.path.join(PROCESSED_DIR, 'ai_skill_gaps.json'), 'w', encoding='utf-8') as f:
        json.dump(gap_scores, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(PROCESSED_DIR, 'cognitive_complexity_gaps.json'), 'w', encoding='utf-8') as f:
        json.dump(cognitive_gaps, f, ensure_ascii=False, indent=2)
    
    print("\nGap analysis complete. Results saved to processed_data and figures directories.")
    print(f"Files created in {PROCESSED_DIR}:")
    print("  - ai_skill_gaps.json")
    print("  - cognitive_complexity_gaps.json")
    
    print(f"\nVisualization files created in {FIGURES_DIR}:")
    print("  - ai_skill_gaps_2018.png (if 2018 data available)")
    print("  - ai_skill_gaps_2024.png (if 2024 data available)")
    print("  - ai_skill_improvement.png")
    print("  - ai_knowledge_network_gaps.png")
    print("  - cognitive_complexity_gaps.png")

if __name__ == "__main__":
    main()