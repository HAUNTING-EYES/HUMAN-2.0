from src.components.self_coding_ai import SelfCodingAI
import os

def test_self_improvement():
    """Test self-improvement with GitHub integration."""
    try:
        # Initialize AI with current directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ai = SelfCodingAI(base_dir)
        
        # Define learning sources
        sources = {
            'github_repos': [
                'HAUNTING-EYES/HUMAN-2.0',
                'HAUNTING-EYES/human2.0'
            ],
            'docs_dirs': [base_dir + '/docs'] if os.path.exists(base_dir + '/docs') else [],
            'pdf_files': []
        }
        
        print("Starting external learning process...")
        # Learn from external sources
        learning_results = ai.learn_from_external_sources(sources)
        print("\nLearning Results:", learning_results)
        
        print("\nStarting self-analysis...")
        # Perform self-analysis
        analysis_results = ai.analyze_and_improve_self()
        print("\nAnalysis Results:", analysis_results)
        
        return True
        
    except Exception as e:
        print(f"Error in self-improvement process: {str(e)}")
        return False

if __name__ == "__main__":
    test_self_improvement() 