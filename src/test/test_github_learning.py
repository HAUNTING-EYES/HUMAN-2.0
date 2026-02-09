from src.components.external_learning import ExternalLearning
import os

def test_github_learning():
    """Test learning from GitHub repositories."""
    try:
        # Initialize with current directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        learner = ExternalLearning(base_dir, testing=True)  # Use testing mode to avoid heavy model loading
        
        # Test learning from a specific repo
        result = learner.learn_from_github("HAUNTING-EYES/HUMAN-2.0")
        print("\nLearning Results:", result)
        
        if result.get('success'):
            print("\nSuccessfully learned from repository:")
            print(f"- Number of documents processed: {result.get('num_documents', 0)}")
            print("\nPatterns found:")
            for pattern in result.get('patterns', []):
                print(f"- {pattern['type']}: {pattern['content'][:100]}...")
            print("\nBest practices found:")
            for practice in result.get('practices', []):
                print(f"- {practice['type']}: {practice['content'][:100]}...")
        else:
            print("\nFailed to learn from repository:", result.get('error'))
            
        return True
        
    except Exception as e:
        print(f"Error in GitHub learning process: {str(e)}")
        return False

if __name__ == "__main__":
    test_github_learning() 