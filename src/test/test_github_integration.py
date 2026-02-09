import os
from github import Github
from dotenv import load_dotenv

def test_github_connection():
    """Test GitHub connection and basic functionality."""
    try:
        # Initialize GitHub connection
        token = "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"
        g = Github(token)
        
        # Test connection
        user = g.get_user()
        print(f"Successfully connected to GitHub as: {user.login}")
        
        # Test repo access
        print("\nAccessible repositories:")
        for repo in g.get_user().get_repos():
            print(f"- {repo.full_name}")
            
        return True
        
    except Exception as e:
        print(f"Error connecting to GitHub: {str(e)}")
        return False

if __name__ == "__main__":
    test_github_connection() 