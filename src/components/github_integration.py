import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests
from datetime import datetime, timedelta

class GitHubIntegration:
    """GitHub integration for collecting PR review data."""
    
    def __init__(self, token: Optional[str] = None, cache_dir: str = "data/github_cache"):
        """Initialize GitHub integration.
        
        Args:
            token: GitHub API token (optional)
            cache_dir: Directory to cache PR data
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable.")
            
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def get_pr_reviews(self, repo: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get PR reviews from a repository.
        
        Args:
            repo: Repository name (owner/repo)
            days: Number of days to look back
            
        Returns:
            List of PR reviews with code changes
        """
        cache_file = self.cache_dir / f"{repo.replace('/', '_')}_{days}d.json"
        
        # Check cache first
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=24):
                with open(cache_file) as f:
                    return json.load(f)
                    
        # Fetch from GitHub API
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Get PRs
        prs_url = f"https://api.github.com/repos/{repo}/pulls"
        params = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": 100
        }
        
        prs = []
        page = 1
        
        while True:
            params["page"] = page
            response = requests.get(prs_url, headers=headers, params=params)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching PRs: {response.status_code}")
                break
                
            page_prs = response.json()
            if not page_prs:
                break
                
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            for pr in page_prs:
                updated_at = datetime.strptime(pr["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
                if updated_at < cutoff_date:
                    break
                prs.append(pr)
                
            page += 1
            
        # Get reviews for each PR
        reviews = []
        for pr in prs:
            pr_reviews = self._get_pr_reviews(repo, pr["number"], headers)
            if pr_reviews:
                reviews.extend(pr_reviews)
                
        # Cache results
        with open(cache_file, "w") as f:
            json.dump(reviews, f)
            
        return reviews
        
    def _get_pr_reviews(self, repo: str, pr_number: int, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """Get reviews for a specific PR.
        
        Args:
            repo: Repository name
            pr_number: PR number
            headers: API headers
            
        Returns:
            List of reviews with code changes
        """
        reviews_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
        response = requests.get(reviews_url, headers=headers)
        
        if response.status_code != 200:
            self.logger.error(f"Error fetching PR reviews: {response.status_code}")
            return []
            
        reviews = response.json()
        
        # Get PR files
        files_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
        response = requests.get(files_url, headers=headers)
        
        if response.status_code != 200:
            self.logger.error(f"Error fetching PR files: {response.status_code}")
            return []
            
        files = response.json()
        
        # Combine reviews with code changes
        result = []
        for review in reviews:
            if review["state"] in ["APPROVED", "CHANGES_REQUESTED"]:
                result.append({
                    "pr_number": pr_number,
                    "review_id": review["id"],
                    "reviewer": review["user"]["login"],
                    "state": review["state"],
                    "body": review["body"],
                    "submitted_at": review["submitted_at"],
                    "files": [
                        {
                            "filename": f["filename"],
                            "status": f["status"],
                            "additions": f["additions"],
                            "deletions": f["deletions"],
                            "patch": f.get("patch", "")
                        }
                        for f in files
                    ]
                })
                
        return result
        
    def get_code_changes(self, repo: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get code changes from PR reviews.
        
        Args:
            repo: Repository name
            days: Number of days to look back
            
        Returns:
            List of code changes with before/after states
        """
        reviews = self.get_pr_reviews(repo, days)
        
        changes = []
        for review in reviews:
            for file in review["files"]:
                if file["patch"]:
                    changes.append({
                        "pr_number": review["pr_number"],
                        "review_id": review["review_id"],
                        "filename": file["filename"],
                        "patch": file["patch"],
                        "review_state": review["state"],
                        "review_comment": review["body"]
                    })
                    
        return changes 