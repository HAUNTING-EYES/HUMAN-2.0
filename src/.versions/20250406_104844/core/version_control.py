import os
import shutil
import time
from datetime import datetime

class VersionControl:
    def __init__(self):
        self.versions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".versions")
        if not os.path.exists(self.versions_dir):
            os.makedirs(self.versions_dir)
    
    def _safe_file_operation(self, operation, max_retries=3, delay=1):
        """Safely perform a file operation with retries"""
        for i in range(max_retries):
            try:
                return operation()
            except PermissionError:
                if i == max_retries - 1:
                    raise
                time.sleep(delay)
    
    def commit_changes(self, message):
        """Create a new version commit"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(self.versions_dir, timestamp)
        
        def _do_commit():
            os.makedirs(version_dir, exist_ok=True)
            
            # Copy current state to version directory
            for root, dirs, files in os.walk(os.path.dirname(self.versions_dir)):
                if ".versions" in root:
                    continue
                for file in files:
                    src = os.path.join(root, file)
                    rel_path = os.path.relpath(src, os.path.dirname(self.versions_dir))
                    dst = os.path.join(version_dir, rel_path)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            
            # Save commit message
            with open(os.path.join(version_dir, "commit_message.txt"), "w") as f:
                f.write(message)
            
            return timestamp
        
        return self._safe_file_operation(_do_commit)
    
    def rollback_to_version(self, commit_message):
        """Rollback to a specific version based on commit message"""
        def _do_rollback():
            # Find the version directory with matching commit message
            for version_dir in os.listdir(self.versions_dir):
                msg_file = os.path.join(self.versions_dir, version_dir, "commit_message.txt")
                if os.path.exists(msg_file):
                    with open(msg_file, "r") as f:
                        if f.read().strip() == commit_message:
                            # Restore files from this version
                            version_path = os.path.join(self.versions_dir, version_dir)
                            base_dir = os.path.dirname(self.versions_dir)
                            
                            for root, dirs, files in os.walk(version_path):
                                for file in files:
                                    if file == "commit_message.txt":
                                        continue
                                    src = os.path.join(root, file)
                                    rel_path = os.path.relpath(src, version_path)
                                    dst = os.path.join(base_dir, rel_path)
                                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                                    shutil.copy2(src, dst)
                            return True
            return False
        
        return self._safe_file_operation(_do_rollback)

    def test_version_control(self):
        """Test version control functionality"""
        test_dir = os.path.join(os.path.dirname(self.versions_dir), "test")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_version.txt")
        
        try:
            # Create a test file
            with open(test_file, "w") as f:
                f.write("Initial version")
            
            # Create initial commit
            self.commit_changes("Initial commit")
            
            # Make a change
            with open(test_file, "w") as f:
                f.write("Modified version")
            
            # Create second commit
            self.commit_changes("Modified file")
            
            # Rollback to initial version
            success = self.rollback_to_version("Initial commit")
            if not success:
                return "Version control test failed: Could not find initial version"
            
            # Verify rollback
            with open(test_file, "r") as f:
                content = f.read()
                if content != "Initial version":
                    return "Version control test failed: Content mismatch after rollback"
            
            # Clean up
            shutil.rmtree(test_dir)
            return "Version control test completed successfully"
            
        except Exception as e:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            return f"Version control test failed: {str(e)}" 