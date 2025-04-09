import os
import shutil
import time
from datetime import datetime

class VersionControl:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.versions_dir = os.path.join(self.base_dir, ".versions")
        if not os.path.exists(self.versions_dir):
            os.makedirs(self.versions_dir)
        print(f"Version control initialized with base_dir: {self.base_dir}")
        print(f"Versions directory: {self.versions_dir}")
    
    def _safe_file_operation(self, operation, max_retries=3, delay=1):
        """Safely perform a file operation with retries"""
        for i in range(max_retries):
            try:
                return operation()
            except (PermissionError, OSError) as e:
                print(f"Attempt {i+1} failed: {str(e)}")
                if i == max_retries - 1:
                    raise
                time.sleep(delay)
    
    def commit_changes(self, message):
        """Create a new version commit"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(self.versions_dir, timestamp)
        print(f"\nCreating new version in: {version_dir}")
        print(f"Commit message: {message}")
        
        def _do_commit():
            if not os.path.exists(version_dir):
                os.makedirs(version_dir)
            
            files_copied = 0
            # Copy current state to version directory
            for root, dirs, files in os.walk(self.base_dir):
                if ".versions" in root or "__pycache__" in root:
                    continue
                
                for file in files:
                    src = os.path.join(root, file)
                    rel_path = os.path.relpath(src, self.base_dir)
                    dst = os.path.join(version_dir, rel_path)
                    
                    try:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
                        files_copied += 1
                    except (PermissionError, OSError) as e:
                        print(f"Failed to copy {src}: {str(e)}")
                        continue
            
            print(f"Copied {files_copied} files to version directory")
            
            # Save commit message
            msg_file = os.path.join(version_dir, "commit_message.txt")
            with open(msg_file, "w") as f:
                f.write(message)
            
            return timestamp
        
        return self._safe_file_operation(_do_commit)
    
    def rollback_to_version(self, commit_message):
        """Rollback to a specific version based on commit message"""
        print(f"\nAttempting to rollback to version with message: {commit_message}")
        
        def _do_rollback():
            # Find the version directory with matching commit message
            for version_name in os.listdir(self.versions_dir):
                version_dir = os.path.join(self.versions_dir, version_name)
                msg_file = os.path.join(version_dir, "commit_message.txt")
                print(f"Checking version: {version_name}")
                
                if os.path.exists(msg_file):
                    with open(msg_file, "r") as f:
                        current_msg = f.read().strip()
                        print(f"Found message: {current_msg}")
                        if current_msg == commit_message:
                            print(f"Found matching version: {version_name}")
                            files_restored = 0
                            
                            # Restore files from this version
                            for root, dirs, files in os.walk(version_dir):
                                for file in files:
                                    if file == "commit_message.txt":
                                        continue
                                        
                                    src = os.path.join(root, file)
                                    rel_path = os.path.relpath(src, version_dir)
                                    dst = os.path.join(self.base_dir, rel_path)
                                    
                                    try:
                                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                                        shutil.copy2(src, dst)
                                        files_restored += 1
                                    except (PermissionError, OSError) as e:
                                        print(f"Failed to restore {src}: {str(e)}")
                                        continue
                            
                            print(f"Restored {files_restored} files")
                            return True
            
            print("No matching version found")
            return False
        
        return self._safe_file_operation(_do_rollback)

    def test_version_control(self):
        """Test version control functionality"""
        test_dir = os.path.join(self.base_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_version.txt")
        print(f"\nStarting version control test")
        print(f"Test directory: {test_dir}")
        print(f"Test file: {test_file}")
        
        try:
            # Create a test file
            with open(test_file, "w") as f:
                f.write("Initial version")
            print("Created test file with initial content")
            
            # Create initial commit
            initial_version = self.commit_changes("Initial commit")
            if not initial_version:
                return "Version control test failed: Could not create initial commit"
            print(f"Created initial commit: {initial_version}")
            
            # Make a change
            with open(test_file, "w") as f:
                f.write("Modified version")
            print("Modified test file")
            
            # Create second commit
            modified_version = self.commit_changes("Modified file")
            if not modified_version:
                return "Version control test failed: Could not create modified commit"
            print(f"Created modified commit: {modified_version}")
            
            # Verify the versions directory contains our commits
            versions = os.listdir(self.versions_dir)
            print(f"Found versions: {versions}")
            if not versions:
                return "Version control test failed: No versions found"
            
            # Rollback to initial version
            success = self.rollback_to_version("Initial commit")
            if not success:
                return "Version control test failed: Could not find initial version"
            print("Successfully rolled back to initial version")
            
            # Verify rollback
            if not os.path.exists(test_file):
                return "Version control test failed: Test file missing after rollback"
            
            with open(test_file, "r") as f:
                content = f.read()
                print(f"File content after rollback: '{content}'")
                if content != "Initial version":
                    return f"Version control test failed: Content mismatch after rollback (got '{content}', expected 'Initial version')"
            
            # Clean up
            shutil.rmtree(test_dir)
            print("Test cleanup completed")
            return "Version control test completed successfully"
            
        except Exception as e:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            return f"Version control test failed: {str(e)}" 