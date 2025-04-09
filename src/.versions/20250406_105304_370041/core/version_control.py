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
    
    def _write_commit_message(self, version_dir, message):
        """Write commit message to a file"""
        msg_file = os.path.join(version_dir, "commit_message.txt")
        try:
            with open(msg_file, "w") as f:
                f.write(message)
            return True
        except Exception as e:
            print(f"Failed to write commit message: {str(e)}")
            return False
    
    def _read_commit_message(self, version_dir):
        """Read commit message from a file"""
        msg_file = os.path.join(version_dir, "commit_message.txt")
        try:
            if os.path.exists(msg_file):
                with open(msg_file, "r") as f:
                    return f.read().strip()
        except Exception as e:
            print(f"Failed to read commit message: {str(e)}")
        return None
    
    def _get_unique_version_dir(self, base_timestamp):
        """Get a unique version directory name"""
        counter = 0
        while True:
            if counter == 0:
                version_dir = os.path.join(self.versions_dir, base_timestamp)
            else:
                version_dir = os.path.join(self.versions_dir, f"{base_timestamp}_{counter}")
            
            if not os.path.exists(version_dir):
                return version_dir
            counter += 1
    
    def commit_changes(self, message):
        """Create a new version commit"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        version_dir = self._get_unique_version_dir(timestamp)
        print(f"\nCreating new version in: {version_dir}")
        print(f"Commit message: {message}")
        
        def _do_commit():
            # Create version directory
            if not os.path.exists(version_dir):
                os.makedirs(version_dir)
            
            # Write commit message first
            if not self._write_commit_message(version_dir, message):
                raise Exception("Failed to write commit message")
            
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
            
            # Verify commit message was written
            saved_message = self._read_commit_message(version_dir)
            if saved_message != message:
                raise Exception("Commit message verification failed")
            
            return os.path.basename(version_dir)
        
        return self._safe_file_operation(_do_commit)
    
    def rollback_to_version(self, commit_message):
        """Rollback to a specific version based on commit message"""
        print(f"\nAttempting to rollback to version with message: {commit_message}")
        
        def _do_rollback():
            # Find the version directory with matching commit message
            for version_name in sorted(os.listdir(self.versions_dir)):
                version_dir = os.path.join(self.versions_dir, version_name)
                print(f"Checking version: {version_name}")
                
                saved_message = self._read_commit_message(version_dir)
                if saved_message:
                    print(f"Found message: {saved_message}")
                    if saved_message == commit_message:
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
            # Clean up any existing test files
            if os.path.exists(test_file):
                os.remove(test_file)
            
            # Create a test file
            with open(test_file, "w") as f:
                f.write("Initial version")
            print("Created test file with initial content")
            
            # Create initial commit
            initial_version = self.commit_changes("Initial commit")
            if not initial_version:
                return "Version control test failed: Could not create initial commit"
            print(f"Created initial commit: {initial_version}")
            
            # Verify initial commit message
            initial_msg = self._read_commit_message(os.path.join(self.versions_dir, initial_version))
            if initial_msg != "Initial commit":
                return f"Version control test failed: Initial commit message mismatch (got '{initial_msg}')"
            
            # Make a change
            with open(test_file, "w") as f:
                f.write("Modified version")
            print("Modified test file")
            
            # Create second commit
            modified_version = self.commit_changes("Modified file")
            if not modified_version:
                return "Version control test failed: Could not create modified commit"
            print(f"Created modified commit: {modified_version}")
            
            # Verify modified commit message
            modified_msg = self._read_commit_message(os.path.join(self.versions_dir, modified_version))
            if modified_msg != "Modified file":
                return f"Version control test failed: Modified commit message mismatch (got '{modified_msg}')"
            
            # Verify the versions directory contains our commits
            versions = sorted(os.listdir(self.versions_dir))
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