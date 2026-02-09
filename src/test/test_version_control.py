from core.version_control import VersionControl
import os
import shutil

def main():
    vc = VersionControl()
    
    # Create test directory
    test_dir = os.path.join(os.path.dirname(__file__), "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Create test file
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Initial content")
    
    print(f"Created test file: {test_file}")
    print(f"Initial content: Initial content")
    
    # Create first version
    version1 = vc.commit_changes("Version 1")
    print(f"Created version 1: {version1}")
    
    # Modify file
    with open(test_file, "w") as f:
        f.write("Modified content")
    
    print(f"Modified content: Modified content")
    
    # Create second version
    version2 = vc.commit_changes("Version 2")
    print(f"Created version 2: {version2}")
    
    # List versions
    versions_dir = os.path.join(os.path.dirname(__file__), ".versions")
    print(f"\nVersions directory: {versions_dir}")
    print("Available versions:")
    for version in sorted(os.listdir(versions_dir)):
        msg_file = os.path.join(versions_dir, version, "commit_message.txt")
        if os.path.exists(msg_file):
            with open(msg_file, "r") as f:
                msg = f.read().strip()
            print(f"- {version}: {msg}")
    
    # Rollback to first version
    print("\nRolling back to Version 1...")
    if vc.rollback_to_version("Version 1"):
        with open(test_file, "r") as f:
            content = f.read()
        print(f"Content after rollback: {content}")
        if content == "Initial content":
            print("Rollback successful!")
        else:
            print("Rollback failed: content mismatch")
    else:
        print("Rollback failed: version not found")
    
    # Clean up
    shutil.rmtree(test_dir)
    print("\nTest completed")

if __name__ == "__main__":
    main() 