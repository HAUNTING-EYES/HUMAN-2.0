import os
import json
import shutil
import subprocess
from evaluator import evaluate_code  # Ensure evaluator.py has evaluate_code function

# Paths
PARENT_AI_FILE = "parent_ai.py"
CHILD_AI_FILE = "child_ai.py"
BACKUP_FOLDER = "history"
LOG_FILE = "logs.txt"
KNOWLEDGE_FILE = "knowledge.json"

# Ensure history folder exists
os.makedirs(BACKUP_FOLDER, exist_ok=True)

def log(message):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(message + "\n")
    print(message)

# Backup current AI version
def backup_ai():
    backup_path = os.path.join(BACKUP_FOLDER, f"parent_backup_{len(os.listdir(BACKUP_FOLDER))}.py")
    shutil.copy(PARENT_AI_FILE, backup_path)
    log(f"[Backup] Parent AI backed up at {backup_path}")

# Load knowledge base
def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r") as f:
            return json.load(f)
    return {}

# Save knowledge base
def save_knowledge(data):
    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Generate new AI version
def generate_new_ai():
    try:
        subprocess.run(["python", "ai_generator.py"], check=True)
        log("[Generation] New AI code generated successfully.")
    except subprocess.CalledProcessError as e:
        log(f"[Error] AI generation failed: {e}")
        return False
    return True

# Evaluate the new AI
def test_new_ai():
    if not os.path.exists(CHILD_AI_FILE):
        log("[Error] No new AI code found.")
        return False
    
    try:
        result = evaluate_code(CHILD_AI_FILE)
        log(f"[Evaluation] AI Evaluation Result: {result}")
        return result
    except Exception as e:
        log(f"[Error] Evaluation failed: {e}")
        return False

# Promote child AI to parent if it passes
def promote_ai():
    try:
        shutil.copy(CHILD_AI_FILE, PARENT_AI_FILE)
        log("[Update] New AI version accepted and promoted to Parent AI.")
        return True
    except Exception as e:
        log(f"[Error] Failed to promote AI: {e}")
        return False

# Main self-improvement cycle
def self_improvement_cycle():
    log("[Start] Initiating self-improvement cycle...")
    backup_ai()
    
    if generate_new_ai() and test_new_ai():
        if promote_ai():
            log("[Success] AI improved successfully!")
        else:
            log("[Failure] AI promotion failed.")
    else:
        log("[Revert] New AI did not pass evaluation. Keeping the existing AI.")

if __name__ == "__main__":
    self_improvement_cycle()
