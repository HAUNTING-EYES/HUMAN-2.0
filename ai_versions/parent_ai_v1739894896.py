import subprocess
import time
import os
import random
import psutil
import requests  # Hugging Face API
import shutil
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    """
    Fetches the Hugging Face API key securely.
    """
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise ValueError("❌ API key not found. Ensure HF_API_KEY is set in your .env file.")
    return api_key

def run_child():
    """
    Runs the child process (a modified version of the AI).
    """
    print("🚀 Running child AI process...")
    try:
        start_time = time.time()
        process = subprocess.Popen(['python', 'child_ai.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=5)  # Prevent infinite hang
        except subprocess.TimeoutExpired:
            process.kill()
            print("⏳ Child process took too long! Terminating.")
            return "TIMEOUT", None, None

        execution_time = time.time() - start_time

        # Optionally, add real memory tracking using psutil
        # For now, we keep it simplified
        memory_usage = 0  
        return stdout.strip(), execution_time, memory_usage
    except Exception as e:
        return f"Child process failed: {e}", None, None

def evaluate_child_performance(output, execution_time, memory_usage):
    """
    Evaluates whether the child's performance is better.
    """
    if "SUCCESS" in output:
        return True

    if execution_time is None or memory_usage is None:
        return False

    return execution_time < 2.5 and memory_usage < 100

def improve_code_with_huggingface(code):
    """
    Uses Hugging Face's API to introduce targeted improvements (fine-tuning modifications)
    in the AI logic.
    """
    print("🔍 Sending code to Hugging Face API for structured improvements...")
    try:
        API_KEY = get_api_key()
        client = InferenceClient(api_key=API_KEY)
        # Select a focus area from predefined improvement areas.
        improvement_areas = [
            "Optimize execution speed by reducing redundant operations",
            "Improve error handling to make AI more robust",
            "Enhance modularity and readability of the code"
        ]
        selected_improvement = random.choice(improvement_areas)
        prompt = f"""
        You are an AI software engineer. Improve the following Python AI system by:
        - **Primary Focus**: {selected_improvement}
        - Ensure that the new version retains all functionality and is free of syntax and logic errors.
        - Return only a fully functional Python script without any explanations.
        
        Here is the current AI code:
        {code}

        """
        retries = 3
        for attempt in range(retries):
            try:
                result = client.text_generation(
                    model="codellama/CodeLlama-13b-hf",
                    prompt=prompt,
                    max_new_tokens=1000
                )
                print(f"✅ AI-generated improvements received! [Focus: {selected_improvement}]")
                result = result.strip()
                if "def " not in result or "if __name__" not in result:
                    print("❌ AI-generated code is incomplete. Keeping current version.")
                    return code
                return result
            except Exception as e:
                print(f"❌ AI Code Improvement Failed (Attempt {attempt+1}/{retries}):", e)
                if attempt < retries - 1:
                    print("⏳ Retrying...")
                    time.sleep(random.uniform(1, 4))
                else:
                    print("❌ Failed after retries, keeping current version.")
                    return code
    except Exception as e:
        print("❌ Error:", e)
        return code

def upgrade_ai_module(code):
    """
    Uses Hugging Face's API to generate a radical upgrade in the AI's capabilities—
    specifically, integrating a self-reflection/meta-learning module.
    
    The new module should:
    - Log internal decision processes (chain-of-thought) during code modifications.
    - Analyze and critique its own reasoning to guide future improvements.
    - Introduce a mechanism for radical code changes that can later be fine-tuned.
    
    Return only the updated Python script.
    """
    print("🔍 Requesting radical upgrade (self-reflection module) from Hugging Face API...")
    try:
        API_KEY = get_api_key()
        client = InferenceClient(api_key=API_KEY)
        prompt = f"""
        You are an expert AI software architect. The current AI system iteratively improves itself by generating a child version,
        evaluating its performance, and updating the parent if the child is better.
        
        The next radical upgrade is to incorporate a self-reflection and meta-learning module into the AI.
        This new module should:
        - Enable the AI to log its chain-of-thought (its internal reasoning process) during code generation.
        - Analyze its reasoning to determine when a radical, functionality-changing update is needed.
        - Adjust the "learning rate" for self-modification: allow for larger, radical changes in early iterations,
          and gradually shift to fine-tuning as performance improves.
        - Be modular, so that the new self-reflection component can be maintained separately.
        - Ensure that the overall system remains functional and free of errors.
        
        Return only the fully functional Python script with these changes integrated, without explanations.
        
        Here is the current AI code:
        {code}
        """
        retries = 3
        for attempt in range(retries):
            try:
                result = client.text_generation(
                    model="codellama/CodeLlama-13b-hf",
                    prompt=prompt,
                    max_new_tokens=1500
                )
                print("✅ Radical upgrade received: self-reflection module integrated!")
                result = result.strip()
                if "def " not in result or "if __name__" not in result:
                    print("❌ AI-generated radical upgrade is incomplete. Keeping current version.")
                    return code
                return result
            except Exception as e:
                print(f"❌ Radical upgrade attempt {attempt+1}/{retries} failed:", e)
                if attempt < retries - 1:
                    print("⏳ Retrying radical upgrade...")
                    time.sleep(random.uniform(1, 4))
                else:
                    print("❌ Failed radical upgrade after retries, keeping current version.")
                    return code
    except Exception as e:
        print("❌ Error during radical upgrade:", e)
        return code

def save_best_ai():
    """
    Saves the current best AI version before updating.
    """
    if not os.path.exists("ai_versions"):
        os.makedirs("ai_versions")
    timestamp = int(time.time())
    shutil.copy("parent_ai.py", f"ai_versions/parent_ai_v{timestamp}.py")
    print(f"💾 Saved best AI version: ai_versions/parent_ai_v{timestamp}.py")

def update_parent_code(learning_stage):
    """
    Updates the parent AI code based on the child AI if performance has improved.
    
    The 'learning_stage' parameter controls the type of improvement:
    - Stage 0: Radical upgrade (introduce self-reflection/meta-learning module).
    - Stage 1: Fine-tuning of existing functionalities.
    """
    with open("child_ai.py", "r", encoding="utf-8") as child_file:
        child_logic = child_file.read()
    
    if learning_stage == 0:
        # Radical change: incorporate a new self-reflection module.
        improved_logic = upgrade_ai_module(child_logic)
    else:
        # Fine-tuning improvements.
        improved_logic = improve_code_with_huggingface(child_logic)
    
    if not improved_logic or len(improved_logic.strip()) == 0:
        print("❌ AI-generated code is empty. Keeping current version.")
        return

    with open("ai_generated_code.py", "w", encoding="utf-8") as debug_file:
        debug_file.write(improved_logic)
    
    if "def" in improved_logic:
        save_best_ai()
        with open("parent_ai.py", "w", encoding="utf-8") as parent_file:
            parent_file.write(improved_logic)
        print("✅ Parent AI logic improved with AI-generated updates!")
    else:
        print("❌ Failed to update Parent AI. AI did not return valid Python code.")

def generate_child_code():
    """
    Generates a new child AI script with incremental improvements.
    """
    print("📝 Generating child AI code...")
    with open("parent_ai.py", "r", encoding="utf-8") as parent_file:
        parent_code = parent_file.read()
    
    modified_code = parent_code.replace("time.sleep(random.uniform(1, 4))", "time.sleep(random.uniform(1, 4))")
    
    with open("child_ai.py", "w", encoding="utf-8") as child_file:
        child_file.write(modified_code)
    
    print("✅ Child AI code generated successfully!")

if __name__ == "__main__":
    print("🔹 Starting Parent AI...")
    try:
        API_KEY = get_api_key()
        print("✅ API Key Loaded Successfully!")
    except ValueError as e:
        print(f"❌ ERROR: {e}. Please set your Hugging Face API key in the .env file.")
        exit(1)
    
    print("Parent AI running... Looping automatically...")
    learning_stage = 0  # 0 = radical changes; later, switch to fine-tuning (e.g., learning_stage = 1)
    successful_updates = 0

    while True:
        print("\n🔹 Generating new child AI code...")
        generate_child_code()
        
        print("🔹 Running the child AI...")
        child_output, execution_time, memory_usage = run_child()
        print(f"🔹 Child AI output: {child_output}, Execution Time: {execution_time:.2f}s, Memory: {memory_usage:.2f}MB")
        
        if evaluate_child_performance(child_output, execution_time, memory_usage):
            print("✅ Child AI is better. Updating parent AI...")
            update_parent_code(learning_stage)
            successful_updates += 1
            # After a certain number of radical updates, transition to fine-tuning.
            if successful_updates >= 5:
                learning_stage = 1
                print("🔄 Transitioning to fine-tuning stage for incremental improvements.")
        else:
            print("❌ Child AI is worse. Keeping current version.")
        
        time.sleep(random.uniform(1, 4))
