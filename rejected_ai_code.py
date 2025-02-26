Here is the current AI code:
        import subprocess
import time
import os
import random
import psutil
import requests  # Hugging Face API
import shutil
import textwrap
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import ast

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    
    Fetches the Hugging Face API key securely.
    
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise ValueError("❌ API key not found. Ensure HF_API_KEY is set in your .env file.")
    return api_key

def run_child():
    
    Runs the child process (a modified version of the AI).
    
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
        return stdout.strip(), execution_time, 0  # Memory tracking simplified
    except Exception as e:
        return f"Child process failed: {e}", None, None

def evaluate_child_performance(output, execution_time, memory_usage):
    
    Evaluates whether the child's performance is better.
    
    if "SUCCESS" in output:
        return True  # Allow more flexibility in performance changes

    if execution_time is None or memory_usage is None:
        return False  # If metrics are missing, reject update

    return execution_time < 2.0 and memory_usage < 75  # Stricter optimization targets

def clean_ai_code(ai_code):
    
    Cleans AI-generated code by removing unwanted text artifacts and validating syntax.
    
    ai_code = textwrap.dedent(ai_code.strip()).replace("", "").strip()
    ai_code = ai_code.replace('', '').replace("", '')

    try:
        ast.parse(ai_code)  # Ensure the AI-generated code is valid Python
    except SyntaxError as e:
        print(f"❌ AI-generated code is invalid: {e}. Keeping current version.")
        with open("rejected_ai_code.py", "w", encoding="utf-8") as debug_file:
            debug_file.write(ai_code)
        return None

    return ai_code

def improve_code_with_huggingface(code):
    
    Uses Hugging Face's Hosted API to introduce **major** improvements in AI logic.
    
    print("🔍 Sending code to Hugging Face API for significant improvements...")

    try:
        API_KEY = get_api_key()
        client = InferenceClient(api_key=API_KEY)

        prompt = f
        You are an expert AI software engineer. Improve the following Python AI system by:
        - **Optimize performance** using better algorithms and parallel execution.
        - **Refactor the code** for better readability and modularity.
        - **Introduce new functionalities** like self-logging and adaptive learning.
        - **Ensure self-correction mechanisms** to track and learn from past mistakes.
        - **Enhance logging** to explain why changes were made and their impact.
        - Return only a **fully functional Python script**, without explanations.

        Here is the current AI code:
        {code}

        Return only the updated Python script. Do NOT include explanations, comments, or duplicated code.
        

        result = client.text_generation(
            model="codellama/CodeLlama-13b-hf",
            prompt=prompt,
            max_new_tokens=1500
        )

        print("✅ AI-generated major improvements received!")
        return clean_ai_code(result)
    except Exception as e:
        print("❌ AI Code Improvement Failed:", e)
        return code  # Return original if API fails

def save_best_ai():
    
    Saves the current best AI version before updating.
    
    if not os.path.exists("ai_versions"):
        os.makedirs("ai_versions")

    timestamp = int(time.time())
    shutil.copy("parent_ai.py", f"ai_versions/parent_ai_v{timestamp}.py")
    print(f"💾 Saved best AI version: ai_versions/parent_ai_v{timestamp}.py")

def update_parent_code():
    
    If the child performs better, use AI to enhance the logic and update the parent.
    
    with open("child_ai.py", "r", encoding="utf-8") as child_file:
        child_logic = child_file.read()

    improved_logic = improve_code_with_huggingface(child_logic)

    if not improved_logic:
        return

    with open("ai_generated_code.py", "w", encoding="utf-8") as debug_file:
        debug_file.write(improved_logic)

    save_best_ai()
    with open("parent_ai.py", "w", encoding="utf-8") as parent_file:
        parent_file.write(improved_logic)
    print("✅ Parent AI logic improved with AI-generated updates!")

def generate_child_code():
    
    Generates a new child AI script with **radical