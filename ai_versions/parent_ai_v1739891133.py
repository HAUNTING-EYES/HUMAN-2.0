Here is the current AI code:
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

# Test fetching the API key
if __name__ == "__main__":
    print("Your Hugging Face API key is:", get_api_key())


def run_child():
    """
    Runs the child process (a modified version of the AI).
    """
    print("🚀 Running child AI process...")
    try:
        start time.time()
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
    """
    Evaluates whether the child's performance is better.
    """
    if "SUCCESS" in output:
        return True  # Allow more flexibility in performance changes

    if execution_time is None or memory_usage is None:
        return False  # If metrics are missing, reject update

    return execution_time < 2.5 and memory_usage < 100

def improve_code_with_huggingface(code):
    """
    Uses Hugging Face's Hosted API to introduce **targeted** improvements in AI logic.
    Retries in case of a server error.
    """
    print("🔍 Sending code to Hugging Face API for structured improvements...")

    try:
        API_KEY = get_api_key()
        client = InferenceClient(api_key=API_KEY)

        # 🔥 Select a Focus Area for This Iteration
        improvement_areas = [
            "Make the AI smarter by improving its learning algorithm",
            "Optimize execution speed by reducing redundant operations",
            "Reduce memory usage while keeping functionality intact",
            "Improve error handling to make AI more robust",
            "Enhance AI’s ability to evaluate and compare different solutions",
            "Refactor the AI’s code to be more readable and modular",
            "Introduce a feature where the AI logs how and why it modified itself"
        ]
        selected_improvement = random.choice(improvement_areas)

        prompt = f"""
        You are an AI software engineer. Improve the following Python AI system by:
        - **Primary Focus**: {selected_improvement}
        - Ensure the AI correctly updates itself without duplicating functions
        - Make the AI smarter by refining decision-making mechanisms
        - Remove redundant or unnecessary code
        - Ensure the new AI does not have any syntax or logic errors
        - Return only a **fully functional Python script**, without explanations

        Here is the current AI code:
        {code}

        Return only the updated Python script. Do NOT include explanations, comments, or duplicated code.
        """

        # Retry logic
        retries = 3
        for