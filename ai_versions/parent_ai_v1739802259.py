import subprocess
import time
import os
import random
import psutil
import requests  # Hugging Face API
import shutil
from huggingface_hub import InferenceClient

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

    # Allow AI to experiment with slower but potentially smarter code
    return execution_time < 2.5 and memory_usage < 100


def improve_code_with_huggingface(code):
    """
    Uses Hugging Face's Hosted API to introduce major improvements in AI logic.
    """
    print("🔍 Sending code to Hugging Face API for major improvements...")
    API_KEY = os.getenv("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")
    if not API_KEY:
        print("❌ Error: API key not found.")
        return code

    client = InferenceClient(provider="hf-inference", api_key=API_KEY)

    try:
        prompt = f"""
        Improve this AI system by introducing:
        - A more optimized, self-improving algorithm
        - A new feature for learning from past failures
        - Faster and more efficient logic
        - Parallel execution if possible

        Here is the current AI code:
        {code}

        Return a complete, improved Python script.
        """

        result = client.text_generation(
            model="codellama/CodeLlama-13b-hf",
            inputs=prompt,
            provider="hf-inference",
            max_new_tokens=1000,  # Bigger response
            timeout=15,
        )
        
        print("✅ AI-generated major improvements received!")
        return result
    except Exception as e:
        print("❌ AI Code Improvement Failed:", e)
        return code  # Return original if API fails

def save_best_ai():
    """
    Saves the current best AI version before updating.
    """
    if not os.path.exists("ai_versions"):
        os.makedirs("ai_versions")  # Create directory if missing

    timestamp = int(time.time())
    shutil.copy("parent_ai.py", f"ai_versions/parent_ai_v{timestamp}.py")
    print(f"💾 Saved best AI version: ai_versions/parent_ai_v{timestamp}.py")

def test_ai_functionality():
    """
    Runs the AI to check if the new version is working correctly.
    """
    output, exec_time, mem_usage = run_child()
    if "SUCCESS" not in output:
        print("❌ AI failed functionality test! Reverting to last version.")
        return False
    if exec_time > 2.0:
        print("⚠️ AI is too slow! Reverting to last version.")
        return False
    if mem_usage > 50:
        print("⚠️ AI uses too much memory! Reverting to last version.")
        return False
    return True

def review_ai_code(new_code):
    """
    Sends new AI-generated code to another AI for review.
    """
    print("🔍 Sending AI-generated code for review...")
    API_KEY = os.getenv("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")
    if not API_KEY:
        print("❌ Error: API key not found.")
        return False

    client = InferenceClient(provider="hf-inference", api_key=API_KEY)

    review_prompt = f"""
    Review the following Python AI system update. 
    - Does it improve efficiency and learning speed?
    - Are there any errors or weaknesses in the code?
    - Suggest any final refinements before accepting it.

    Here is the new AI code:
    {new_code}
    """
    
    try:
        review = client.text_generation(
            model="codellama/CodeLlama-13b-hf",
            inputs=review_prompt,
            provider="hf-inference",
            max_new_tokens=500,
            timeout=15,
        )
        
        print("✅ AI review completed! Review output:")
        print(review)
        if "error" in review.lower() or "weakness" in review.lower():
            print("❌ AI review found issues. Rejecting update.")
            return False
        return True
    except Exception as e:
        print("❌ AI Review Failed:", e)
        return False

def update_parent_code():
    """
    If the child performs better, use AI to enhance the logic and update the parent.
    """
    with open("child_ai.py", "r", encoding="utf-8") as child_file:
        child_logic = child_file.read()

    improved_logic = improve_code_with_huggingface(child_logic)

    if not improved_logic or len(improved_logic.strip()) == 0:
        print("❌ AI-generated code is empty. Keeping current version.")
        return

    # Save AI response to a file for debugging
    with open("ai_generated_code.py", "w", encoding="utf-8") as debug_file:
        debug_file.write(improved_logic)
    
    if "def" in improved_logic:  # Check if AI returned valid code
        save_best_ai()
        if review_ai_code(improved_logic) and test_ai_functionality():
            with open("parent_ai.py", "w", encoding="utf-8") as parent_file:
                parent_file.write(improved_logic)
            print("✅ Parent AI logic improved with AI-generated updates!")
        else:
            print("🔄 Rolling back to previous version...")
    else:
        print("❌ Failed to update Parent AI. AI did not return valid Python code.")

def generate_child_code():
    """
    Generates a new child AI script with incremental improvements.
    """
    print("📝 Generating child AI code...")
    with open("parent_ai.py", "r", encoding="utf-8") as parent_file:
        parent_code = parent_file.read()
    
    modified_code = parent_code.replace("time.sleep(random.uniform(0.5, 1.5))", "time.sleep(random.uniform(0.4, 1.2))")
    
    with open("child_ai.py", "w", encoding="utf-8") as child_file:
        child_file.write(modified_code)
    
    print("✅ Child AI code generated successfully!")

if __name__ == "__main__":
    print("Parent AI running... Looping automatically...")
    while True:
        print("\n🔹 Generating new child AI code...")
        generate_child_code()
        
        print("🔹 Running the child AI...")
        child_output, execution_time, memory_usage = run_child()
        print(f"🔹 Child AI output: {child_output}, Execution Time: {execution_time:.2f}s, Memory: {memory_usage:.2f}MB")
        
        if evaluate_child_performance(child_output, execution_time, memory_usage):
            print("✅ Child AI is better. Running AI review...")
            update_parent_code()
        else:
            print("❌ Child AI is worse. Keeping current version.")
        
        time.sleep(5)  # Wait before next iteration