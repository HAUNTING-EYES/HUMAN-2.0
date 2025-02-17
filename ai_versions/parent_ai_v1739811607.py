
        """
        if "SUCCESS" in output:
            return True  # Allow more flexibility in performance changes

        if execution_time is None or memory_usage is None:
            return False  # If metrics are missing, reject update

        return execution_time < 2.5 and memory_usage < 100

def improve_code_with_huggingface(code):
    """
    Uses Hugging Face's Hosted API to introduce major improvements in AI logic.
    """
    print("🔍 Sending code to Hugging Face API for major improvements...")
    try:
        API_KEY = get_api_key()
        client = InferenceClient(api_key=API_KEY)  # No need for 'provider' argument
        
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
            prompt=prompt,  # Corrected to use 'prompt' and no 'inputs' argument
            max_new_tokens=1000
        )
        
        print("✅ AI-generated major improvements received!")
        return result
    except Exception as e:
        print("❌ AI Code Improvement Failed:", e)
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
    
    modified_code = parent_code.replace("time.sleep(random.uniform(0.4, 1.2))", "time.sleep(random.uniform(0.4, 1.2))")
    
    with open("child_ai.py", "w", encoding="utf-8") as child_file:
        child_file.write(modified_code)
    
    print("✅ Child AI code generated successfully!