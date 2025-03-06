def evaluate_code(code):
    """
    Evaluates the given AI-generated code.

    Returns:
        (bool, str): A tuple where the first value indicates success (True/False),
        and the second value provides feedback or an error message.
    """
    try:
        exec(code, {})  # Run the code in an isolated environment
        return True, "Code executed successfully!"
    except Exception as e:
        return False, f"Error: {str(e)}"
