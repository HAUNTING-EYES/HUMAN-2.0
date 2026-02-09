import unittest
from src.components.code_metrics import CodeMetrics

class TestCodeMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = CodeMetrics()
        
    def test_complexity_calculation(self):
        """Test complexity score calculation"""
        # Simple code
        simple_code = """
def add(a, b):
    return a + b
"""
        simple_score = self.metrics._calculate_complexity(simple_code)
        self.assertGreater(simple_score, 0.8)  # Should have high score
        
        # Complex code
        complex_code = """
def complex_function(a, b, c):
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        if b > 0:
            return b
        else:
            return 0
"""
        complex_score = self.metrics._calculate_complexity(complex_code)
        self.assertLess(complex_score, 0.8)  # Should have lower score
        
    def test_maintainability_calculation(self):
        """Test maintainability score calculation"""
        # Well-maintained code
        good_code = """
def well_maintained_function():
    \"\"\"This is a well-documented function.\"\"\"
    # Clear variable names
    result = 42
    return result
"""
        good_score = self.metrics._calculate_maintainability(good_code)
        self.assertGreater(good_score, 0.7)
        
        # Poorly maintained code
        bad_code = """
def bad_function():
    a=1
    b=2
    c=3
    d=4
    e=5
    f=6
    g=7
    h=8
    i=9
    j=10
    return a+b+c+d+e+f+g+h+i+j
"""
        bad_score = self.metrics._calculate_maintainability(bad_code)
        self.assertLess(bad_score, 0.7)
        
    def test_security_calculation(self):
        """Test security score calculation"""
        # Secure code
        secure_code = """
def secure_function(user_input):
    # Sanitize input
    sanitized = user_input.strip()
    return sanitized
"""
        secure_score = self.metrics._calculate_security(secure_code)
        self.assertGreater(secure_score, 0.7)
        
        # Insecure code
        insecure_code = """
def insecure_function(user_input):
    # Unsafe eval
    result = eval(user_input)
    return result
"""
        insecure_score = self.metrics._calculate_security(insecure_code)
        self.assertLess(insecure_score, 0.7)
        
    def test_style_calculation(self):
        """Test style score calculation"""
        # Good style
        good_style = """
def good_style_function():
    \"\"\"Function with good style.\"\"\"
    result = 42
    return result
"""
        good_score = self.metrics._calculate_style(good_style)
        self.assertGreater(good_score, 0.7)
        
        # Bad style
        bad_style = """
def bad_style_function():
	result=42
	return result
"""
        bad_score = self.metrics._calculate_style(bad_style)
        self.assertLess(bad_score, 0.7)
        
    def test_documentation_calculation(self):
        """Test documentation score calculation"""
        # Well documented
        documented = """
def documented_function():
    \"\"\"This is a well-documented function.\"\"\"
    return 42
"""
        doc_score = self.metrics._calculate_documentation(documented)
        self.assertGreater(doc_score, 0.7)
        
        # Poorly documented
        undocumented = """
def undocumented_function():
    return 42
"""
        undoc_score = self.metrics._calculate_documentation(undocumented)
        self.assertLess(undoc_score, 0.7)
        
    def test_test_coverage_calculation(self):
        """Test test coverage score calculation"""
        # Code with tests
        code_with_tests = """
def function_to_test():
    return 42

def test_function():
    assert function_to_test() == 42
"""
        test_score = self.metrics._calculate_test_coverage(code_with_tests)
        self.assertGreater(test_score, 0.0)
        
        # Code without tests
        code_without_tests = """
def function_without_tests():
    return 42
"""
        no_test_score = self.metrics._calculate_test_coverage(code_without_tests)
        self.assertEqual(no_test_score, 0.0)
        
    def test_overall_metrics(self):
        """Test overall metrics calculation"""
        code = """
def example_function(a, b):
    \"\"\"Example function with good practices.\"\"\"
    try:
        result = a + b
        return result
    except Exception as e:
        raise Exception(f"Error in example_function: {str(e)}")
"""
        metrics = self.metrics.calculate_metrics(code)
        
        # Check all metrics are present
        self.assertIn('complexity', metrics)
        self.assertIn('maintainability', metrics)
        self.assertIn('security', metrics)
        self.assertIn('style', metrics)
        self.assertIn('documentation', metrics)
        self.assertIn('test_coverage', metrics)
        self.assertIn('overall_score', metrics)
        
        # Check scores are in valid range
        for metric, score in metrics.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            
    def test_improvement_suggestions(self):
        """Test improvement suggestions generation"""
        code = """
def complex_function(a,b,c):
    if a>0:
        if b>0:
            if c>0:
                return a+b+c
            else:
                return a+b
        else:
            return a
    else:
        if b>0:
            return b
        else:
            return 0
"""
        suggestions = self.metrics.get_improvement_suggestions(code)
        
        # Check suggestions are generated
        self.assertGreater(len(suggestions), 0)
        
        # Check suggestion format
        for suggestion in suggestions:
            self.assertIn('type', suggestion)
            self.assertIn('description', suggestion)
            self.assertIn('severity', suggestion)
            self.assertIn(suggestion['severity'], ['high', 'medium'])

if __name__ == '__main__':
    unittest.main() 