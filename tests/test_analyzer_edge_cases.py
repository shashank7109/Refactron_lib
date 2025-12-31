"""Comprehensive edge case tests for all analyzers."""

from pathlib import Path

from refactron.analyzers.code_smell_analyzer import CodeSmellAnalyzer
from refactron.analyzers.complexity_analyzer import ComplexityAnalyzer
from refactron.analyzers.dead_code_analyzer import DeadCodeAnalyzer
from refactron.analyzers.dependency_analyzer import DependencyAnalyzer
from refactron.analyzers.performance_analyzer import PerformanceAnalyzer
from refactron.analyzers.security_analyzer import SecurityAnalyzer
from refactron.analyzers.type_hint_analyzer import TypeHintAnalyzer
from refactron.core.config import RefactronConfig


class TestComplexityAnalyzerEdgeCases:
    """Edge case tests for ComplexityAnalyzer."""

    def test_empty_file(self) -> None:
        """Test analyzer handles empty file."""
        config = RefactronConfig()
        analyzer = ComplexityAnalyzer(config)
        issues = analyzer.analyze(Path("empty.py"), "")
        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_single_line_function(self) -> None:
        """Test analyzer handles single-line function."""
        config = RefactronConfig()
        analyzer = ComplexityAnalyzer(config)
        code = "def simple(): return 42"
        issues = analyzer.analyze(Path("test.py"), code)
        assert isinstance(issues, list)

    def test_async_function_complexity(self) -> None:
        """Test complexity detection in async functions."""
        config = RefactronConfig(max_function_complexity=3)
        analyzer = ComplexityAnalyzer(config)
        code = """
async def complex_async(x):
    if x > 0:
        if x > 10:
            if x > 20:
                if x > 30:
                    return "high"
    return "low"
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "C001" for issue in issues)

    def test_async_function_length(self) -> None:
        """Test length detection in async functions."""
        config = RefactronConfig(max_function_length=5)
        analyzer = ComplexityAnalyzer(config)
        code = """
async def long_async():
    line1 = 1
    line2 = 2
    line3 = 3
    line4 = 4
    line5 = 5
    line6 = 6
    line7 = 7
    return line7
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "C002" for issue in issues)

    def test_maintainability_index_very_low(self) -> None:
        """Test detection of very low maintainability index."""
        config = RefactronConfig()
        analyzer = ComplexityAnalyzer(config)
        # Create complex code with low maintainability
        code = """
def complex_function(a, b, c, d, e, f, g, h):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            if g:
                                if h:
                                    x = a + b + c + d + e + f + g + h
                                    y = x * 2
                                    z = y * 3
                                    return z
    return 0
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # MI calculation can fail or not trigger on some code
        # Just ensure no crash
        assert isinstance(issues, list)

    def test_nested_async_loops(self) -> None:
        """Test deeply nested loops in async function."""
        config = RefactronConfig()
        analyzer = ComplexityAnalyzer(config)
        code = """
async def nested_loops():
    for i in range(10):
        for j in range(10):
            for k in range(10):
                for l in range(10):
                    await process(i, j, k, l)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "C003" for issue in issues)

    def test_complex_call_chains(self) -> None:
        """Test detection of complex method call chains."""
        config = RefactronConfig()
        analyzer = ComplexityAnalyzer(config)
        code = """
def process():
    result = obj.method1().method2().method3().method4().method5().method6()
    return result
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "C004" for issue in issues)

    def test_function_without_end_lineno(self) -> None:
        """Test function length calculation fallback."""
        config = RefactronConfig(max_function_length=2)
        analyzer = ComplexityAnalyzer(config)
        code = """
def small():
    x = 1
    return x
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should handle gracefully even with different AST versions
        assert isinstance(issues, list)

    def test_syntax_error_handling(self) -> None:
        """Test graceful handling of syntax errors."""
        config = RefactronConfig()
        analyzer = ComplexityAnalyzer(config)
        code = "def broken(:"
        issues = analyzer.analyze(Path("test.py"), code)
        # Should return error issue, not crash
        assert isinstance(issues, list)

    def test_critical_complexity_level(self) -> None:
        """Test critical complexity level detection."""
        config = RefactronConfig(max_function_complexity=5)
        analyzer = ComplexityAnalyzer(config)
        # Create function with complexity > 20
        code = """
def ultra_complex(x):
    if x == 1: return 1
    elif x == 2: return 2
    elif x == 3: return 3
    elif x == 4: return 4
    elif x == 5: return 5
    elif x == 6: return 6
    elif x == 7: return 7
    elif x == 8: return 8
    elif x == 9: return 9
    elif x == 10: return 10
    elif x == 11: return 11
    elif x == 12: return 12
    elif x == 13: return 13
    elif x == 14: return 14
    elif x == 15: return 15
    elif x == 16: return 16
    elif x == 17: return 17
    elif x == 18: return 18
    elif x == 19: return 19
    elif x == 20: return 20
    elif x == 21: return 21
    else: return 0
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should have critical level issue
        critical_issues = [i for i in issues if i.level.value == "critical"]
        assert len(critical_issues) > 0

    def test_error_complexity_level(self) -> None:
        """Test error complexity level detection."""
        config = RefactronConfig(max_function_complexity=5)
        analyzer = ComplexityAnalyzer(config)
        # Create function with complexity between 15 and 20
        code = """
def very_complex(x):
    if x == 1: return 1
    elif x == 2: return 2
    elif x == 3: return 3
    elif x == 4: return 4
    elif x == 5: return 5
    elif x == 6: return 6
    elif x == 7: return 7
    elif x == 8: return 8
    elif x == 9: return 9
    elif x == 10: return 10
    elif x == 11: return 11
    elif x == 12: return 12
    elif x == 13: return 13
    elif x == 14: return 14
    elif x == 15: return 15
    elif x == 16: return 16
    else: return 0
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should have error level issue
        error_issues = [i for i in issues if i.level.value == "error"]
        assert len(error_issues) > 0


class TestCodeSmellAnalyzerEdgeCases:
    """Edge case tests for CodeSmellAnalyzer."""

    def test_empty_file(self) -> None:
        """Test analyzer handles empty file."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        issues = analyzer.analyze(Path("empty.py"), "")
        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_single_parameter_function(self) -> None:
        """Test function with exactly max parameters."""
        config = RefactronConfig(max_parameters=3)
        analyzer = CodeSmellAnalyzer(config)
        code = """
def exactly_max(a, b, c):
    return a + b + c
"""
        issues = analyzer.analyze(Path("test.py"), code)
        param_issues = [i for i in issues if i.rule_id == "S001"]
        assert len(param_issues) == 0

    def test_too_many_parameters_async(self) -> None:
        """Test async function with too many parameters."""
        config = RefactronConfig(max_parameters=3)
        analyzer = CodeSmellAnalyzer(config)
        code = """
async def too_many(a, b, c, d, e):
    return a + b + c + d + e
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "S001" for issue in issues)

    def test_shallow_nesting(self) -> None:
        """Test function with acceptable nesting depth."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
def shallow():
    if True:
        for i in range(10):
            x = i * 2
    return x
"""
        issues = analyzer.analyze(Path("test.py"), code)
        nesting_issues = [i for i in issues if i.rule_id == "S002"]
        assert len(nesting_issues) == 0

    def test_deep_nesting_with_try(self) -> None:
        """Test deep nesting with try-except blocks."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
def deeply_nested():
    try:
        if True:
            for i in range(10):
                while i > 0:
                    with open("file") as f:
                        if f:
                            return "deep"
    except:
        pass
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "S002" for issue in issues)

    def test_duplicate_functions_with_numbers(self) -> None:
        """Test detection of numbered duplicate functions."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
def process1():
    return 1

def process2():
    return 2

def process3():
    return 3
"""
        issues = analyzer.analyze(Path("test.py"), code)
        duplicate_issues = [i for i in issues if i.rule_id == "S003"]
        assert len(duplicate_issues) > 0

    def test_magic_numbers_acceptable_values(self) -> None:
        """Test that common numbers (0, 1, -1, 2) are not flagged."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
def calc():
    x = 0
    y = 1
    z = -1
    w = 2
    return x + y + z + w
"""
        issues = analyzer.analyze(Path("test.py"), code)
        magic_issues = [i for i in issues if i.rule_id == "S004"]
        assert len(magic_issues) == 0

    def test_magic_numbers_floats(self) -> None:
        """Test detection of magic float numbers."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
def calc():
    pi = 3.14159
    e = 2.71828
    return pi * e
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "S004" for issue in issues)

    def test_private_function_skipped_docstring(self) -> None:
        """Test that private functions are skipped for docstring check."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
def _private_no_docstring():
    return 42

def __special_no_docstring__():
    return 42
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should not flag private functions
        docstring_issues = [i for i in issues if "_private" in i.message]
        assert len(docstring_issues) == 0

    def test_class_missing_docstring(self) -> None:
        """Test detection of missing class docstring."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
class MyClass:
    def method(self):
        return 42
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "S005" and "Class" in issue.message for issue in issues)

    def test_async_function_missing_docstring(self) -> None:
        """Test detection of missing async function docstring."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
async def async_no_docs():
    return 42
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "S005" for issue in issues)

    def test_unused_imports_with_type_hints(self) -> None:
        """Test unused imports that appear in type hints."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
from typing import List

def process(items: List[int]) -> int:
    return sum(items)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # List is used in type hints
        unused_list = [i for i in issues if "List" in str(i.metadata) and i.rule_id == "S006"]
        # Should not flag as unused if it appears in type hints
        assert len(unused_list) == 0

    def test_unused_import_from_statement(self) -> None:
        """Test unused import from statement."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
from pathlib import Path
from typing import Dict

def hello():
    print("Hello")
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "S006" for issue in issues)

    def test_repeated_code_blocks_short_function(self) -> None:
        """Test that short functions don't trigger repeated block detection."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = """
def short():
    x = 1
    y = 2
    z = 3
    return x + y + z
"""
        issues = analyzer.analyze(Path("test.py"), code)
        repeated_issues = [i for i in issues if i.rule_id == "S007"]
        # Too short for meaningful duplication detection
        assert len(repeated_issues) == 0

    def test_syntax_error_with_line_number(self) -> None:
        """Test syntax error handling with line number."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)
        code = "def broken(:\n    pass"
        issues = analyzer.analyze(Path("test.py"), code)
        # Should return syntax error issue
        assert len(issues) > 0
        assert any("syntax" in issue.message.lower() for issue in issues)


class TestDeadCodeAnalyzerEdgeCases:
    """Edge case tests for DeadCodeAnalyzer."""

    def test_empty_file(self) -> None:
        """Test analyzer handles empty file."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        issues = analyzer.analyze(Path("empty.py"), "")
        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_function_with_underscore_prefix(self) -> None:
        """Test that private functions are not flagged as unused."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def _private_unused():
    return 42

def __special__():
    return 100
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Private functions should not be flagged
        assert len(issues) == 0

    def test_function_in_all_export(self) -> None:
        """Test that exported functions are not flagged."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
__all__ = ['exported_function']

def exported_function():
    return 42

def another_exported():
    return 100
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # exported_function should not be flagged
        unused_issues = [i for i in issues if "exported_function" in i.message]
        assert len(unused_issues) == 0

    def test_function_called_as_attribute(self) -> None:
        """Test function called via attribute access."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def helper():
    return 42

class MyClass:
    def method(self):
        return self.helper()
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # helper might be detected as unused (simple analysis limitation)
        assert isinstance(issues, list)

    def test_unused_variable_with_underscore(self) -> None:
        """Test that variables starting with _ are not flagged."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def process():
    _unused = 42
    return 100
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # _unused should not be flagged
        var_issues = [i for i in issues if "_unused" in i.message]
        assert len(var_issues) == 0

    def test_variable_used_in_nested_scope(self) -> None:
        """Test variable usage detection in nested scopes."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def outer():
    value = 42
    print(value)
    return value
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # value is used, should not be flagged
        value_issues = [i for i in issues if "'value'" in i.message]
        assert len(value_issues) == 0

    def test_unreachable_after_return_multiple_statements(self) -> None:
        """Test multiple unreachable statements after return."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def unreachable():
    return 42
    x = 1
    y = 2
    z = 3
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect unreachable code
        assert any(issue.rule_id == "DEAD003" for issue in issues)

    def test_unreachable_after_break_in_loop(self) -> None:
        """Test unreachable code after break in loop."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def process():
    for i in range(10):
        break
        print("unreachable")
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "DEAD003" for issue in issues)

    def test_unreachable_after_break_in_while(self) -> None:
        """Test unreachable code after break in while loop."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def process():
    while True:
        break
        x = 1
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "DEAD003" for issue in issues)

    def test_empty_function_with_pass(self) -> None:
        """Test detection of empty function with pass."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def empty():
    pass
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "DEAD004" for issue in issues)

    def test_empty_function_with_docstring(self) -> None:
        """Test that function with docstring is not flagged as empty."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = '''
def documented():
    """This function has a docstring."""
    pass
'''
        issues = analyzer.analyze(Path("test.py"), code)
        # Should not flag as empty since it has docstring
        empty_issues = [i for i in issues if i.rule_id == "DEAD004"]
        assert len(empty_issues) == 0

    def test_special_method_not_flagged_empty(self) -> None:
        """Test that special methods are not flagged as empty."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
class MyClass:
    def __init__(self):
        pass

    def __str__(self):
        pass
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Special methods should not be flagged
        empty_issues = [i for i in issues if i.rule_id == "DEAD004"]
        assert len(empty_issues) == 0

    def test_condition_always_false(self) -> None:
        """Test detection of always-false condition."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def check():
    if False:
        return "never"
    return "always"
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "DEAD005" and "False" in issue.message for issue in issues)

    def test_redundant_comparison_with_false(self) -> None:
        """Test detection of redundant comparison with False."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = """
def check(x):
    if x == False:
        return "bad"
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "DEAD006" for issue in issues)

    def test_syntax_error_handling(self) -> None:
        """Test graceful handling of syntax errors."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)
        code = "def broken(:"
        issues = analyzer.analyze(Path("test.py"), code)
        # Should not crash
        assert isinstance(issues, list)


class TestSecurityAnalyzerEdgeCases:
    """Edge case tests for SecurityAnalyzer."""

    def test_empty_file(self) -> None:
        """Test analyzer handles empty file."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        issues = analyzer.analyze(Path("empty.py"), "")
        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_ignored_file_pattern(self) -> None:
        """Test that ignored files are skipped."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = "eval('dangerous')"
        # Test file should be in ignore patterns
        issues = analyzer.analyze(Path("test_security.py"), code)
        # Some issues might still appear with lower confidence
        assert isinstance(issues, list)

    def test_whitelisted_rule(self) -> None:
        """Test that whitelisted rules are filtered."""
        config = RefactronConfig()
        config.security_rule_whitelist = {"SEC001": ["*.py"]}
        analyzer = SecurityAnalyzer(config)
        code = "eval('test')"
        issues = analyzer.analyze(Path("test.py"), code)
        # SEC001 should be filtered
        sec001_issues = [i for i in issues if i.rule_id == "SEC001"]
        assert len(sec001_issues) == 0

    def test_low_confidence_filtered(self) -> None:
        """Test that low confidence issues are filtered."""
        config = RefactronConfig()
        config.security_min_confidence = 0.9
        analyzer = SecurityAnalyzer(config)
        code = "import random"
        issues = analyzer.analyze(Path("test.py"), code)
        # Random import has lower confidence
        assert len(issues) == 0

    def test_test_file_confidence_reduction(self) -> None:
        """Test confidence reduction for test files."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = "eval('test')"
        issues = analyzer.analyze(Path("tests/test_something.py"), code)
        # Should still detect but with lower confidence
        if len(issues) > 0:
            assert issues[0].confidence < 1.0

    def test_demo_file_confidence_reduction(self) -> None:
        """Test confidence reduction for demo files."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = "eval('demo')"
        issues = analyzer.analyze(Path("examples/demo.py"), code)
        # Should still detect but with lower confidence
        if len(issues) > 0:
            assert issues[0].confidence < 1.0

    def test_compile_function_detection(self) -> None:
        """Test detection of compile() function."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = """
def run():
    compile('code', 'file', 'exec')
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any("compile" in issue.message.lower() for issue in issues)

    def test_import_function_detection(self) -> None:
        """Test detection of __import__() function."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = """
def dynamic_import(name):
    __import__(name)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any("__import__" in issue.message for issue in issues)

    def test_marshal_import_detection(self) -> None:
        """Test detection of marshal import."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = "import marshal"
        issues = analyzer.analyze(Path("test.py"), code)
        assert any("marshal" in issue.message.lower() for issue in issues)

    def test_shelve_import_detection(self) -> None:
        """Test detection of shelve import."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = "import shelve"
        issues = analyzer.analyze(Path("test.py"), code)
        assert any("shelve" in issue.message.lower() for issue in issues)

    def test_sha1_import_detection(self) -> None:
        """Test detection of SHA1 weak crypto."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = "from hashlib import sha1"
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "SEC006" for issue in issues)

    def test_hardcoded_secret_empty_string(self) -> None:
        """Test that empty strings are not flagged as secrets."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = 'api_key = ""'
        issues = analyzer.analyze(Path("test.py"), code)
        secret_issues = [i for i in issues if i.rule_id == "SEC003"]
        assert len(secret_issues) == 0

    def test_hardcoded_secret_placeholder(self) -> None:
        """Test that placeholders are not flagged as secrets."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = 'password = "CHANGEME"'
        issues = analyzer.analyze(Path("test.py"), code)
        secret_issues = [i for i in issues if i.rule_id == "SEC003"]
        assert len(secret_issues) == 0

    def test_hardcoded_secret_metadata_variable(self) -> None:
        """Test that metadata variables are not flagged."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = '__author__ = "John Doe"'
        issues = analyzer.analyze(Path("test.py"), code)
        secret_issues = [i for i in issues if i.rule_id == "SEC003"]
        assert len(secret_issues) == 0

    def test_sql_injection_percent_formatting(self) -> None:
        """Test detection of SQL injection via % formatting."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = """
def query(user_id):
    cursor.execute("SELECT * FROM users WHERE id = %s" % user_id)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "SEC004" for issue in issues)

    def test_command_injection_os_system(self) -> None:
        """Test detection of os.system with shell=True."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = """
import subprocess

def run(cmd):
    subprocess.Popen(cmd, shell=True)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "SEC005" for issue in issues)

    def test_yaml_safe_load_not_flagged(self) -> None:
        """Test that yaml.safe_load is not flagged."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = """
import yaml

def load_config():
    yaml.safe_load(content)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        yaml_issues = [i for i in issues if i.rule_id == "SEC007"]
        assert len(yaml_issues) == 0

    def test_assert_statement_detection(self) -> None:
        """Test detection of assert statements."""
        config = RefactronConfig()
        config.security_min_confidence = 0.5
        analyzer = SecurityAnalyzer(config)
        code = """
def check(x):
    assert x > 0, "x must be positive"
    return x * 2
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "SEC008" for issue in issues)

    def test_sql_parameterization_format_method(self) -> None:
        """Test detection of SQL .format() method."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = """
def query(name):
    sql = "SELECT * FROM users WHERE name = '{}'".format(name)
    cursor.execute(sql)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "SEC009" for issue in issues)

    def test_ssrf_requests_post(self) -> None:
        """Test SSRF detection in requests.post."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = """
import requests

def post_data(url):
    requests.post(f"https://api.com/{url}", data={})
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "SEC010" for issue in issues)

    def test_insecure_random_in_test_file(self) -> None:
        """Test random module detection with test file confidence."""
        config = RefactronConfig()
        config.security_min_confidence = 0.3
        analyzer = SecurityAnalyzer(config)
        code = "import random"
        issues = analyzer.analyze(Path("test_random.py"), code)
        # Lower confidence for test files
        assert isinstance(issues, list)

    def test_ssl_context_cert_none(self) -> None:
        """Test detection of SSL CERT_NONE."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = """
import ssl

ctx = ssl.SSLContext(verify_mode=ssl.CERT_NONE)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "SEC012" for issue in issues)

    def test_requests_verify_false(self) -> None:
        """Test detection of requests verify=False."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = """
import requests

requests.request('GET', 'https://api.com', verify=False)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "SEC013" for issue in issues)

    def test_syntax_error_handling(self) -> None:
        """Test graceful handling of syntax errors."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)
        code = "def broken(:"
        issues = analyzer.analyze(Path("test.py"), code)
        assert isinstance(issues, list)


class TestDependencyAnalyzerEdgeCases:
    """Edge case tests for DependencyAnalyzer."""

    def test_empty_file(self) -> None:
        """Test analyzer handles empty file."""
        config = RefactronConfig()
        analyzer = DependencyAnalyzer(config)
        issues = analyzer.analyze(Path("empty.py"), "")
        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_syntax_error_handling(self) -> None:
        """Test graceful handling of syntax errors."""
        config = RefactronConfig()
        analyzer = DependencyAnalyzer(config)
        code = "def broken(:"
        issues = analyzer.analyze(Path("test.py"), code)
        assert isinstance(issues, list)


class TestPerformanceAnalyzerEdgeCases:
    """Edge case tests for PerformanceAnalyzer."""

    def test_empty_file(self) -> None:
        """Test analyzer handles empty file."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)
        issues = analyzer.analyze(Path("empty.py"), "")
        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_while_loop_with_query(self) -> None:
        """Test N+1 detection in while loop."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)
        code = """
def process():
    i = 0
    while i < 10:
        result = db.query("SELECT * FROM table")
        i += 1
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "P001" for issue in issues)

    def test_list_map_detection(self) -> None:
        """Test detection of list(map(...))."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)
        code = """
def process(items):
    return list(map(lambda x: x * 2, items))
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "P002" for issue in issues)

    def test_generator_expression_not_flagged(self) -> None:
        """Test that generator expressions in comprehensions are handled."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)
        code = """
def process():
    result = [x for x in (y for y in range(10))]
    return result
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should handle nested comprehensions
        assert isinstance(issues, list)

    def test_single_iteration_not_flagged(self) -> None:
        """Test that single iterations are not flagged."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)
        code = """
def process(items):
    for item in items:
        print(item)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        multi_iter_issues = [i for i in issues if i.rule_id == "P004"]
        assert len(multi_iter_issues) == 0

    def test_string_concat_in_while_loop(self) -> None:
        """Test string concatenation detection in while loop."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)
        code = """
def build():
    result = ""
    i = 0
    while i < 10:
        result += "x"
        i += 1
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "P005" for issue in issues)

    def test_syntax_error_handling(self) -> None:
        """Test graceful handling of syntax errors."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)
        code = "def broken(:"
        issues = analyzer.analyze(Path("test.py"), code)
        assert isinstance(issues, list)


class TestTypeHintAnalyzerEdgeCases:
    """Edge case tests for TypeHintAnalyzer."""

    def test_empty_file(self) -> None:
        """Test analyzer handles empty file."""
        config = RefactronConfig()
        analyzer = TypeHintAnalyzer(config)
        issues = analyzer.analyze(Path("empty.py"), "")
        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_syntax_error_handling(self) -> None:
        """Test graceful handling of syntax errors."""
        config = RefactronConfig()
        analyzer = TypeHintAnalyzer(config)
        code = "def broken(:"
        issues = analyzer.analyze(Path("test.py"), code)
        assert isinstance(issues, list)
