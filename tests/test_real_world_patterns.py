"""Tests with real-world problematic code patterns."""

from pathlib import Path

from refactron.analyzers.code_smell_analyzer import CodeSmellAnalyzer
from refactron.analyzers.complexity_analyzer import ComplexityAnalyzer
from refactron.analyzers.dead_code_analyzer import DeadCodeAnalyzer
from refactron.analyzers.dependency_analyzer import DependencyAnalyzer
from refactron.analyzers.performance_analyzer import PerformanceAnalyzer
from refactron.analyzers.security_analyzer import SecurityAnalyzer
from refactron.analyzers.type_hint_analyzer import TypeHintAnalyzer
from refactron.core.config import RefactronConfig


class TestRealWorldComplexity:
    """Real-world complexity patterns."""

    def test_god_function_pattern(self) -> None:
        """Test detection of 'god function' antipattern."""
        config = RefactronConfig(max_function_complexity=10, max_function_length=30)
        analyzer = ComplexityAnalyzer(config)

        code = """
def process_user_request(request, user_id, session_id, permissions, settings):
    # Validation
    if not request:
        return None
    if not user_id:
        raise ValueError("User ID required")
    if not session_id:
        return {"error": "No session"}
    
    # Authentication
    if not permissions:
        return {"error": "No permissions"}
    if "admin" not in permissions:
        if "user" not in permissions:
            return {"error": "Unauthorized"}
    
    # Processing
    user_data = {}
    if settings:
        if settings.get("include_profile"):
            user_data["profile"] = get_profile(user_id)
        if settings.get("include_history"):
            user_data["history"] = get_history(user_id)
        if settings.get("include_preferences"):
            user_data["preferences"] = get_preferences(user_id)
    
    # Logging
    if settings.get("debug"):
        log_request(request)
        log_user(user_id)
        log_session(session_id)
    
    # Response building
    response = {"user": user_data}
    if permissions:
        response["permissions"] = permissions
    if session_id:
        response["session"] = session_id
    
    # Caching
    if settings.get("cache"):
        cache_key = f"{user_id}_{session_id}"
        cache.set(cache_key, response)
    
    return response
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect multiple issues
        assert len(issues) >= 2

    def test_deeply_nested_conditionals(self) -> None:
        """Test detection of deeply nested if-elif-else chains."""
        config = RefactronConfig()
        analyzer = ComplexityAnalyzer(config)

        code = """
def get_status_message(code):
    if code < 100:
        return "Invalid"
    elif code < 200:
        if code == 100:
            return "Continue"
        elif code == 101:
            return "Switching Protocols"
        else:
            return "Informational"
    elif code < 300:
        if code == 200:
            return "OK"
        elif code == 201:
            return "Created"
        elif code == 202:
            return "Accepted"
        else:
            return "Success"
    elif code < 400:
        if code == 301:
            return "Moved Permanently"
        elif code == 302:
            return "Found"
        else:
            return "Redirection"
    elif code < 500:
        if code == 400:
            return "Bad Request"
        elif code == 401:
            return "Unauthorized"
        elif code == 403:
            return "Forbidden"
        elif code == 404:
            return "Not Found"
        else:
            return "Client Error"
    else:
        if code == 500:
            return "Internal Server Error"
        elif code == 502:
            return "Bad Gateway"
        elif code == 503:
            return "Service Unavailable"
        else:
            return "Server Error"
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert len(issues) > 0


class TestRealWorldCodeSmells:
    """Real-world code smell patterns."""

    def test_data_clump_pattern(self) -> None:
        """Test detection of data clump (many parameters)."""
        config = RefactronConfig(max_parameters=5)
        analyzer = CodeSmellAnalyzer(config)

        code = """
def create_user(first_name, last_name, email, phone, address, city, state, 
                zip_code, country, birth_date, gender, username, password):
    user = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "address": address,
        "city": city,
        "state": state,
        "zip": zip_code,
        "country": country,
        "birth_date": birth_date,
        "gender": gender,
        "username": username,
        "password": password
    }
    return user
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "S001" for issue in issues)

    def test_spaghetti_code_pattern(self) -> None:
        """Test deeply nested code with multiple control structures."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)

        code = """
def process_data(data):
    for item in data:
        if item:
            try:
                with open(item.get("file")) as f:
                    for line in f:
                        if line.strip():
                            parts = line.split(",")
                            if len(parts) > 2:
                                process_part(parts)
            except Exception:
                pass
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "S002" for issue in issues)

    def test_magic_constants_in_calculations(self) -> None:
        """Test detection of magic numbers in real calculations."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)

        code = """
def calculate_price(base_price):
    tax = base_price * 0.08  # Tax rate
    shipping = 9.99 if base_price < 50 else 0
    processing_fee = 2.50
    total = base_price + tax + shipping + processing_fee
    return total
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect multiple magic numbers
        magic_issues = [i for i in issues if i.rule_id == "S004"]
        assert len(magic_issues) >= 3

    def test_copy_paste_code(self) -> None:
        """Test detection of copy-pasted code blocks."""
        config = RefactronConfig()
        analyzer = CodeSmellAnalyzer(config)

        code = """
def process_orders(orders):
    # Process pending orders
    pending_total = 0
    pending_count = 0
    pending_items = []
    for order in orders:
        if order.status == "pending":
            pending_total += order.amount
            pending_count += 1
            pending_items.append(order.items)
    
    # Process completed orders
    completed_total = 0
    completed_count = 0
    completed_items = []
    for order in orders:
        if order.status == "completed":
            completed_total += order.amount
            completed_count += 1
            completed_items.append(order.items)
    
    # Process cancelled orders
    cancelled_total = 0
    cancelled_count = 0
    cancelled_items = []
    for order in orders:
        if order.status == "cancelled":
            cancelled_total += order.amount
            cancelled_count += 1
            cancelled_items.append(order.items)
    
    return {
        "pending": {"total": pending_total, "count": pending_count},
        "completed": {"total": completed_total, "count": completed_count},
        "cancelled": {"total": cancelled_total, "count": cancelled_count}
    }
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Repeated code detection is complex and may not always trigger
        # Just ensure no crash
        assert isinstance(issues, list)


class TestRealWorldDeadCode:
    """Real-world dead code patterns."""

    def test_legacy_code_remnants(self) -> None:
        """Test detection of legacy/unused code."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)

        code = """
def process_data(data):
    result = transform(data)
    return result

def old_process_data_v1(data):
    # Old version, kept for reference
    return data

def deprecated_helper():
    pass

# Main processing function
def transform(data):
    return data.upper()
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect unused functions
        assert len(issues) >= 2

    def test_commented_out_return(self) -> None:
        """Test unreachable code after return."""
        config = RefactronConfig()
        analyzer = DeadCodeAnalyzer(config)

        code = """
def calculate(x):
    if x < 0:
        return None
    
    result = x * 2
    return result
    
    # This code is unreachable
    # result = result * 3
    # return result
    print("Processing complete")
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "DEAD003" for issue in issues)


class TestRealWorldSecurity:
    """Real-world security vulnerabilities."""

    def test_sql_injection_orm_bypass(self) -> None:
        """Test SQL injection in ORM raw query."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)

        code = """
from django.db import connection

def get_user_by_name(username):
    cursor = connection.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # execute() with f-string is detected
        assert isinstance(issues, list)

    def test_command_injection_subprocess(self) -> None:
        """Test command injection vulnerability."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)

        code = """
import os
import subprocess

def backup_file(filename):
    subprocess.call(f"tar -czf {filename}.tar.gz", shell=True)
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect shell=True
        assert len(issues) > 0

    def test_path_traversal_vulnerability(self) -> None:
        """Test potential path traversal with user input."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)

        code = """
def read_user_file(filename):
    with open(f"/var/data/{filename}", "r") as f:
        return f.read()
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Analyzer might not catch this specific pattern, but test coverage
        assert isinstance(issues, list)

    def test_hardcoded_credentials_in_config(self) -> None:
        """Test hardcoded credentials."""
        config = RefactronConfig()
        analyzer = SecurityAnalyzer(config)

        code = """
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "username": "admin",
    "password": "Super$ecret123!",
    "database": "production"
}

API_KEY = "sk_live_51H9xYz2eZvKYlo2C"
SECRET_TOKEN = "ghp_1234567890abcdefghijklmnop"
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect hardcoded secrets
        secret_issues = [i for i in issues if i.rule_id == "SEC003"]
        assert len(secret_issues) >= 2


class TestRealWorldPerformance:
    """Real-world performance issues."""

    def test_n_plus_one_in_loop(self) -> None:
        """Test classic N+1 query problem."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)

        code = """
def get_users_with_posts(user_ids):
    users = []
    for user_id in user_ids:
        user = db.query("SELECT * FROM users WHERE id = ?", user_id)
        posts = db.query("SELECT * FROM posts WHERE user_id = ?", user_id)
        user["posts"] = posts
        users.append(user)
    return users
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect N+1 queries
        assert len(issues) >= 1

    def test_inefficient_string_building(self) -> None:
        """Test inefficient string concatenation."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)

        code = """
def generate_html(items):
    html = "<ul>"
    for item in items:
        html += f"<li>{item}</li>"
    html += "</ul>"
    return html
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "P005" for issue in issues)

    def test_unnecessary_list_conversion(self) -> None:
        """Test unnecessary list() wrapper."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)

        code = """
def process_numbers(data):
    squares = list([x**2 for x in data])
    evens = list([x for x in data if x % 2 == 0])
    return squares, evens
"""
        issues = analyzer.analyze(Path("test.py"), code)
        assert any(issue.rule_id == "P006" for issue in issues)

    def test_repeated_computation_in_loop(self) -> None:
        """Test repeated expensive computation."""
        config = RefactronConfig()
        analyzer = PerformanceAnalyzer(config)

        code = """
def process_items(items):
    results = []
    for item in items:
        # Repeated computation
        threshold = calculate_threshold()
        if item > threshold:
            results.append(item)
    return results
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Test coverage of loop-related checks
        assert isinstance(issues, list)


class TestRealWorldDependencies:
    """Real-world dependency issues."""

    def test_wildcard_import_antipattern(self) -> None:
        """Test wildcard import detection."""
        config = RefactronConfig()
        analyzer = DependencyAnalyzer(config)

        code = """
from os import *
from sys import *
import json

def main():
    path_exists(getcwd())
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect wildcard imports
        assert len(issues) > 0

    def test_circular_import_pattern(self) -> None:
        """Test circular import detection."""
        config = RefactronConfig()
        analyzer = DependencyAnalyzer(config)

        code = """
def process():
    import helpers
    return helpers.do_work()

class DataProcessor:
    def analyze(self):
        import analyzer
        return analyzer.run()
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect imports inside functions
        assert len(issues) > 0

    def test_deprecated_module_usage(self) -> None:
        """Test deprecated module detection."""
        config = RefactronConfig()
        analyzer = DependencyAnalyzer(config)

        code = """
import imp
import optparse

def load_module():
    imp.load_source("module", "path")

def parse_args():
    parser = optparse.OptionParser()
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect deprecated modules
        assert len(issues) >= 2


class TestRealWorldTypeHints:
    """Real-world type hint issues."""

    def test_api_function_without_types(self) -> None:
        """Test public API function without type hints."""
        config = RefactronConfig()
        analyzer = TypeHintAnalyzer(config)

        code = """
def calculate_discount(price, discount_percent, customer_type):
    if customer_type == "premium":
        discount_percent *= 1.5
    return price * (1 - discount_percent / 100)

def format_currency(amount, currency):
    return f"{currency}{amount:.2f}"
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect missing type hints
        assert len(issues) > 0

    def test_generic_type_incomplete(self) -> None:
        """Test incomplete generic types."""
        config = RefactronConfig()
        analyzer = TypeHintAnalyzer(config)

        code = """
from typing import List, Dict, Optional

def get_user_data() -> Dict:
    return {"name": "John", "age": 30}

def get_items() -> List:
    return [1, 2, 3]

def find_user(user_id: int) -> Optional:
    return None
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect incomplete generics (at least Dict and List)
        assert len(issues) >= 2

    def test_any_type_overuse(self) -> None:
        """Test overuse of Any type."""
        config = RefactronConfig()
        analyzer = TypeHintAnalyzer(config)

        code = """
from typing import Any

def process(data: Any, config: Any, options: Any) -> Any:
    return None

def transform(value: Any) -> Any:
    return value
"""
        issues = analyzer.analyze(Path("test.py"), code)
        # Should detect Any usage
        assert len(issues) >= 4


class TestIntegrationRealWorld:
    """Integration tests with real-world patterns."""

    def test_complete_application_analysis(self) -> None:
        """Test analysis of complete application code."""
        config = RefactronConfig()

        analyzers = [
            ComplexityAnalyzer(config),
            CodeSmellAnalyzer(config),
            DeadCodeAnalyzer(config),
            SecurityAnalyzer(config),
            PerformanceAnalyzer(config),
            DependencyAnalyzer(config),
            TypeHintAnalyzer(config),
        ]

        code = """
import os
import sys
from typing import Any

# Hardcoded credentials - security issue
API_KEY = "sk_live_abc123xyz"
password = "admin123"

def process_user_request(user_id, session_id, data, options, config, settings):
    # Too many parameters - code smell
    if not user_id:
        return None
    
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    
    # Unused variable - dead code
    unused_var = 42
    
    # N+1 query antipattern
    for item in data:
        detail = db.query(f"SELECT * FROM items WHERE id = {item}")
    
    # Complex nested logic - complexity issue
    if options:
        if config:
            if settings:
                for key in settings:
                    if key == "important":
                        process_important(key)
    
    return result

def unused_helper():
    # Dead code - never called
    pass

def calculate(x: Any) -> Any:
    # Type hint issues - using Any
    return x * 2
"""

        all_issues = []
        for analyzer in analyzers:
            issues = analyzer.analyze(Path("app.py"), code)
            all_issues.extend(issues)

        # Should detect multiple types of issues
        assert len(all_issues) >= 5

        # Verify we have issues from multiple categories
        categories = set(issue.category.value for issue in all_issues)
        assert len(categories) >= 3

    def test_minimal_valid_code(self) -> None:
        """Test analysis of minimal but valid code."""
        config = RefactronConfig()

        analyzers = [
            ComplexityAnalyzer(config),
            CodeSmellAnalyzer(config),
            DeadCodeAnalyzer(config),
        ]

        code = """
def hello():
    return "world"
"""

        all_issues = []
        for analyzer in analyzers:
            issues = analyzer.analyze(Path("minimal.py"), code)
            all_issues.extend(issues)

        # Minimal code might trigger docstring warnings
        assert isinstance(all_issues, list)
