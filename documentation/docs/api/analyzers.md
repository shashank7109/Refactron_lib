# refactron.analyzers

Analyzers for detecting code issues and patterns.

## Classes

## Functions


---

# refactron.analyzers.base_analyzer

Base analyzer class.

## Classes

### BaseAnalyzer

```python
BaseAnalyzer(config: refactron.core.config.RefactronConfig)
```

Base class for all analyzers.

#### BaseAnalyzer.__init__

```python
BaseAnalyzer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the analyzer.

Args:
    config: Refactron configuration

#### BaseAnalyzer.analyze

```python
BaseAnalyzer.analyze(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.CodeIssue]
```

Analyze source code and return detected issues.

Args:
    file_path: Path to the file being analyzed
    source_code: Source code content

Returns:
    List of detected code issues

#### BaseAnalyzer.parse_astroid

```python
BaseAnalyzer.parse_astroid(self, source_code: str, file_path: Optional[pathlib._local.Path] = None) -> Any
```

Helper to parse code into an astroid tree.

Args:
    source_code: The code to parse
    file_path: Optional path for module naming context

Returns:
    astroid.nodes.Module

## Functions


---

# refactron.analyzers.code_smell_analyzer

Analyzer for code smells and anti-patterns.

## Classes

### CodeSmellAnalyzer

```python
CodeSmellAnalyzer(config: refactron.core.config.RefactronConfig)
```

Detects common code smells and anti-patterns.

#### CodeSmellAnalyzer.__init__

```python
CodeSmellAnalyzer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the analyzer.

Args:
    config: Refactron configuration

#### CodeSmellAnalyzer.analyze

```python
CodeSmellAnalyzer.analyze(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.CodeIssue]
```

Analyze code for smells and anti-patterns.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of detected code smell issues

#### CodeSmellAnalyzer.parse_astroid

```python
CodeSmellAnalyzer.parse_astroid(self, source_code: str, file_path: Optional[pathlib._local.Path] = None) -> Any
```

Helper to parse code into an astroid tree.

Args:
    source_code: The code to parse
    file_path: Optional path for module naming context

Returns:
    astroid.nodes.Module

## Functions


---

# refactron.analyzers.complexity_analyzer

Analyzer for code complexity metrics.

## Classes

### ComplexityAnalyzer

```python
ComplexityAnalyzer(config: refactron.core.config.RefactronConfig)
```

Analyzes code complexity using cyclomatic complexity and other metrics.

#### ComplexityAnalyzer.__init__

```python
ComplexityAnalyzer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the analyzer.

Args:
    config: Refactron configuration

#### ComplexityAnalyzer.analyze

```python
ComplexityAnalyzer.analyze(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.CodeIssue]
```

Analyze complexity of the source code.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of complexity-related issues

#### ComplexityAnalyzer.parse_astroid

```python
ComplexityAnalyzer.parse_astroid(self, source_code: str, file_path: Optional[pathlib._local.Path] = None) -> Any
```

Helper to parse code into an astroid tree.

Args:
    source_code: The code to parse
    file_path: Optional path for module naming context

Returns:
    astroid.nodes.Module

## Functions


---

# refactron.analyzers.dead_code_analyzer

Analyzer for dead code - unused functions, variables, and imports.

## Classes

### DeadCodeAnalyzer

```python
DeadCodeAnalyzer(config: refactron.core.config.RefactronConfig)
```

Detects unused code that can be safely removed.

#### DeadCodeAnalyzer.__init__

```python
DeadCodeAnalyzer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the analyzer.

Args:
    config: Refactron configuration

#### DeadCodeAnalyzer.analyze

```python
DeadCodeAnalyzer.analyze(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.CodeIssue]
```

Analyze code for unused elements.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of dead code issues

#### DeadCodeAnalyzer.parse_astroid

```python
DeadCodeAnalyzer.parse_astroid(self, source_code: str, file_path: Optional[pathlib._local.Path] = None) -> Any
```

Helper to parse code into an astroid tree.

Args:
    source_code: The code to parse
    file_path: Optional path for module naming context

Returns:
    astroid.nodes.Module

## Functions


---

# refactron.analyzers.dependency_analyzer

Analyzer for import dependencies and module relationships.

## Classes

### DependencyAnalyzer

```python
DependencyAnalyzer(config: 'RefactronConfig') -> None
```

Analyzes import statements and dependencies.

#### DependencyAnalyzer.__init__

```python
DependencyAnalyzer.__init__(self, config: 'RefactronConfig') -> None
```

Initialize the analyzer.

Args:
    config: Refactron configuration

#### DependencyAnalyzer.analyze

```python
DependencyAnalyzer.analyze(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.CodeIssue]
```

Analyze imports and dependencies.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of dependency-related issues

#### DependencyAnalyzer.parse_astroid

```python
DependencyAnalyzer.parse_astroid(self, source_code: str, file_path: Optional[pathlib._local.Path] = None) -> Any
```

Helper to parse code into an astroid tree.

Args:
    source_code: The code to parse
    file_path: Optional path for module naming context

Returns:
    astroid.nodes.Module

## Functions


---

# refactron.analyzers.performance_analyzer

Analyzer for performance antipatterns.

## Classes

### PerformanceAnalyzer

```python
PerformanceAnalyzer(config: refactron.core.config.RefactronConfig)
```

Detects common performance antipatterns and inefficiencies.

#### PerformanceAnalyzer.__init__

```python
PerformanceAnalyzer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the analyzer.

Args:
    config: Refactron configuration

#### PerformanceAnalyzer.analyze

```python
PerformanceAnalyzer.analyze(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.CodeIssue]
```

Analyze code for performance antipatterns.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of performance-related issues

#### PerformanceAnalyzer.parse_astroid

```python
PerformanceAnalyzer.parse_astroid(self, source_code: str, file_path: Optional[pathlib._local.Path] = None) -> Any
```

Helper to parse code into an astroid tree.

Args:
    source_code: The code to parse
    file_path: Optional path for module naming context

Returns:
    astroid.nodes.Module

## Functions


---

# refactron.analyzers.security_analyzer

Analyzer for security vulnerabilities and unsafe patterns.

## Classes

### SecurityAnalyzer

```python
SecurityAnalyzer(config: refactron.core.config.RefactronConfig)
```

Detects common security vulnerabilities and unsafe code patterns.

#### SecurityAnalyzer.__init__

```python
SecurityAnalyzer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the analyzer.

Args:
    config: Refactron configuration

#### SecurityAnalyzer.analyze

```python
SecurityAnalyzer.analyze(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.CodeIssue]
```

Analyze code for security vulnerabilities.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of security-related issues

#### SecurityAnalyzer.parse_astroid

```python
SecurityAnalyzer.parse_astroid(self, source_code: str, file_path: Optional[pathlib._local.Path] = None) -> Any
```

Helper to parse code into an astroid tree.

Args:
    source_code: The code to parse
    file_path: Optional path for module naming context

Returns:
    astroid.nodes.Module

## Functions


---

# refactron.analyzers.type_hint_analyzer

Analyzer for type hints and type annotations.

## Classes

### TypeHintAnalyzer

```python
TypeHintAnalyzer(config: refactron.core.config.RefactronConfig)
```

Analyzes type hint usage and suggests improvements.

#### TypeHintAnalyzer.__init__

```python
TypeHintAnalyzer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the analyzer.

Args:
    config: Refactron configuration

#### TypeHintAnalyzer.analyze

```python
TypeHintAnalyzer.analyze(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.CodeIssue]
```

Analyze type hints in code.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of type hint issues

#### TypeHintAnalyzer.parse_astroid

```python
TypeHintAnalyzer.parse_astroid(self, source_code: str, file_path: Optional[pathlib._local.Path] = None) -> Any
```

Helper to parse code into an astroid tree.

Args:
    source_code: The code to parse
    file_path: Optional path for module naming context

Returns:
    astroid.nodes.Module

## Functions

