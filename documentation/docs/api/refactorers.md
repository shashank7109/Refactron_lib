# refactron.refactorers

Refactorers for automated code transformations.

## Classes

## Functions


---

# refactron.refactorers.add_docstring_refactorer

Refactorer for adding docstrings to functions and classes.

## Classes

### AddDocstringRefactorer

```python
AddDocstringRefactorer(config: refactron.core.config.RefactronConfig)
```

Suggests adding docstrings to undocumented functions and classes.

#### AddDocstringRefactorer.__init__

```python
AddDocstringRefactorer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the refactorer.

Args:
    config: Refactron configuration

#### AddDocstringRefactorer.refactor

```python
AddDocstringRefactorer.refactor(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.RefactoringOperation]
```

Find functions/classes without docstrings and suggest adding them.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of add docstring operations

## Functions


---

# refactron.refactorers.base_refactorer

Base refactorer class.

## Classes

### BaseRefactorer

```python
BaseRefactorer(config: refactron.core.config.RefactronConfig)
```

Base class for all refactorers.

#### BaseRefactorer.__init__

```python
BaseRefactorer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the refactorer.

Args:
    config: Refactron configuration

#### BaseRefactorer.refactor

```python
BaseRefactorer.refactor(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.RefactoringOperation]
```

Analyze source code and return refactoring operations.

Args:
    file_path: Path to the file being refactored
    source_code: Source code content

Returns:
    List of refactoring operations

## Functions


---

# refactron.refactorers.extract_method_refactorer

Refactorer for extracting methods from complex functions.

## Classes

### ExtractMethodRefactorer

```python
ExtractMethodRefactorer(config: refactron.core.config.RefactronConfig)
```

Suggests extracting methods from overly complex functions.

#### ExtractMethodRefactorer.__init__

```python
ExtractMethodRefactorer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the refactorer.

Args:
    config: Refactron configuration

#### ExtractMethodRefactorer.refactor

```python
ExtractMethodRefactorer.refactor(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.RefactoringOperation]
```

Find opportunities to extract methods.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of extract method operations

## Functions


---

# refactron.refactorers.magic_number_refactorer

Refactorer for extracting magic numbers into named constants.

## Classes

### MagicNumberRefactorer

```python
MagicNumberRefactorer(config: refactron.core.config.RefactronConfig)
```

Suggests extracting magic numbers into named constants.

#### MagicNumberRefactorer.__init__

```python
MagicNumberRefactorer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the refactorer.

Args:
    config: Refactron configuration

#### MagicNumberRefactorer.refactor

```python
MagicNumberRefactorer.refactor(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.RefactoringOperation]
```

Find magic numbers and suggest extracting them to constants.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of extract constant operations

## Functions


---

# refactron.refactorers.reduce_parameters_refactorer

Refactorer for reducing function parameters using configuration objects.

## Classes

### ReduceParametersRefactorer

```python
ReduceParametersRefactorer(config: refactron.core.config.RefactronConfig)
```

Suggests using configuration objects for functions with many parameters.

#### ReduceParametersRefactorer.__init__

```python
ReduceParametersRefactorer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the refactorer.

Args:
    config: Refactron configuration

#### ReduceParametersRefactorer.refactor

```python
ReduceParametersRefactorer.refactor(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.RefactoringOperation]
```

Find functions with too many parameters and suggest config objects.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of parameter reduction operations

## Functions


---

# refactron.refactorers.simplify_conditionals_refactorer

Refactorer for simplifying complex conditional statements.

## Classes

### SimplifyConditionalsRefactorer

```python
SimplifyConditionalsRefactorer(config: refactron.core.config.RefactronConfig)
```

Suggests simplifying deeply nested conditionals.

#### SimplifyConditionalsRefactorer.__init__

```python
SimplifyConditionalsRefactorer.__init__(self, config: refactron.core.config.RefactronConfig)
```

Initialize the refactorer.

Args:
    config: Refactron configuration

#### SimplifyConditionalsRefactorer.refactor

```python
SimplifyConditionalsRefactorer.refactor(self, file_path: pathlib._local.Path, source_code: str) -> List[refactron.core.models.RefactoringOperation]
```

Find deeply nested conditionals and suggest simplifications.

Args:
    file_path: Path to the file
    source_code: Source code content

Returns:
    List of simplification operations

## Functions
