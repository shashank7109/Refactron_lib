# refactron.autofix

Auto-fix module for automatically fixing code issues.

This module provides rule-based code fixes without requiring expensive AI APIs.
All fixers use AST analysis and pattern matching for fast, reliable transformations.

## Classes

## Functions


---

# refactron.autofix.engine

Auto-fix engine for applying rule-based code transformations.

This engine uses AST analysis and pattern matching to apply safe
automatic fixes without requiring expensive AI APIs.

## Classes

### AutoFixEngine

```python
AutoFixEngine(safety_level: refactron.autofix.models.FixRiskLevel = <FixRiskLevel.SAFE: 0.0>)
```

Engine for applying automatic fixes to code issues.

All fixes use rule-based AST transformations for reliability
and performance. No expensive AI APIs required!

#### AutoFixEngine.__init__

```python
AutoFixEngine.__init__(self, safety_level: refactron.autofix.models.FixRiskLevel = <FixRiskLevel.SAFE: 0.0>)
```

Initialize the auto-fix engine.

Args:
    safety_level: Maximum risk level to apply automatically

#### AutoFixEngine.can_fix

```python
AutoFixEngine.can_fix(self, issue: refactron.core.models.CodeIssue) -> bool
```

Check if an issue can be auto-fixed.

Args:
    issue: The issue to check

Returns:
    True if a fixer is available, False otherwise

#### AutoFixEngine.fix

```python
AutoFixEngine.fix(self, issue: refactron.core.models.CodeIssue, code: str, preview: bool = True) -> refactron.autofix.models.FixResult
```

Apply automatic fix to an issue.

Args:
    issue: The issue to fix
    code: The original code
    preview: If True, only preview changes (don't apply)

Returns:
    FixResult with success status and details

#### AutoFixEngine.fix_all

```python
AutoFixEngine.fix_all(self, issues: list, code: str, preview: bool = True) -> Dict[int, refactron.autofix.models.FixResult]
```

Apply fixes to multiple issues.

Args:
    issues: List of issues to fix
    code: The original code
    preview: If True, only preview changes

Returns:
    Dictionary mapping issue index to fix result

### BaseFixer

```python
BaseFixer(name: str, risk_score: float = 0.0)
```

Base class for all automatic fixers.

#### BaseFixer.__init__

```python
BaseFixer.__init__(self, name: str, risk_score: float = 0.0)
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### BaseFixer.apply

```python
BaseFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply the fix.

Args:
    issue: The issue to fix
    code: The original code

Returns:
    FixResult with fixed code

#### BaseFixer.preview

```python
BaseFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview the fix without applying it.

Args:
    issue: The issue to fix
    code: The original code

Returns:
    FixResult with diff showing proposed changes

## Functions


---

# refactron.autofix.file_ops

File operations for auto-fix system with backup and rollback support.

## Classes

### FileOperations

```python
FileOperations(backup_dir: Optional[pathlib._local.Path] = None)
```

Handle file operations with safety guarantees.

#### FileOperations.__init__

```python
FileOperations.__init__(self, backup_dir: Optional[pathlib._local.Path] = None)
```

Initialize file operations.

Args:
    backup_dir: Directory for backups (default: .refactron_backups)

#### FileOperations.backup_file

```python
FileOperations.backup_file(self, filepath: pathlib._local.Path) -> pathlib._local.Path
```

Create a backup of a file.

Args:
    filepath: Path to file to backup

Returns:
    Path to backup file

#### FileOperations.clear_backups

```python
FileOperations.clear_backups(self) -> int
```

Clear all backups.

Returns:
    Number of backups cleared

#### FileOperations.list_backups

```python
FileOperations.list_backups(self) -> List[Any]
```

List all backups.

Returns:
    List of backup information

#### FileOperations.rollback_all

```python
FileOperations.rollback_all(self) -> int
```

Rollback all backed up files.

Returns:
    Number of files rolled back

#### FileOperations.rollback_file

```python
FileOperations.rollback_file(self, filepath: pathlib._local.Path) -> bool
```

Rollback a file to its last backup.

Args:
    filepath: Path to file to rollback

Returns:
    True if successful, False otherwise

#### FileOperations.write_with_backup

```python
FileOperations.write_with_backup(self, filepath: pathlib._local.Path, content: str) -> Dict
```

Write content to file with automatic backup.

Args:
    filepath: Path to file to write
    content: Content to write

Returns:
    Dictionary with operation details

## Functions


---

# refactron.autofix.fixers

Concrete fixer implementations for common code issues.

All fixers use AST-based transformations for reliability and speed.
No expensive AI APIs required!

## Classes

### AddDocstringsFixer

```python
AddDocstringsFixer() -> None
```

Adds missing docstrings to functions and classes.

#### AddDocstringsFixer.__init__

```python
AddDocstringsFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### AddDocstringsFixer.apply

```python
AddDocstringsFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply docstring addition.

#### AddDocstringsFixer.preview

```python
AddDocstringsFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview docstring addition.

### AddMissingCommasFixer

```python
AddMissingCommasFixer() -> None
```

Add missing trailing commas in lists/dicts.

#### AddMissingCommasFixer.__init__

```python
AddMissingCommasFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### AddMissingCommasFixer.apply

```python
AddMissingCommasFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply comma addition.

#### AddMissingCommasFixer.preview

```python
AddMissingCommasFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview comma addition.

### ConvertToFStringFixer

```python
ConvertToFStringFixer() -> None
```

Convert old-style format strings to f-strings.

#### ConvertToFStringFixer.__init__

```python
ConvertToFStringFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### ConvertToFStringFixer.apply

```python
ConvertToFStringFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply f-string conversion.

#### ConvertToFStringFixer.preview

```python
ConvertToFStringFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview f-string conversion.

### ExtractMagicNumbersFixer

```python
ExtractMagicNumbersFixer() -> None
```

Extracts magic numbers into named constants.

#### ExtractMagicNumbersFixer.__init__

```python
ExtractMagicNumbersFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### ExtractMagicNumbersFixer.apply

```python
ExtractMagicNumbersFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply magic number extraction.

#### ExtractMagicNumbersFixer.preview

```python
ExtractMagicNumbersFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview magic number extraction.

### FixIndentationFixer

```python
FixIndentationFixer(spaces: int = 4)
```

Fix inconsistent indentation.

#### FixIndentationFixer.__init__

```python
FixIndentationFixer.__init__(self, spaces: int = 4)
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### FixIndentationFixer.apply

```python
FixIndentationFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply indentation fix.

#### FixIndentationFixer.preview

```python
FixIndentationFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview indentation fix.

### FixTypeHintsFixer

```python
FixTypeHintsFixer() -> None
```

Adds or fixes type hints.

#### FixTypeHintsFixer.__init__

```python
FixTypeHintsFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### FixTypeHintsFixer.apply

```python
FixTypeHintsFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply type hint fix.

#### FixTypeHintsFixer.preview

```python
FixTypeHintsFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview type hint fix.

### NormalizeQuotesFixer

```python
NormalizeQuotesFixer(prefer_double: bool = True)
```

Normalize string quotes (single → double or vice versa).

#### NormalizeQuotesFixer.__init__

```python
NormalizeQuotesFixer.__init__(self, prefer_double: bool = True)
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### NormalizeQuotesFixer.apply

```python
NormalizeQuotesFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply quote normalization.

#### NormalizeQuotesFixer.preview

```python
NormalizeQuotesFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview quote normalization.

### RemoveDeadCodeFixer

```python
RemoveDeadCodeFixer() -> None
```

Removes dead/unreachable code.

#### RemoveDeadCodeFixer.__init__

```python
RemoveDeadCodeFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### RemoveDeadCodeFixer.apply

```python
RemoveDeadCodeFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply dead code removal.

#### RemoveDeadCodeFixer.preview

```python
RemoveDeadCodeFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview dead code removal.

### RemovePrintStatementsFixer

```python
RemovePrintStatementsFixer(convert_to_logging: bool = False)
```

Remove or convert print statements to logging.

#### RemovePrintStatementsFixer.__init__

```python
RemovePrintStatementsFixer.__init__(self, convert_to_logging: bool = False)
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### RemovePrintStatementsFixer.apply

```python
RemovePrintStatementsFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply print statement removal/conversion.

#### RemovePrintStatementsFixer.preview

```python
RemovePrintStatementsFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview print statement removal/conversion.

### RemoveTrailingWhitespaceFixer

```python
RemoveTrailingWhitespaceFixer() -> None
```

Remove trailing whitespace from lines.

#### RemoveTrailingWhitespaceFixer.__init__

```python
RemoveTrailingWhitespaceFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### RemoveTrailingWhitespaceFixer.apply

```python
RemoveTrailingWhitespaceFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply whitespace removal.

#### RemoveTrailingWhitespaceFixer.preview

```python
RemoveTrailingWhitespaceFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview whitespace removal.

### RemoveUnusedImportsFixer

```python
RemoveUnusedImportsFixer() -> None
```

Removes unused import statements.

#### RemoveUnusedImportsFixer.__init__

```python
RemoveUnusedImportsFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### RemoveUnusedImportsFixer.apply

```python
RemoveUnusedImportsFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply the fix to remove unused imports.

#### RemoveUnusedImportsFixer.preview

```python
RemoveUnusedImportsFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview the removal of unused imports.

### RemoveUnusedVariablesFixer

```python
RemoveUnusedVariablesFixer() -> None
```

Remove unused variables.

#### RemoveUnusedVariablesFixer.__init__

```python
RemoveUnusedVariablesFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### RemoveUnusedVariablesFixer.apply

```python
RemoveUnusedVariablesFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply unused variable removal.

#### RemoveUnusedVariablesFixer.preview

```python
RemoveUnusedVariablesFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview unused variable removal.

### SimplifyBooleanFixer

```python
SimplifyBooleanFixer() -> None
```

Simplify boolean expressions.

#### SimplifyBooleanFixer.__init__

```python
SimplifyBooleanFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### SimplifyBooleanFixer.apply

```python
SimplifyBooleanFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply boolean simplification.

#### SimplifyBooleanFixer.preview

```python
SimplifyBooleanFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview boolean simplification.

### SortImportsFixer

```python
SortImportsFixer() -> None
```

Sort and organize imports using isort.

#### SortImportsFixer.__init__

```python
SortImportsFixer.__init__(self) -> None
```

Initialize a fixer.

Args:
    name: Name of the fixer
    risk_score: Risk level (0.0 = safe, 1.0 = dangerous)

#### SortImportsFixer.apply

```python
SortImportsFixer.apply(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Apply import sorting.

#### SortImportsFixer.preview

```python
SortImportsFixer.preview(self, issue: refactron.core.models.CodeIssue, code: str) -> refactron.autofix.models.FixResult
```

Preview import sorting.

## Functions


---

# refactron.autofix.models

Models for auto-fix system.

## Classes

### FixResult

```python
FixResult(success: bool, reason: str = '', diff: Optional[str] = None, original: Optional[str] = None, fixed: Optional[str] = None, risk_score: float = 1.0, files_affected: List[str] = None) -> None
```

Result of an automatic fix.

#### FixResult.__init__

```python
FixResult.__init__(self, success: bool, reason: str = '', diff: Optional[str] = None, original: Optional[str] = None, fixed: Optional[str] = None, risk_score: float = 1.0, files_affected: List[str] = None) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### FixRiskLevel

```python
FixRiskLevel(*values)
```

Risk levels for automatic fixes.

## Functions

