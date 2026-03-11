# CLI Reference

This document contains the reference for all Refactron CLI commands.

## Global Options

```bash

╭────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                        │
│                                              ⚡ REFACTRON                                              │
│                                      INTELLIGENT CODE REFACTORING                                      │
│                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯

COMMAND CENTER
Select a command by name or number


        ID     COMMAND                        DESCRIPTION
 ────────────────────────────────────────────────────────────────────────────────────────────────────────
        01     ANALYZE                        Analyze code for issues and technical debt.
        02     AUTH                           Manage authentication state.
        03     AUTOFIX                        Automatically fix code issues (Phase 3...
        04     DOCUMENT                       Generate Google-style docstrings for a...
        05     FEEDBACK                       Provide feedback on a refactoring operation.
        06     GENERATE-CICD                  Generate CI/CD integration templates.
        07     INIT                           Initialize Refactron configuration in the...
        08     LOGIN                          Log in to Refactron CLI via device-code flow.
        09     LOGOUT                         Log out of Refactron CLI.
        10     METRICS                        Display collected metrics from the current...
        11     PATTERNS                       Pattern learning and project-specific...
        12     RAG                            RAG (Retrieval-Augmented Generation)...
        13     REFACTOR                       Refactor code with intelligent...
        14     REPO                           Manage GitHub repository connections.
        15     REPORT                         Generate a detailed technical debt report.
        16     ROLLBACK                       Rollback refactoring changes to restore...
        17     SERVE-METRICS                  Start a Prometheus metrics HTTP server.
        18     SUGGEST                        Generate AI-powered refactoring suggestions.
        19     TELEMETRY                      Manage telemetry settings.


GLOBAL OPTIONS
--version  Show the version and exit.
--help     Show this message and exit.

USAGE: refactron <command> ...
EXAMPLE: refactron analyze . --detailed



```

## analyze

```bash
Usage: refactron analyze [OPTIONS] [TARGET]

  Analyze code for issues and technical debt.

  TARGET: Path to file or directory to analyze (optional if workspace is
  connected)

Options:
  -c, --config PATH               Path to configuration file
  --detailed / --summary          Show detailed or summary report
  --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                  Set log level
  --log-format [json|text]        Set log format (json for CI/CD, text for
                                  console)
  --metrics / --no-metrics        Enable or disable metrics collection
  --show-metrics                  Show metrics summary after analysis
  -p, --profile [dev|staging|prod]
                                  Named configuration profile to use (dev,
                                  staging, prod). Profiles typically group
                                  config defaults; if both --profile and
                                  --environment are set, the environment
                                  determines the final effective configuration.
  -e, --environment [dev|staging|prod]
                                  Target runtime environment (dev, staging,
                                  prod). When both --profile and --environment
                                  are provided, the environment overrides the
                                  selected profile.
  --help                          Show this message and exit.

```

## auth

```bash
Usage: refactron auth [OPTIONS] COMMAND [ARGS]...

  Manage authentication state.

Options:
  --help  Show this message and exit.

Commands:
  logout  Log out of Refactron CLI.
  status  Show current authentication status.

```

## autofix

```bash
Usage: refactron autofix [OPTIONS] TARGET

  Automatically fix code issues (Phase 3 feature).

  TARGET: Path to file or directory to fix

  Examples:   refactron autofix myfile.py --preview   refactron autofix
  myproject/ --apply --safety-level moderate

Options:
  -c, --config PATH               Path to configuration file
  -p, --profile [dev|staging|prod]
                                  Named configuration profile to use (dev,
                                  staging, prod). Profiles typically group
                                  config defaults; if both --profile and
                                  --environment are set, the environment
                                  determines the final effective configuration.
  -e, --environment [dev|staging|prod]
                                  Target runtime environment (dev, staging,
                                  prod). When both --profile and --environment
                                  are provided, the environment overrides the
                                  selected profile.
  --preview / --apply             Preview fixes or apply them
  -s, --safety-level [safe|low|moderate|high]
                                  Maximum risk level for automatic fixes
  --help                          Show this message and exit.

```

## document

```bash
Usage: refactron document [OPTIONS] TARGET

  Generate Google-style docstrings for a Python file.

  Uses AI to analyze code and add comprehensive documentation.

Options:
  --apply / --no-apply            Apply the documentation changes to the file
  --interactive / --no-interactive
                                  Use interactive mode for apply
  --help                          Show this message and exit.

```

## feedback

```bash
Usage: refactron feedback [OPTIONS] OPERATION_ID

  Provide feedback on a refactoring operation.

  OPERATION_ID: The unique identifier of the refactoring operation

  Examples:   refactron feedback abc-123 --action accepted --reason "Improved
  readability"   refactron feedback xyz-789 --action rejected --reason "Too
  risky"

Options:
  -a, --action [accepted|rejected|ignored]
                                  Feedback action: accepted, rejected, or
                                  ignored  [required]
  -r, --reason TEXT               Optional reason for the feedback
  -c, --config PATH               Path to configuration file
  --help                          Show this message and exit.

```

## generate-cicd

```bash
Usage: refactron generate-cicd [OPTIONS] {github|gitlab|pre-commit|all}

  Generate CI/CD integration templates.

  TYPE: Type of template to generate (github, gitlab, pre-commit, all)

  Examples:   refactron generate-cicd github --output .github/workflows
  refactron generate-cicd gitlab --output .   refactron generate-cicd pre-commit
  --output .   refactron generate-cicd all --output .

Options:
  -o, --output PATH               Output directory (default: current directory)
  --python-versions TEXT          Comma-separated Python versions (default:
                                  3.8,3.9,3.10,3.11,3.12)
  --fail-on-critical / --no-fail-on-critical
                                  Fail build on critical issues (default: True)
  --fail-on-errors / --no-fail-on-errors
                                  Fail build on error-level issues (default:
                                  False)
  --max-critical INTEGER          Maximum allowed critical issues (default: 0)
  --max-errors INTEGER            Maximum allowed error-level issues (default:
                                  10)
  --help                          Show this message and exit.

```

## init

```bash
Usage: refactron init [OPTIONS]

  Initialize Refactron configuration in the current directory.

Options:
  -t, --template [base|django|fastapi|flask]
                                  Configuration template to use (base, django,
                                  fastapi, flask)
  --help                          Show this message and exit.

```

## login

```bash
Usage: refactron login [OPTIONS]

  Log in to Refactron CLI via device-code flow.

Options:
  --api-base-url TEXT  Refactron API base URL  [default:
                       https://api.refactron.dev]
  --no-browser         Do not open a browser automatically (print the URL
                       instead)
  --timeout INTEGER    HTTP timeout in seconds for each request  [default: 10]
  --force              Force re-login even if already logged in
  --help               Show this message and exit.

```

## logout

```bash
Usage: refactron logout [OPTIONS]

  Log out of Refactron CLI.

Options:
  --help  Show this message and exit.

```

## metrics

```bash
Usage: refactron metrics [OPTIONS]

  Display collected metrics from the current session.

  Shows performance metrics, analyzer hit counts, and other statistics from
  Refactron operations.

  Examples:   refactron metrics              # Show metrics in text format
  refactron metrics --format json  # Show metrics in JSON format

Options:
  -f, --format [text|json]  Output format
  --help                    Show this message and exit.

```

## patterns

```bash
Usage: refactron patterns [OPTIONS] COMMAND [ARGS]...

  Pattern learning and project-specific tuning commands.

Options:
  --help  Show this message and exit.

Commands:
  analyze    Analyze learned patterns for a specific project.
  profile    Show the current pattern profile for a project.
  recommend  Show rule tuning recommendations for a project.
  tune       Apply tuning recommendations to the project profile.

```

## rag

```bash
Usage: refactron rag [OPTIONS] COMMAND [ARGS]...

  RAG (Retrieval-Augmented Generation) management commands.

Options:
  --help  Show this message and exit.

Commands:
  index   Index the current workspace for RAG retrieval.
  search  Search the RAG index for similar code.
  status  Show RAG index statistics.

```

## refactor

```bash
Usage: refactron refactor [OPTIONS] [TARGET]

  Refactor code with intelligent transformations.

  TARGET: Path to file or directory to refactor (optional if workspace is
  connected)

Options:
  -c, --config PATH               Path to configuration file
  -p, --profile [dev|staging|prod]
                                  Named configuration profile to use (dev,
                                  staging, prod). Profiles typically group
                                  config defaults; if both --profile and
                                  --environment are set, the environment
                                  determines the final effective configuration.
  -e, --environment [dev|staging|prod]
                                  Target runtime environment (dev, staging,
                                  prod). When both --profile and --environment
                                  are provided, the environment overrides the
                                  selected profile.
  --preview / --apply             Preview changes or apply them
  -t, --types TEXT                Specific refactoring types to apply
  --feedback / --no-feedback      Collect interactive feedback on refactoring
                                  suggestions
  --help                          Show this message and exit.

```

## repo

```bash
Usage: refactron repo [OPTIONS] COMMAND [ARGS]...

  Manage GitHub repository connections.

Options:
  --help  Show this message and exit.

Commands:
  connect     Connect to a GitHub repository.
  disconnect  Disconnect a repository and optionally delete local files.
  list        List all GitHub repositories connected to your account.

```

## report

```bash
Usage: refactron report [OPTIONS] TARGET

  Generate a detailed technical debt report.

  TARGET: Path to file or directory to analyze

Options:
  -c, --config PATH               Path to configuration file
  -p, --profile [dev|staging|prod]
                                  Configuration profile to use (dev, staging,
                                  prod)
  -e, --environment [dev|staging|prod]
                                  Environment to use (overrides profile)
  -f, --format [text|json|html]   Report format
  -o, --output PATH               Output file path
  --help                          Show this message and exit.

```

## rollback

```bash
Usage: refactron rollback [OPTIONS] [SESSION_ID]

  Rollback refactoring changes to restore original files.

  By default, restores files from the latest backup session.

  Arguments:     SESSION_ID: Optional specific session ID to rollback.

  Examples:   refactron rollback              # Rollback latest session
  refactron rollback session_123  # Rollback specific session   refactron
  rollback --list       # List all backup sessions   refactron rollback --use-
  git    # Use Git rollback   refactron rollback --clear      # Clear all
  backups

Options:
  -s, --session TEXT  Specific session ID to rollback (deprecated, use argument
                      instead)
  --use-git           Use Git rollback instead of file backup
  --list              List all backup sessions
  --clear             Clear all backup sessions
  --help              Show this message and exit.

```

## serve-metrics

```bash
Usage: refactron serve-metrics [OPTIONS]

  Start a Prometheus metrics HTTP server.

  This command starts a persistent HTTP server that exposes Refactron metrics in
  Prometheus format on the /metrics endpoint.

  Examples:   refactron serve-metrics                    # Start on 0.0.0.0:9090
  refactron serve-metrics --port 8080        # Start on port 8080   refactron
  serve-metrics --host 127.0.0.1   # Bind to localhost only

Options:
  --host TEXT     Host to bind Prometheus metrics server to (default: 127.0.0.1
                  for localhost-only)
  --port INTEGER  Port for Prometheus metrics server
  --help          Show this message and exit.

```

## suggest

```bash
Usage: refactron suggest [OPTIONS] [TARGET]

  Generate AI-powered refactoring suggestions.

  Uses RAG and LLM to analyze code and propose fixes.

Options:
  --line INTEGER                  Specific line number to fix
  --interactive / --no-interactive
                                  Use interactive mode
  --apply / --no-apply            Apply the suggested changes to the file
  --help                          Show this message and exit.

```

## telemetry

```bash
Usage: refactron telemetry [OPTIONS]

  Manage telemetry settings.

Options:
  --enable   Enable telemetry collection
  --disable  Disable telemetry collection
  --status   Show current telemetry status
  --help     Show this message and exit.

```
