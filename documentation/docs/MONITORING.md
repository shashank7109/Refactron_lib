# Refactron Logging & Monitoring

This document describes the logging and monitoring features added to Refactron in Phase 3.2.

## Table of Contents

- [Overview](#overview)
- [Structured Logging](#structured-logging)
- [Metrics Collection](#metrics-collection)
- [Prometheus Integration](#prometheus-integration)
- [Telemetry](#telemetry)
- [Configuration](#configuration)
- [CLI Commands](#cli-commands)
- [Examples](#examples)

## Overview

Refactron now includes comprehensive logging and monitoring capabilities designed for production environments:

- **Structured Logging**: JSON-formatted logs for CI/CD and log aggregation systems
- **Metrics Collection**: Track analysis time, success rates, and rule hit counts
- **Prometheus Integration**: Expose metrics via HTTP endpoint for monitoring dashboards
- **Opt-in Telemetry**: Anonymous usage data to understand real-world usage patterns

## Structured Logging

### Features

- JSON or text format logging
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Automatic log rotation with configurable file size and backup count
- Console and file logging can be enabled/disabled independently
- Timestamps in UTC with ISO 8601 format

### Configuration

```yaml
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_level: INFO

# Log format: json (for CI/CD) or text (for console)
log_format: json

# Log file location (null = default: ~/.refactron/logs/refactron.log)
log_file: null

# Maximum log file size before rotation (in bytes)
log_max_bytes: 10485760  # 10 MB

# Number of backup log files to keep
log_backup_count: 5

# Enable/disable console and file logging
enable_console_logging: true
enable_file_logging: true
```

### Example JSON Log Entry

```json
{
  "timestamp": "2026-01-02T17:43:58.520137Z",
  "level": "INFO",
  "logger": "refactron.core.refactron",
  "message": "Analysis completed successfully",
  "module": "refactron",
  "function": "analyze",
  "line": 205,
  "file_path": "/path/to/file.py",
  "duration_ms": 123.45
}
```

## Metrics Collection

### Features

Refactron tracks detailed metrics about its operations:

**Analysis Metrics:**
- Total files analyzed and failed
- Total issues found
- Analysis time per file and total run time
- Success rate percentage
- Analyzer hit counts (how many times each analyzer found issues)
- Issue type distribution

**Refactoring Metrics:**
- Total refactorings applied and failed
- Refactoring time per operation
- Success rate percentage
- Refactorer hit counts
- Risk level distribution

### Configuration

```yaml
# Enable metrics collection
enable_metrics: true

# Track detailed per-file metrics
metrics_detailed: true
```

### Viewing Metrics

```bash
# View metrics in text format
refactron metrics

# View metrics in JSON format
refactron metrics --format json
```

### Example Metrics Output

```
📈 Refactron Metrics

Analysis Metrics:
  Files analyzed: 45
  Files failed: 2
  Issues found: 123
  Total time: 5234.56ms
  Avg time per file: 116.32ms
  Success rate: 95.6%

Analyzer Hit Counts:
  complexity: 23
  security: 15
  type_hints: 45
  code_smells: 30
  dead_code: 10

Refactoring Metrics:
  Applied: 15
  Failed: 1
  Total time: 2345.67ms
  Success rate: 93.8%

Refactorer Hit Counts:
  extract_method: 5
  simplify_conditionals: 4
  add_docstring: 6
```

## Prometheus Integration

### Features

- HTTP server exposing metrics in Prometheus format
- Standard Prometheus metric types (counter, gauge)
- `/metrics` endpoint for Prometheus scraping
- `/health` endpoint for health checks
- Configurable host and port

### Configuration

```yaml
# Enable Prometheus metrics endpoint
enable_prometheus: true

# Prometheus server host
prometheus_host: "0.0.0.0"

# Prometheus server port
prometheus_port: 9090
```

### Starting the Metrics Server

```bash
# Start server on default port 9090
refactron serve-metrics

# Start server on custom port
refactron serve-metrics --port 8080

# Bind to specific host
refactron serve-metrics --host 127.0.0.1 --port 9090
```

### Prometheus Metrics

All metrics are prefixed with `refactron_`:

**Counters:**
- `refactron_files_analyzed_total` - Total number of files analyzed
- `refactron_files_failed_total` - Total number of files that failed analysis
- `refactron_issues_found_total` - Total number of issues found
- `refactron_refactorings_applied_total` - Total number of refactorings applied
- `refactron_refactorings_failed_total` - Total number of refactorings that failed
- `refactron_analyzer_hits_total{analyzer="name"}` - Hit count per analyzer
- `refactron_refactorer_hits_total{refactorer="name"}` - Hit count per refactorer

**Gauges:**
- `refactron_analysis_duration_ms` - Total analysis duration
- `refactron_avg_analysis_time_per_file_ms` - Average analysis time per file
- `refactron_analysis_success_rate` - Analysis success rate percentage
- `refactron_refactoring_duration_ms` - Total refactoring duration
- `refactron_refactoring_success_rate` - Refactoring success rate percentage

### Prometheus Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'refactron'
    static_configs:
      - targets: ['localhost:9090']
```

## Telemetry

### Features

- **Opt-in only**: Telemetry is disabled by default
- **Anonymous**: Uses a machine-specific hash (no personal information)
- **Privacy-focused**: No code, file names, or error details are collected
- **Local storage**: Events stored locally in `~/.refactron/telemetry/`

### What is Collected

**Collected:**
- Number of files analyzed
- Analysis execution time
- Number of issues found (not the actual issues)
- Python version and OS platform
- Refactoring operations applied
- Feature usage statistics

**NOT Collected:**
- Your code or file names
- Personal information
- Specific error messages or stack traces
- IP addresses or location data

### Configuration

```yaml
# Enable opt-in telemetry
enable_telemetry: false
```

### Managing Telemetry

```bash
# Check telemetry status
refactron telemetry --status

# Enable telemetry
refactron telemetry --enable

# Disable telemetry
refactron telemetry --disable
```

## Configuration

### Complete Configuration Example

See `examples/refactron_monitoring.yaml` for a complete configuration file with all monitoring options.

### Environment-Specific Configurations

**Development:**
```yaml
log_level: DEBUG
log_format: text
enable_metrics: true
enable_prometheus: false
enable_telemetry: false
```

**CI/CD:**
```yaml
log_level: INFO
log_format: json
enable_metrics: true
enable_prometheus: false
enable_telemetry: false
```

**Production:**
```yaml
log_level: WARNING
log_format: json
enable_metrics: true
enable_prometheus: true
prometheus_port: 9090
enable_telemetry: true
```

## CLI Commands

### Analysis with Monitoring

```bash
# Use JSON logging for CI/CD
refactron analyze mycode.py --log-format json

# Set log level
refactron analyze mycode.py --log-level DEBUG

# Show metrics after analysis
refactron analyze mycode.py --show-metrics

# Combine options
refactron analyze mycode.py --log-level INFO --log-format json --show-metrics
```

### Viewing Metrics

```bash
# View metrics in text format
refactron metrics

# View metrics in JSON format
refactron metrics --format json
```

### Prometheus Server

```bash
# Start metrics server
refactron serve-metrics

# Custom host and port
refactron serve-metrics --host 127.0.0.1 --port 8080

# The server will run until Ctrl+C is pressed
```

### Telemetry Management

```bash
# Check status
refactron telemetry --status

# Enable telemetry
refactron telemetry --enable

# Disable telemetry
refactron telemetry --disable
```

## Examples

### Example 1: CI/CD Pipeline

```yaml
# .refactron.yaml
log_level: INFO
log_format: json
enable_metrics: true
enable_file_logging: true
log_file: /var/log/refactron/analysis.log
```

```bash
# Run analysis in CI
refactron analyze src/ --log-format json > analysis.json
```

### Example 2: Production Monitoring

```yaml
# .refactron.yaml
log_level: WARNING
log_format: json
enable_metrics: true
enable_prometheus: true
prometheus_host: "0.0.0.0"
prometheus_port: 9090
```

```bash
# Start Prometheus server as a service
refactron serve-metrics &

# Configure Prometheus to scrape metrics
# Add to prometheus.yml:
scrape_configs:
  - job_name: 'refactron'
    static_configs:
      - targets: ['localhost:9090']
```

### Example 3: Development with Detailed Logging

```bash
# Run with debug logging and show metrics
refactron analyze mycode.py \
  --log-level DEBUG \
  --log-format text \
  --show-metrics
```

## Best Practices

1. **Use JSON logging in CI/CD**: JSON format integrates easily with log aggregation systems
2. **Enable metrics in production**: Track performance and identify bottlenecks
3. **Use Prometheus for monitoring**: Set up dashboards to visualize Refactron metrics
4. **Keep telemetry optional**: Respect user privacy by making telemetry opt-in
5. **Rotate logs regularly**: Configure appropriate log size limits and backup counts
6. **Monitor success rates**: Track analysis and refactoring success rates to identify issues
7. **Use appropriate log levels**: Use WARNING or ERROR in production to reduce noise

## Troubleshooting

### Logs not appearing

Check that file logging is enabled:
```yaml
enable_file_logging: true
```

### Metrics showing zero

Metrics are collected per Refactron instance. Make sure you're viewing metrics immediately after running an analysis in the same session.

### Prometheus server won't start

Check if the port is already in use:
```bash
lsof -i :9090
```

Try a different port:
```bash
refactron serve-metrics --port 9091
```

### Telemetry not working

Telemetry must be explicitly enabled:
```bash
refactron telemetry --enable
```

Check the telemetry configuration:
```bash
cat ~/.refactron/telemetry.json
```

## Support

For issues or questions about logging and monitoring features, please:
1. Check this documentation
2. Review the example configuration file
3. Open an issue on GitHub: https://github.com/Refactron-ai/Refactron_lib/issues
