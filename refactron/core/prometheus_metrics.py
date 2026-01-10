"""Prometheus metrics exporter for Refactron.

This module provides Prometheus-compatible metrics endpoint for monitoring
Refactron's performance and usage in production environments.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

from refactron.core.metrics import get_metrics_collector


class PrometheusMetrics:
    """Prometheus metrics formatter and exporter."""

    def __init__(self) -> None:
        """Initialize Prometheus metrics."""
        self.metrics_collector = get_metrics_collector()

    def format_metrics(self) -> str:
        """Format metrics in Prometheus exposition format.

        Returns:
            String containing Prometheus-formatted metrics
        """
        lines = []
        summary = self.metrics_collector.get_combined_summary()

        # Analysis metrics
        analysis = summary.get("analysis", {})

        # HELP and TYPE declarations
        lines.append("# HELP refactron_files_analyzed_total Total number of files analyzed")
        lines.append("# TYPE refactron_files_analyzed_total counter")
        lines.append(f"refactron_files_analyzed_total {analysis.get('total_files_analyzed', 0)}")
        lines.append("")

        lines.append(
            "# HELP refactron_files_failed_total Total number of files that failed analysis"
        )
        lines.append("# TYPE refactron_files_failed_total counter")
        lines.append(f"refactron_files_failed_total {analysis.get('total_files_failed', 0)}")
        lines.append("")

        lines.append("# HELP refactron_issues_found_total Total number of issues found")
        lines.append("# TYPE refactron_issues_found_total counter")
        lines.append(f"refactron_issues_found_total {analysis.get('total_issues_found', 0)}")
        lines.append("")

        lines.append(
            "# HELP refactron_analysis_duration_ms Total analysis duration in milliseconds"
        )
        lines.append("# TYPE refactron_analysis_duration_ms gauge")
        lines.append(f"refactron_analysis_duration_ms {analysis.get('total_analysis_time_ms', 0)}")
        lines.append("")

        lines.append(
            "# HELP refactron_avg_analysis_time_per_file_ms "
            "Average analysis time per file in milliseconds"
        )
        lines.append("# TYPE refactron_avg_analysis_time_per_file_ms gauge")
        lines.append(
            f"refactron_avg_analysis_time_per_file_ms "
            f"{analysis.get('average_time_per_file_ms', 0)}"
        )
        lines.append("")

        lines.append("# HELP refactron_analysis_success_rate Analysis success rate as percentage")
        lines.append("# TYPE refactron_analysis_success_rate gauge")
        lines.append(f"refactron_analysis_success_rate {analysis.get('success_rate_percent', 0)}")
        lines.append("")

        # Analyzer hit counts
        lines.append(
            "# HELP refactron_analyzer_hits_total Number of times each analyzer found issues"
        )
        lines.append("# TYPE refactron_analyzer_hits_total counter")
        analyzer_hits = analysis.get("analyzer_hit_counts", {})
        for analyzer, count in analyzer_hits.items():
            lines.append(f'refactron_analyzer_hits_total{{analyzer="{analyzer}"}} {count}')
        if not analyzer_hits:
            lines.append('refactron_analyzer_hits_total{analyzer=""} 0')
        lines.append("")

        # Issue type counts
        lines.append("# HELP refactron_issue_type_total Number of issues by type")
        lines.append("# TYPE refactron_issue_type_total counter")
        issue_types = analysis.get("issue_type_counts", {})
        for issue_type, count in issue_types.items():
            lines.append(f'refactron_issue_type_total{{issue_type="{issue_type}"}} {count}')
        if not issue_types:
            lines.append('refactron_issue_type_total{issue_type=""} 0')
        lines.append("")

        # Refactoring metrics
        refactoring = summary.get("refactoring", {})

        lines.append(
            "# HELP refactron_refactorings_applied_total Total number of refactorings applied"
        )
        lines.append("# TYPE refactron_refactorings_applied_total counter")
        lines.append(
            f"refactron_refactorings_applied_total "
            f"{refactoring.get('total_refactorings_applied', 0)}"
        )
        lines.append("")

        lines.append(
            "# HELP refactron_refactorings_failed_total Total number of refactorings that failed"
        )
        lines.append("# TYPE refactron_refactorings_failed_total counter")
        lines.append(
            f"refactron_refactorings_failed_total {refactoring.get('total_refactorings_failed', 0)}"
        )
        lines.append("")

        lines.append(
            "# HELP refactron_refactoring_duration_ms Total refactoring duration in milliseconds"
        )
        lines.append("# TYPE refactron_refactoring_duration_ms gauge")
        lines.append(
            f"refactron_refactoring_duration_ms {refactoring.get('total_refactoring_time_ms', 0)}"
        )
        lines.append("")

        lines.append(
            "# HELP refactron_avg_refactoring_time_per_operation_ms "
            "Average refactoring time per operation in milliseconds"
        )
        lines.append("# TYPE refactron_avg_refactoring_time_per_operation_ms gauge")
        lines.append(
            f"refactron_avg_refactoring_time_per_operation_ms "
            f"{refactoring.get('average_time_per_operation_ms', 0)}"
        )
        lines.append("")

        lines.append(
            "# HELP refactron_refactoring_success_rate Refactoring success rate as percentage"
        )
        lines.append("# TYPE refactron_refactoring_success_rate gauge")
        lines.append(
            f"refactron_refactoring_success_rate {refactoring.get('success_rate_percent', 0)}"
        )
        lines.append("")

        # Refactorer hit counts
        lines.append(
            "# HELP refactron_refactorer_hits_total Number of times each refactorer was applied"
        )
        lines.append("# TYPE refactron_refactorer_hits_total counter")
        refactorer_hits = refactoring.get("refactorer_hit_counts", {})
        for refactorer, count in refactorer_hits.items():
            lines.append(f'refactron_refactorer_hits_total{{refactorer="{refactorer}"}} {count}')
        if not refactorer_hits:
            lines.append('refactron_refactorer_hits_total{refactorer=""} 0')
        lines.append("")

        # Risk level distribution
        lines.append(
            "# HELP refactron_refactoring_risk_level_total Number of refactorings by risk level"
        )
        lines.append("# TYPE refactron_refactoring_risk_level_total counter")
        risk_levels = refactoring.get("risk_level_distribution", {})
        for risk_level, count in risk_levels.items():
            lines.append(
                f'refactron_refactoring_risk_level_total{{risk_level="{risk_level}"}} {count}'
            )
        if not risk_levels:
            lines.append('refactron_refactoring_risk_level_total{risk_level=""} 0')
        lines.append("")

        return "\n".join(lines)


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    def do_GET(self) -> None:
        """Handle GET requests to /metrics endpoint."""
        if self.path == "/metrics":
            metrics = PrometheusMetrics()
            content = metrics.format_metrics()

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        elif self.path == "/health":
            # Health check endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass


class PrometheusMetricsServer:
    """HTTP server for exposing Prometheus metrics."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9090) -> None:
        """Initialize Prometheus metrics server.

        Args:
            host: Host to bind to (default: 127.0.0.1 for localhost-only access)
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        if self.server is not None:
            return  # Already running

        self.server = HTTPServer((self.host, self.port), MetricsHTTPHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def stop(self) -> None:
        """Stop the metrics server."""
        if self.server is not None:
            self.server.shutdown()
            self.server = None
            self.server_thread = None

    def is_running(self) -> bool:
        """Check if the metrics server is running.

        Returns:
            True if server is running, False otherwise
        """
        return self.server is not None


# Global metrics server instance
_global_metrics_server: Optional[PrometheusMetricsServer] = None
_server_lock = threading.Lock()


def start_metrics_server(host: str = "127.0.0.1", port: int = 9090) -> PrometheusMetricsServer:
    """Start the global Prometheus metrics server.

    Args:
        host: Host to bind to (default: 127.0.0.1 for localhost-only access)
        port: Port to listen on

    Returns:
        PrometheusMetricsServer instance
    """
    global _global_metrics_server
    with _server_lock:
        if _global_metrics_server is None:
            _global_metrics_server = PrometheusMetricsServer(host, port)
        _global_metrics_server.start()
        return _global_metrics_server


def stop_metrics_server() -> None:
    """Stop the global Prometheus metrics server."""
    global _global_metrics_server
    with _server_lock:
        if _global_metrics_server is not None:
            _global_metrics_server.stop()
            _global_metrics_server = None


def get_metrics_server() -> Optional[PrometheusMetricsServer]:
    """Get the global metrics server instance.

    Returns:
        PrometheusMetricsServer instance or None if not started
    """
    return _global_metrics_server
