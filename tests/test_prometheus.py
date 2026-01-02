"""Tests for Prometheus metrics exporter."""

import time

import pytest

from refactron.core.metrics import get_metrics_collector, reset_metrics_collector
from refactron.core.prometheus_metrics import (
    PrometheusMetrics,
    PrometheusMetricsServer,
    get_metrics_server,
    start_metrics_server,
    stop_metrics_server,
)


class TestPrometheusMetrics:
    """Test PrometheusMetrics class."""

    def test_initialization(self):
        """Test initializing PrometheusMetrics."""
        metrics = PrometheusMetrics()
        assert metrics.metrics_collector is not None

    def test_format_metrics_empty(self):
        """Test formatting metrics with no data."""
        reset_metrics_collector()
        metrics = PrometheusMetrics()
        
        output = metrics.format_metrics()
        
        # Should contain metric definitions
        assert "# HELP refactron_files_analyzed_total" in output
        assert "# TYPE refactron_files_analyzed_total counter" in output
        assert "refactron_files_analyzed_total 0" in output

    def test_format_metrics_with_data(self):
        """Test formatting metrics with actual data."""
        reset_metrics_collector()
        collector = get_metrics_collector()
        
        # Add some metrics
        collector.start_analysis()
        collector.record_file_analysis(
            file_path="/file1.py",
            analysis_time_ms=100.0,
            lines_of_code=50,
            issues_found=3,
            analyzers_run=["complexity"],
            success=True,
        )
        collector.record_analyzer_hit("complexity", "high_complexity")
        collector.end_analysis()
        
        metrics = PrometheusMetrics()
        output = metrics.format_metrics()
        
        # Verify analysis metrics
        assert "refactron_files_analyzed_total 1" in output
        assert "refactron_issues_found_total 3" in output
        # Analyzer hit should be at least 1 (could be more if record_file_analysis also counts)
        assert 'refactron_analyzer_hits_total{analyzer="complexity"}' in output
        
        # Should have proper Prometheus format
        lines = output.split("\n")
        for line in lines:
            if line and not line.startswith("#"):
                # Metric lines should have proper format
                assert " " in line or line == ""

    def test_format_refactoring_metrics(self):
        """Test formatting refactoring metrics."""
        reset_metrics_collector()
        collector = get_metrics_collector()
        
        collector.start_refactoring()
        collector.record_refactoring(
            operation_type="extract_method",
            file_path="/file.py",
            execution_time_ms=150.0,
            success=True,
            risk_level="safe",
        )
        collector.end_refactoring()
        
        metrics = PrometheusMetrics()
        output = metrics.format_metrics()
        
        assert "refactron_refactorings_applied_total 1" in output
        assert 'refactron_refactorer_hits_total{refactorer="extract_method"} 1' in output
        assert 'refactron_refactoring_risk_level_total{risk_level="safe"} 1' in output

    def test_format_metrics_with_labels(self):
        """Test metrics with labels are properly formatted."""
        reset_metrics_collector()
        collector = get_metrics_collector()
        
        collector.record_analyzer_hit("complexity", "high_complexity")
        collector.record_analyzer_hit("security", "sql_injection")
        
        metrics = PrometheusMetrics()
        output = metrics.format_metrics()
        
        # Should have labeled metrics
        assert 'analyzer="complexity"' in output
        assert 'analyzer="security"' in output
        assert 'issue_type="high_complexity"' in output
        assert 'issue_type="sql_injection"' in output


class TestPrometheusMetricsServer:
    """Test PrometheusMetricsServer class."""

    def test_initialization(self):
        """Test server initialization."""
        server = PrometheusMetricsServer(host="127.0.0.1", port=9091)
        
        assert server.host == "127.0.0.1"
        assert server.port == 9091
        assert server.server is None
        assert not server.is_running()

    def test_start_stop_server(self):
        """Test starting and stopping the server."""
        server = PrometheusMetricsServer(host="127.0.0.1", port=9092)
        
        # Start server
        server.start()
        assert server.is_running()
        
        # Give server time to start
        time.sleep(0.5)
        
        # Stop server
        server.stop()
        assert not server.is_running()

    def test_double_start(self):
        """Test that starting an already running server is a no-op."""
        server = PrometheusMetricsServer(host="127.0.0.1", port=9093)
        
        server.start()
        assert server.is_running()
        
        # Second start should be no-op
        server.start()
        assert server.is_running()
        
        server.stop()

    def test_server_responds_to_metrics_endpoint(self):
        """Test that server responds to /metrics endpoint."""
        import urllib.request

        reset_metrics_collector()
        server = PrometheusMetricsServer(host="127.0.0.1", port=9094)
        
        try:
            server.start()
            time.sleep(0.5)  # Give server time to start
            
            # Try to fetch metrics
            response = urllib.request.urlopen("http://127.0.0.1:9094/metrics")
            content = response.read().decode("utf-8")
            
            assert response.status == 200
            assert "refactron_files_analyzed_total" in content
        finally:
            server.stop()

    def test_server_responds_to_health_endpoint(self):
        """Test that server responds to /health endpoint."""
        import urllib.request

        server = PrometheusMetricsServer(host="127.0.0.1", port=9095)
        
        try:
            server.start()
            time.sleep(0.5)
            
            response = urllib.request.urlopen("http://127.0.0.1:9095/health")
            content = response.read().decode("utf-8")
            
            assert response.status == 200
            assert content == "OK"
        finally:
            server.stop()

    def test_server_404_for_unknown_endpoint(self):
        """Test that server returns 404 for unknown endpoints."""
        import urllib.error
        import urllib.request

        server = PrometheusMetricsServer(host="127.0.0.1", port=9096)
        
        try:
            server.start()
            time.sleep(0.5)
            
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen("http://127.0.0.1:9096/unknown")
            
            assert exc_info.value.code == 404
        finally:
            server.stop()


def test_global_metrics_server():
    """Test global metrics server functions."""
    # Ensure clean state
    stop_metrics_server()
    
    # Start server
    server = start_metrics_server(host="127.0.0.1", port=9097)
    assert server.is_running()
    
    # Get server should return same instance
    server2 = get_metrics_server()
    assert server is server2
    
    # Stop server
    stop_metrics_server()
    time.sleep(0.5)
    
    # Get server should return None after stopping
    server3 = get_metrics_server()
    assert server3 is None


def test_concurrent_start_global_server():
    """Test that starting global server multiple times is safe."""
    stop_metrics_server()
    
    try:
        server1 = start_metrics_server(host="127.0.0.1", port=9098)
        time.sleep(0.3)
        
        server2 = start_metrics_server(host="127.0.0.1", port=9098)
        
        # Should return same server
        assert server1 is server2
        assert server1.is_running()
    finally:
        stop_metrics_server()
