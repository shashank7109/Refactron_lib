"""Coverage test for cli.__main__ entrypoint."""

from __future__ import annotations

import runpy


def test_cli_main_module_calls_main(monkeypatch) -> None:
    called = {"n": 0}

    def fake_main() -> None:
        called["n"] += 1

    monkeypatch.setattr("refactron.cli.main.main", fake_main)
    runpy.run_module("refactron.cli.__main__", run_name="__main__")
    assert called["n"] == 1
