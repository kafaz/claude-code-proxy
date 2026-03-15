import importlib


def test_transport_tuning_values_are_sanitized(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MAX_CONNECTIONS", "10")
    monkeypatch.setenv("OPENAI_MAX_KEEPALIVE_CONNECTIONS", "999")
    monkeypatch.setenv("OPENAI_KEEPALIVE_EXPIRY", "-5")
    monkeypatch.setenv("OPENAI_ENABLE_HTTP2", "true")
    monkeypatch.setenv("STREAM_DISCONNECT_CHECK_INTERVAL", "0")

    config_module = importlib.import_module("src.core.config")
    config_module = importlib.reload(config_module)
    config = config_module.Config()

    assert config.openai_max_connections == 10
    assert config.openai_max_keepalive_connections == 10
    assert config.openai_keepalive_expiry == 1.0
    assert config.openai_enable_http2 is True
    assert config.stream_disconnect_check_interval == 1
