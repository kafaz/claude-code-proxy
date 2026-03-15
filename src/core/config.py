import os
import sys


class Config:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print(
                "Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled."
            )

        self.openai_base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))

        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # High-throughput transport tuning (proxy -> model provider)
        self.openai_max_connections = self._get_int_env(
            "OPENAI_MAX_CONNECTIONS", default=200, minimum=1
        )
        self.openai_max_keepalive_connections = self._get_int_env(
            "OPENAI_MAX_KEEPALIVE_CONNECTIONS", default=100, minimum=1
        )
        if self.openai_max_keepalive_connections > self.openai_max_connections:
            self.openai_max_keepalive_connections = self.openai_max_connections
        self.openai_keepalive_expiry = self._get_float_env(
            "OPENAI_KEEPALIVE_EXPIRY", default=60.0, minimum=1.0
        )
        self.openai_enable_http2 = (
            os.environ.get("OPENAI_ENABLE_HTTP2", "true").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        # Stream processing tuning
        self.stream_disconnect_check_interval = self._get_int_env(
            "STREAM_DISCONNECT_CHECK_INTERVAL", default=20, minimum=1
        )

        # Model settings - BIG and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

    def _get_int_env(self, key: str, default: int, minimum: int = 1) -> int:
        raw = os.environ.get(key, str(default))
        try:
            value = int(raw)
        except ValueError:
            value = default
        return max(minimum, value)

    def _get_float_env(self, key: str, default: float, minimum: float = 0.0) -> float:
        raw = os.environ.get(key, str(default))
        try:
            value = float(raw)
        except ValueError:
            value = default
        return max(minimum, value)

    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Basic format check for OpenAI API keys
        if not self.openai_api_key.startswith("sk-"):
            return False
        return True

    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no ANTHROPIC_API_KEY is set in environment, skip validation
        if not self.anthropic_api_key:
            return True

        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key

    def get_custom_headers(self):
        """Get custom headers from environment variables"""
        custom_headers = {}

        # Get all environment variables
        env_vars = dict(os.environ)

        # Find CUSTOM_HEADER_* environment variables
        for env_key, env_value in env_vars.items():
            if env_key.startswith("CUSTOM_HEADER_"):
                # Convert CUSTOM_HEADER_KEY to Header-Key
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix

                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace("_", "-")
                    custom_headers[header_name] = env_value

        return custom_headers


try:
    config = Config()
    print(
        f"\x05 Configuration loaded: API_KEY={'*' * 20}..., BASE_URL='{config.openai_base_url}'"
    )
except Exception as e:
    print(f"=4 Configuration Error: {e}")
    sys.exit(1)
