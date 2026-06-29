# SSL and Streaming Configuration

## Overview

llm-bench supports secure connections by default but can be configured to work with endpoints that have self-signed certificates or do not support streaming.

## SSL Certificate Verification

### Problem
Some endpoints (especially development or internal proxies) use self-signed SSL certificates that cause `CERTIFICATE_VERIFY_FAILED` errors.

### Solution
Use `ssl_verify: false` in the model configuration or the CLI flag `--no-ssl-verify`.

### Configuration
```yaml
models:
  - name: internal-endpoint
    base_url: https://internal-proxy.example.com/v1
    model: my-model
    api_key: $ENV:API_KEY
    ssl_verify: false  # Disable SSL verification for self-signed certs
```

### CLI Override
```bash
# Disable SSL verification for all models in this run
llm-bench run -m my-model --no-ssl-verify
```

### Implementation Details
- `ssl_verify` is a field in `ModelRegistryEntry` (default: `true`)
- CLI flag `--no-ssl-verify` overrides the config value when explicitly passed
- SSL verification is applied in `_build_httpx_client()` which passes `verify=ssl_verify` to httpx
- Both regular requests and evaluation requests respect this setting

## Streaming Support

### Problem
Some endpoints do not support Server-Sent Events (SSE) streaming and will return errors or incomplete responses when streaming is requested.

### Solution
Use `stream: false` in the model configuration to disable streaming for that specific endpoint.

### Configuration
```yaml
models:
  - name: non-streaming-endpoint
    base_url: https://api.example.com/v1
    model: my-model
    api_key: $ENV:API_KEY
    stream: false  # Disable streaming for this endpoint
```

### Implementation Details
- `stream` is a field in `ModelRegistryEntry` (default: `true`)
- When `stream: false`, the following changes occur:
  - `stream` parameter in the request payload is set to `false`
  - `stream_options` parameter is omitted from the payload
  - `_perform_request()` uses a regular POST request instead of streaming
  - `_classify_non_stream_response()` processes the single response as if it were a complete stream
- This maintains compatibility with the existing streaming-based metrics and recording infrastructure

## Combined Example

For endpoints with both self-signed certificates and no streaming support:

```yaml
models:
  - name: ei-mistral-medium35
    base_url: https://iagen-proxy-api-r-wdep.d-trs.apps.sprd-0ds01c-000.cloud.cm-cic.fr/open-ia/v1/sandbox/watsonXExperimental/testIGP/11-I72HC10/c429635d-0fbe-404d-b8a4-b8162c5c5cc4
    model: mistralai/mistral-medium-3-5-128b
    api_key: a3610fb4c91d41b29a2d283e98554689
    ssl_verify: false
    stream: false
```

## Default Configuration Updates

The `STARTER_CONFIG` in `config.py` has been updated to include proper configurations for common EI models:
- `ei-mistral-medium35`: Uses correct model ID with SSL verification and streaming disabled
- `ei-qwen35`: Qwen3.5-27B with SSL verification and streaming disabled
- `ei-qwen36`: Qwen3.6-27B with SSL verification and streaming disabled

## Known Limitations

1. When streaming is disabled, certain metrics related to streaming performance (like TTFT and TPOT calculated from stream chunks) may show as zero or be approximated
2. Streaming can only be disabled per-model, not per-request
3. Both `ssl_verify` and `stream` must be explicitly set to `false` for endpoints that need both

## Testing

To verify SSL and streaming functionality:

```bash
# Test with preflight (checks connectivity without running full benchmark)
llm-bench run -m ei-mistral-medium35 --preflight

# Test with specific SSL and streaming settings
llm-bench run -m my-model --no-ssl-verify --duration 5s
```

## Security Considerations

Disabling SSL verification (`ssl_verify: false`) should only be used for:
- Development environments
- Internal endpoints with self-signed certificates
- Trusted networks

Never disable SSL verification for production endpoints unless you fully understand the security implications.
