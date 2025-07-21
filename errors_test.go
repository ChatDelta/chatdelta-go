package chatdelta

import (
	"errors"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestClientError_Error(t *testing.T) {
	err := &ClientError{
		Type:    ErrorTypeNetwork,
		Code:    "timeout",
		Message: "request timed out",
	}

	expected := "network: request timed out"
	assert.Equal(t, expected, err.Error())
}

func TestClientError_ErrorWithCause(t *testing.T) {
	cause := errors.New("underlying error")
	err := &ClientError{
		Type:    ErrorTypeNetwork,
		Code:    "connection_failed",
		Message: "failed to connect",
		Cause:   cause,
	}

	expected := "network: failed to connect (caused by: underlying error)"
	assert.Equal(t, expected, err.Error())
}

func TestClientError_Unwrap(t *testing.T) {
	cause := errors.New("underlying error")
	err := &ClientError{
		Type:  ErrorTypeNetwork,
		Code:  "connection_failed",
		Cause: cause,
	}

	assert.Equal(t, cause, err.Unwrap())
}

func TestClientError_Is(t *testing.T) {
	err1 := &ClientError{
		Type: ErrorTypeNetwork,
		Code: "timeout",
	}

	err2 := &ClientError{
		Type: ErrorTypeNetwork,
		Code: "timeout",
	}

	err3 := &ClientError{
		Type: ErrorTypeAPI,
		Code: "timeout",
	}

	assert.True(t, err1.Is(err2))
	assert.False(t, err1.Is(err3))
	assert.False(t, err1.Is(errors.New("different error")))
}

func TestNetworkErrorConstructors(t *testing.T) {
	t.Run("timeout error", func(t *testing.T) {
		err := NewTimeoutError(30 * time.Second)
		assert.Equal(t, ErrorTypeNetwork, err.Type)
		assert.Equal(t, "timeout", err.Code)
		assert.Contains(t, err.Message, "30s")
	})

	t.Run("connection error", func(t *testing.T) {
		cause := errors.New("connection refused")
		err := NewConnectionError(cause)
		assert.Equal(t, ErrorTypeNetwork, err.Type)
		assert.Equal(t, "connection_failed", err.Code)
		assert.Equal(t, cause, err.Cause)
	})

	t.Run("DNS error", func(t *testing.T) {
		cause := errors.New("no such host")
		err := NewDNSError("example.com", cause)
		assert.Equal(t, ErrorTypeNetwork, err.Type)
		assert.Equal(t, "dns_error", err.Code)
		assert.Contains(t, err.Message, "example.com")
		assert.Equal(t, cause, err.Cause)
	})
}

func TestAPIErrorConstructors(t *testing.T) {
	t.Run("rate limit error", func(t *testing.T) {
		err := NewRateLimitError(nil)
		assert.Equal(t, ErrorTypeAPI, err.Type)
		assert.Equal(t, "rate_limit", err.Code)
		assert.Contains(t, err.Message, "rate limit exceeded")
	})

	t.Run("rate limit error with retry after", func(t *testing.T) {
		retryAfter := 60 * time.Second
		err := NewRateLimitError(&retryAfter)
		assert.Equal(t, ErrorTypeAPI, err.Type)
		assert.Equal(t, "rate_limit", err.Code)
		assert.Contains(t, err.Message, "1m0s")
	})

	t.Run("quota exceeded error", func(t *testing.T) {
		err := NewQuotaExceededError()
		assert.Equal(t, ErrorTypeAPI, err.Type)
		assert.Equal(t, "quota_exceeded", err.Code)
	})

	t.Run("invalid model error", func(t *testing.T) {
		err := NewInvalidModelError("invalid-model")
		assert.Equal(t, ErrorTypeAPI, err.Type)
		assert.Equal(t, "invalid_model", err.Code)
		assert.Contains(t, err.Message, "invalid-model")
	})

	t.Run("server error", func(t *testing.T) {
		err := NewServerError(500, "Internal Server Error")
		assert.Equal(t, ErrorTypeAPI, err.Type)
		assert.Equal(t, "server_error", err.Code)
		assert.Contains(t, err.Message, "500")
		assert.Contains(t, err.Message, "Internal Server Error")
	})

	t.Run("bad request error", func(t *testing.T) {
		err := NewBadRequestError("Invalid request")
		assert.Equal(t, ErrorTypeAPI, err.Type)
		assert.Equal(t, "bad_request", err.Code)
		assert.Equal(t, "Invalid request", err.Message)
	})
}

func TestAuthErrorConstructors(t *testing.T) {
	t.Run("invalid API key error", func(t *testing.T) {
		err := NewInvalidAPIKeyError()
		assert.Equal(t, ErrorTypeAuth, err.Type)
		assert.Equal(t, "invalid_api_key", err.Code)
	})

	t.Run("expired token error", func(t *testing.T) {
		err := NewExpiredTokenError()
		assert.Equal(t, ErrorTypeAuth, err.Type)
		assert.Equal(t, "expired_token", err.Code)
	})

	t.Run("permission denied error", func(t *testing.T) {
		err := NewPermissionDeniedError("sensitive-resource")
		assert.Equal(t, ErrorTypeAuth, err.Type)
		assert.Equal(t, "permission_denied", err.Code)
		assert.Contains(t, err.Message, "sensitive-resource")
	})
}

func TestConfigErrorConstructors(t *testing.T) {
	t.Run("invalid parameter error", func(t *testing.T) {
		err := NewInvalidParameterError("temperature", "invalid-value")
		assert.Equal(t, ErrorTypeConfig, err.Type)
		assert.Equal(t, "invalid_parameter", err.Code)
		assert.Contains(t, err.Message, "temperature")
		assert.Contains(t, err.Message, "invalid-value")
	})

	t.Run("missing config error", func(t *testing.T) {
		err := NewMissingConfigError("api_key")
		assert.Equal(t, ErrorTypeConfig, err.Type)
		assert.Equal(t, "missing_config", err.Code)
		assert.Contains(t, err.Message, "api_key")
	})
}

func TestParseErrorConstructors(t *testing.T) {
	t.Run("JSON parse error", func(t *testing.T) {
		cause := errors.New("invalid JSON")
		err := NewJSONParseError(cause)
		assert.Equal(t, ErrorTypeParse, err.Type)
		assert.Equal(t, "json_parse_error", err.Code)
		assert.Equal(t, cause, err.Cause)
	})

	t.Run("missing field error", func(t *testing.T) {
		err := NewMissingFieldError("required_field")
		assert.Equal(t, ErrorTypeParse, err.Type)
		assert.Equal(t, "missing_field", err.Code)
		assert.Contains(t, err.Message, "required_field")
	})
}

func TestStreamErrorConstructors(t *testing.T) {
	t.Run("stream closed error", func(t *testing.T) {
		err := NewStreamClosedError()
		assert.Equal(t, ErrorTypeStream, err.Type)
		assert.Equal(t, "stream_closed", err.Code)
	})

	t.Run("stream read error", func(t *testing.T) {
		cause := errors.New("read error")
		err := NewStreamReadError(cause)
		assert.Equal(t, ErrorTypeStream, err.Type)
		assert.Equal(t, "stream_read_error", err.Code)
		assert.Equal(t, cause, err.Cause)
	})
}

// Mock network error types for testing
type mockNetError struct {
	timeout   bool
	temporary bool
	msg       string
}

func (e *mockNetError) Error() string   { return e.msg }
func (e *mockNetError) Timeout() bool   { return e.timeout }
func (e *mockNetError) Temporary() bool { return e.temporary }

// Ensure it implements net.Error interface
var _ net.Error = (*mockNetError)(nil)

func TestIsNetworkError(t *testing.T) {
	t.Run("client error network type", func(t *testing.T) {
		err := &ClientError{Type: ErrorTypeNetwork}
		assert.True(t, IsNetworkError(err))
	})

	t.Run("client error non-network type", func(t *testing.T) {
		err := &ClientError{Type: ErrorTypeAPI}
		assert.False(t, IsNetworkError(err))
	})

	t.Run("net error timeout", func(t *testing.T) {
		err := &mockNetError{timeout: true}
		assert.True(t, IsNetworkError(err))
	})

	t.Run("net error temporary", func(t *testing.T) {
		err := &mockNetError{temporary: true}
		assert.True(t, IsNetworkError(err))
	})

	// Note: url.Error doesn't always implement net.Error, so we skip this test

	t.Run("other error", func(t *testing.T) {
		err := errors.New("some other error")
		assert.False(t, IsNetworkError(err))
	})
}

func TestIsRetryableError(t *testing.T) {
	t.Run("network error", func(t *testing.T) {
		err := &ClientError{Type: ErrorTypeNetwork}
		assert.True(t, IsRetryableError(err))
	})

	t.Run("rate limit error", func(t *testing.T) {
		err := &ClientError{Type: ErrorTypeAPI, Code: "rate_limit"}
		assert.True(t, IsRetryableError(err))
	})

	t.Run("server error", func(t *testing.T) {
		err := &ClientError{Type: ErrorTypeAPI, Code: "server_error"}
		assert.True(t, IsRetryableError(err))
	})

	t.Run("auth error", func(t *testing.T) {
		err := &ClientError{Type: ErrorTypeAuth}
		assert.False(t, IsRetryableError(err))
	})

	t.Run("config error", func(t *testing.T) {
		err := &ClientError{Type: ErrorTypeConfig}
		assert.False(t, IsRetryableError(err))
	})

	t.Run("net error timeout", func(t *testing.T) {
		err := &mockNetError{timeout: true}
		assert.True(t, IsRetryableError(err))
	})

	t.Run("other error", func(t *testing.T) {
		err := errors.New("some other error")
		assert.False(t, IsRetryableError(err))
	})
}

func TestIsAuthenticationError(t *testing.T) {
	t.Run("auth error", func(t *testing.T) {
		err := &ClientError{Type: ErrorTypeAuth}
		assert.True(t, IsAuthenticationError(err))
	})

	t.Run("non-auth error", func(t *testing.T) {
		err := &ClientError{Type: ErrorTypeNetwork}
		assert.False(t, IsAuthenticationError(err))
	})

	t.Run("other error", func(t *testing.T) {
		err := errors.New("some other error")
		assert.False(t, IsAuthenticationError(err))
	})
}
