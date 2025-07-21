package chatdelta

import (
	"fmt"
	"net"
	"net/url"
	"time"
)

// ErrorType represents the category of error
type ErrorType string

const (
	ErrorTypeNetwork ErrorType = "network"
	ErrorTypeAPI     ErrorType = "api"
	ErrorTypeAuth    ErrorType = "auth"
	ErrorTypeConfig  ErrorType = "config"
	ErrorTypeParse   ErrorType = "parse"
	ErrorTypeStream  ErrorType = "stream"
)

// ClientError represents an error that occurred during client operations
type ClientError struct {
	Type    ErrorType `json:"type"`
	Code    string    `json:"code,omitempty"`
	Message string    `json:"message"`
	Cause   error     `json:"-"`
}

// Error implements the error interface
func (e *ClientError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %s (caused by: %v)", e.Type, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

// Unwrap returns the underlying error
func (e *ClientError) Unwrap() error {
	return e.Cause
}

// Is implements error matching for error types
func (e *ClientError) Is(target error) bool {
	if t, ok := target.(*ClientError); ok {
		return e.Type == t.Type && e.Code == t.Code
	}
	return false
}

// Network Error constructors

// NewTimeoutError creates a new timeout error
func NewTimeoutError(timeout time.Duration) *ClientError {
	return &ClientError{
		Type:    ErrorTypeNetwork,
		Code:    "timeout",
		Message: fmt.Sprintf("request timed out after %v", timeout),
	}
}

// NewConnectionError creates a new connection error
func NewConnectionError(err error) *ClientError {
	return &ClientError{
		Type:    ErrorTypeNetwork,
		Code:    "connection_failed",
		Message: "failed to connect to the API server",
		Cause:   err,
	}
}

// NewDNSError creates a new DNS resolution error
func NewDNSError(hostname string, err error) *ClientError {
	return &ClientError{
		Type:    ErrorTypeNetwork,
		Code:    "dns_error",
		Message: fmt.Sprintf("failed to resolve hostname: %s", hostname),
		Cause:   err,
	}
}

// API Error constructors

// NewRateLimitError creates a new rate limit error
func NewRateLimitError(retryAfter *time.Duration) *ClientError {
	message := "rate limit exceeded"
	if retryAfter != nil {
		message = fmt.Sprintf("rate limit exceeded, retry after %v", *retryAfter)
	}
	return &ClientError{
		Type:    ErrorTypeAPI,
		Code:    "rate_limit",
		Message: message,
	}
}

// NewQuotaExceededError creates a new quota exceeded error
func NewQuotaExceededError() *ClientError {
	return &ClientError{
		Type:    ErrorTypeAPI,
		Code:    "quota_exceeded",
		Message: "API quota has been exceeded",
	}
}

// NewInvalidModelError creates a new invalid model error
func NewInvalidModelError(model string) *ClientError {
	return &ClientError{
		Type:    ErrorTypeAPI,
		Code:    "invalid_model",
		Message: fmt.Sprintf("invalid or unsupported model: %s", model),
	}
}

// NewServerError creates a new server error
func NewServerError(statusCode int, message string) *ClientError {
	return &ClientError{
		Type:    ErrorTypeAPI,
		Code:    "server_error",
		Message: fmt.Sprintf("server returned status %d: %s", statusCode, message),
	}
}

// NewBadRequestError creates a new bad request error
func NewBadRequestError(message string) *ClientError {
	return &ClientError{
		Type:    ErrorTypeAPI,
		Code:    "bad_request",
		Message: message,
	}
}

// Auth Error constructors

// NewInvalidAPIKeyError creates a new invalid API key error
func NewInvalidAPIKeyError() *ClientError {
	return &ClientError{
		Type:    ErrorTypeAuth,
		Code:    "invalid_api_key",
		Message: "invalid or missing API key",
	}
}

// NewExpiredTokenError creates a new expired token error
func NewExpiredTokenError() *ClientError {
	return &ClientError{
		Type:    ErrorTypeAuth,
		Code:    "expired_token",
		Message: "authentication token has expired",
	}
}

// NewPermissionDeniedError creates a new permission denied error
func NewPermissionDeniedError(resource string) *ClientError {
	return &ClientError{
		Type:    ErrorTypeAuth,
		Code:    "permission_denied",
		Message: fmt.Sprintf("insufficient permissions to access: %s", resource),
	}
}

// Config Error constructors

// NewInvalidParameterError creates a new invalid parameter error
func NewInvalidParameterError(parameter, value string) *ClientError {
	return &ClientError{
		Type:    ErrorTypeConfig,
		Code:    "invalid_parameter",
		Message: fmt.Sprintf("invalid parameter %s: %s", parameter, value),
	}
}

// NewMissingConfigError creates a new missing configuration error
func NewMissingConfigError(config string) *ClientError {
	return &ClientError{
		Type:    ErrorTypeConfig,
		Code:    "missing_config",
		Message: fmt.Sprintf("required configuration missing: %s", config),
	}
}

// Parse Error constructors

// NewJSONParseError creates a new JSON parsing error
func NewJSONParseError(err error) *ClientError {
	return &ClientError{
		Type:    ErrorTypeParse,
		Code:    "json_parse_error",
		Message: "failed to parse JSON response",
		Cause:   err,
	}
}

// NewMissingFieldError creates a new missing field error
func NewMissingFieldError(field string) *ClientError {
	return &ClientError{
		Type:    ErrorTypeParse,
		Code:    "missing_field",
		Message: fmt.Sprintf("required field missing in response: %s", field),
	}
}

// Stream Error constructors

// NewStreamClosedError creates a new stream closed error
func NewStreamClosedError() *ClientError {
	return &ClientError{
		Type:    ErrorTypeStream,
		Code:    "stream_closed",
		Message: "stream has been closed unexpectedly",
	}
}

// NewStreamReadError creates a new stream read error
func NewStreamReadError(err error) *ClientError {
	return &ClientError{
		Type:    ErrorTypeStream,
		Code:    "stream_read_error",
		Message: "failed to read from stream",
		Cause:   err,
	}
}

// Helper functions to classify errors

// IsNetworkError checks if the error is a network-related error
func IsNetworkError(err error) bool {
	if ce, ok := err.(*ClientError); ok {
		return ce.Type == ErrorTypeNetwork
	}

	// Check for standard library network errors
	if netErr, ok := err.(net.Error); ok {
		return netErr.Timeout() || netErr.Temporary()
	}

	// Check for URL errors
	if _, ok := err.(*url.Error); ok {
		return true
	}

	return false
}

// IsRetryableError checks if the error is retryable
func IsRetryableError(err error) bool {
	if ce, ok := err.(*ClientError); ok {
		switch ce.Type {
		case ErrorTypeNetwork:
			return true
		case ErrorTypeAPI:
			return ce.Code == "rate_limit" || ce.Code == "server_error"
		default:
			return false
		}
	}

	// Check for standard library network errors
	if netErr, ok := err.(net.Error); ok {
		return netErr.Timeout() || netErr.Temporary()
	}

	return false
}

// IsAuthenticationError checks if the error is authentication-related
func IsAuthenticationError(err error) bool {
	if ce, ok := err.(*ClientError); ok {
		return ce.Type == ErrorTypeAuth
	}
	return false
}
