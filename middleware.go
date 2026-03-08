// Package chatdelta provides a unified interface for interacting with multiple AI APIs.
// middleware.go provides HTTP request middleware utilities shared across AI provider
// client implementations, including JSON response validation and API-error extraction.
// Retry logic lives in utils.go (ExecuteWithRetry / ExecuteWithExponentialBackoff).
package chatdelta

import (
	"encoding/json"
	"fmt"
)

// validateJSONResponse checks that every required field name is present as a
// top-level key in the JSON object encoded in data before full deserialization
// is attempted. It returns NewMissingFieldError for the first absent field, or
// NewJSONParseError if data is not valid JSON.
func validateJSONResponse(data []byte, required ...string) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return NewJSONParseError(err)
	}
	for _, field := range required {
		if _, ok := raw[field]; !ok {
			return NewMissingFieldError(field)
		}
	}
	return nil
}

// extractAPIError attempts to extract a human-readable error message from a
// JSON response body. It recognises several common shapes used by AI provider APIs:
//
//   - {"error": {"message": "…", "code": "…"}}  (OpenAI, Claude)
//   - {"message": "…"}
//   - {"detail": "…"}                            (some REST conventions)
//
// When no recognised shape is matched the raw body bytes are returned as-is.
func extractAPIError(data []byte) string {
	// Shape: {"error": {"message": "…"[, "code": "…"]}}
	var wrapper struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &wrapper); err == nil && wrapper.Error.Message != "" {
		if wrapper.Error.Code != "" {
			return fmt.Sprintf("%s: %s", wrapper.Error.Code, wrapper.Error.Message)
		}
		return wrapper.Error.Message
	}

	// Shape: {"message": "…"}
	var simple struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(data, &simple); err == nil && simple.Message != "" {
		return simple.Message
	}

	// Shape: {"detail": "…"}
	var detail struct {
		Detail string `json:"detail"`
	}
	if err := json.Unmarshal(data, &detail); err == nil && detail.Detail != "" {
		return detail.Detail
	}

	return string(data)
}
