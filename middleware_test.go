package chatdelta

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValidateJSONResponse_Valid(t *testing.T) {
	data := []byte(`{"choices": [], "model": "gpt-4"}`)
	err := validateJSONResponse(data, "choices", "model")
	assert.NoError(t, err)
}

func TestValidateJSONResponse_MissingField(t *testing.T) {
	data := []byte(`{"model": "gpt-4"}`)
	err := validateJSONResponse(data, "choices")
	assert.Error(t, err)
	ce, ok := err.(*ClientError)
	assert.True(t, ok)
	assert.Equal(t, "missing_field", ce.Code)
}

func TestValidateJSONResponse_InvalidJSON(t *testing.T) {
	data := []byte(`not json`)
	err := validateJSONResponse(data, "field")
	assert.Error(t, err)
	ce, ok := err.(*ClientError)
	assert.True(t, ok)
	assert.Equal(t, ErrorTypeParse, ce.Type)
}

func TestExtractAPIError_OpenAIShape(t *testing.T) {
	data := []byte(`{"error":{"message":"invalid api key","code":"invalid_api_key"}}`)
	msg := extractAPIError(data)
	assert.Equal(t, "invalid_api_key: invalid api key", msg)
}

func TestExtractAPIError_MessageShape(t *testing.T) {
	data := []byte(`{"message":"resource not found"}`)
	msg := extractAPIError(data)
	assert.Equal(t, "resource not found", msg)
}

func TestExtractAPIError_DetailShape(t *testing.T) {
	data := []byte(`{"detail":"validation error"}`)
	msg := extractAPIError(data)
	assert.Equal(t, "validation error", msg)
}

func TestExtractAPIError_FallbackRawBody(t *testing.T) {
	data := []byte(`unexpected plain text error`)
	msg := extractAPIError(data)
	assert.Equal(t, "unexpected plain text error", msg)
}

func TestExtractAPIError_ErrorWithoutCode(t *testing.T) {
	data := []byte(`{"error":{"message":"rate limit exceeded"}}`)
	msg := extractAPIError(data)
	assert.Equal(t, "rate limit exceeded", msg)
}
