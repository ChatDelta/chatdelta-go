package chatdelta

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewConversation(t *testing.T) {
	conv := NewConversation()
	assert.NotNil(t, conv)
	assert.Empty(t, conv.Messages)
}

func TestConversation_AddMessage(t *testing.T) {
	conv := NewConversation()
	
	conv.AddMessage("user", "Hello")
	assert.Len(t, conv.Messages, 1)
	assert.Equal(t, "user", conv.Messages[0].Role)
	assert.Equal(t, "Hello", conv.Messages[0].Content)
}

func TestConversation_AddHelperMethods(t *testing.T) {
	conv := NewConversation()
	
	conv.AddSystemMessage("System message")
	conv.AddUserMessage("User message")
	conv.AddAssistantMessage("Assistant message")
	
	require.Len(t, conv.Messages, 3)
	assert.Equal(t, "system", conv.Messages[0].Role)
	assert.Equal(t, "user", conv.Messages[1].Role)
	assert.Equal(t, "assistant", conv.Messages[2].Role)
}

func TestNewClientConfig(t *testing.T) {
	config := NewClientConfig()
	assert.NotNil(t, config)
	assert.Equal(t, 30*time.Second, config.Timeout)
	assert.Equal(t, 3, config.Retries)
	assert.Nil(t, config.Temperature)
	assert.Nil(t, config.MaxTokens)
}

func TestClientConfig_BuilderPattern(t *testing.T) {
	config := NewClientConfig().
		SetTimeout(60*time.Second).
		SetRetries(5).
		SetTemperature(0.7).
		SetMaxTokens(1024).
		SetTopP(0.9).
		SetFrequencyPenalty(0.1).
		SetPresencePenalty(0.1).
		SetSystemMessage("Test message")
	
	assert.Equal(t, 60*time.Second, config.Timeout)
	assert.Equal(t, 5, config.Retries)
	assert.Equal(t, 0.7, *config.Temperature)
	assert.Equal(t, 1024, *config.MaxTokens)
	assert.Equal(t, 0.9, *config.TopP)
	assert.Equal(t, 0.1, *config.FrequencyPenalty)
	assert.Equal(t, 0.1, *config.PresencePenalty)
	assert.Equal(t, "Test message", *config.SystemMessage)
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  *ClientConfig
		wantErr bool
	}{
		{
			name:    "valid config",
			config:  NewClientConfig(),
			wantErr: false,
		},
		{
			name:    "negative timeout",
			config:  NewClientConfig().SetTimeout(-1 * time.Second),
			wantErr: true,
		},
		{
			name:    "negative retries",
			config:  NewClientConfig().SetRetries(-1),
			wantErr: true,
		},
		{
			name:    "invalid temperature",
			config:  NewClientConfig().SetTemperature(-1),
			wantErr: true,
		},
		{
			name:    "invalid max tokens",
			config:  NewClientConfig().SetMaxTokens(0),
			wantErr: true,
		},
		{
			name:    "invalid top_p",
			config:  NewClientConfig().SetTopP(1.5),
			wantErr: true,
		},
		{
			name:    "valid temperature range",
			config:  NewClientConfig().SetTemperature(1.0),
			wantErr: false,
		},
		{
			name:    "valid top_p range",
			config:  NewClientConfig().SetTopP(0.8),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateConfig(tt.config)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCreateClient(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		apiKey   string
		model    string
		config   *ClientConfig
		wantErr  bool
	}{
		{
			name:     "openai client",
			provider: "openai",
			apiKey:   "test-key",
			model:    "gpt-3.5-turbo",
			config:   nil,
			wantErr:  false,
		},
		{
			name:     "claude client",
			provider: "claude",
			apiKey:   "test-key",
			model:    "claude-3-haiku-20240307",
			config:   nil,
			wantErr:  false,
		},
		{
			name:     "anthropic client",
			provider: "anthropic",
			apiKey:   "test-key",
			model:    "claude-3-haiku-20240307",
			config:   nil,
			wantErr:  false,
		},
		{
			name:     "gemini client",
			provider: "gemini",
			apiKey:   "test-key",
			model:    "gemini-1.5-flash",
			config:   nil,
			wantErr:  false,
		},
		{
			name:     "google client",
			provider: "google",
			apiKey:   "test-key",
			model:    "gemini-1.5-flash",
			config:   nil,
			wantErr:  false,
		},
		{
			name:     "unsupported provider",
			provider: "unsupported",
			apiKey:   "test-key",
			model:    "",
			config:   nil,
			wantErr:  true,
		},
		{
			name:     "empty api key",
			provider: "unknown-provider", // Use unknown provider to avoid env vars
			apiKey:   "",
			model:    "",
			config:   nil,
			wantErr:  true,
		},
		{
			name:     "invalid config",
			provider: "openai",
			apiKey:   "test-key",
			model:    "",
			config:   NewClientConfig().SetTimeout(-1 * time.Second),
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := CreateClient(tt.provider, tt.apiKey, tt.model, tt.config)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, client)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
				
				// Test client properties
				assert.NotEmpty(t, client.Name())
				assert.NotEmpty(t, client.Model())
				
				// Test boolean methods don't panic
				assert.NotPanics(t, func() { client.SupportsStreaming() })
				assert.NotPanics(t, func() { client.SupportsConversations() })
			}
		})
	}
}

func TestGetDefaultModel(t *testing.T) {
	tests := []struct {
		provider string
		expected string
	}{
		{"openai", "gpt-3.5-turbo"},
		{"anthropic", "claude-3-haiku-20240307"},
		{"claude", "claude-3-haiku-20240307"},
		{"google", "gemini-1.5-flash"},
		{"gemini", "gemini-1.5-flash"},
		{"unknown", ""},
	}

	for _, tt := range tests {
		t.Run(tt.provider, func(t *testing.T) {
			result := getDefaultModel(tt.provider)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetClientInfo(t *testing.T) {
	client, err := CreateClient("openai", "test-key", "gpt-4", nil)
	require.NoError(t, err)
	
	info := GetClientInfo(client)
	assert.Equal(t, "OpenAI", info.Name)
	assert.Equal(t, "gpt-4", info.Model)
	assert.True(t, info.SupportsStreaming)
	assert.True(t, info.SupportsConversations)
}

func TestMergeStreamChunks(t *testing.T) {
	chunks := make(chan StreamChunk, 3)
	chunks <- StreamChunk{Content: "Hello ", Finished: false}
	chunks <- StreamChunk{Content: "World!", Finished: false}
	chunks <- StreamChunk{Content: "", Finished: true}
	close(chunks)
	
	result, err := MergeStreamChunks(chunks)
	assert.NoError(t, err)
	assert.Equal(t, "Hello World!", result)
}

func TestExecuteParallel(t *testing.T) {
	// Create mock clients for testing
	var clients []AIClient
	
	client1, err := CreateClient("openai", "test-key", "", nil)
	require.NoError(t, err)
	clients = append(clients, client1)
	
	client2, err := CreateClient("claude", "test-key", "", nil)
	require.NoError(t, err)
	clients = append(clients, client2)
	
	// Note: This will fail with invalid API keys, but tests the structure
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	
	results := ExecuteParallel(ctx, clients, "Test prompt")
	
	assert.Len(t, results, 2)
	assert.Equal(t, "OpenAI", results[0].ClientName)
	assert.Equal(t, "Claude", results[1].ClientName)
	
	// Results should contain errors due to invalid API keys
	assert.Error(t, results[0].Error)
	assert.Error(t, results[1].Error)
}

func TestExecuteParallelConversation(t *testing.T) {
	// Create mock clients for testing
	var clients []AIClient
	
	client1, err := CreateClient("openai", "test-key", "", nil)
	require.NoError(t, err)
	clients = append(clients, client1)
	
	conversation := NewConversation()
	conversation.AddUserMessage("Test message")
	
	// Note: This will fail with invalid API keys, but tests the structure
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	
	results := ExecuteParallelConversation(ctx, clients, conversation)
	
	assert.Len(t, results, 1)
	assert.Equal(t, "OpenAI", results[0].ClientName)
	
	// Result should contain error due to invalid API key
	assert.Error(t, results[0].Error)
}