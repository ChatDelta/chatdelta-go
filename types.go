package chatdelta

import (
	"context"
	"time"
)

// Message represents a single message in a conversation.
// Role should be one of "system", "user", or "assistant".
type Message struct {
	// Role of the message sender ("system", "user", or "assistant")
	Role    string `json:"role"`
	// Content of the message
	Content string `json:"content"`
}

// Conversation represents a collection of messages forming a dialogue.
// Messages are ordered chronologically with the oldest first.
type Conversation struct {
	// Messages in chronological order
	Messages []Message `json:"messages"`
}

// NewConversation creates a new conversation
func NewConversation() *Conversation {
	return &Conversation{
		Messages: make([]Message, 0),
	}
}

// AddMessage adds a message to the conversation
func (c *Conversation) AddMessage(role, content string) {
	c.Messages = append(c.Messages, Message{
		Role:    role,
		Content: content,
	})
}

// AddSystemMessage adds a system message to the conversation
func (c *Conversation) AddSystemMessage(content string) {
	c.AddMessage("system", content)
}

// AddUserMessage adds a user message to the conversation
func (c *Conversation) AddUserMessage(content string) {
	c.AddMessage("user", content)
}

// AddAssistantMessage adds an assistant message to the conversation
func (c *Conversation) AddAssistantMessage(content string) {
	c.AddMessage("assistant", content)
}

// ResponseMetadata contains additional information from the AI provider.
// Not all fields are populated by all providers.
type ResponseMetadata struct {
	// ModelUsed is the actual model version used (may differ from requested)
	ModelUsed        string      `json:"model_used,omitempty"`
	// PromptTokens is the number of tokens in the prompt
	PromptTokens     int         `json:"prompt_tokens,omitempty"`
	// CompletionTokens is the number of tokens in the completion
	CompletionTokens int         `json:"completion_tokens,omitempty"`
	// TotalTokens is the sum of prompt and completion tokens
	TotalTokens      int         `json:"total_tokens,omitempty"`
	// FinishReason indicates why generation ended (e.g., "stop", "length", "content_filter")
	FinishReason     string      `json:"finish_reason,omitempty"`
	// SafetyRatings contains provider-specific safety or content filter results
	SafetyRatings    interface{} `json:"safety_ratings,omitempty"`
	// RequestID for debugging and tracking
	RequestID        string      `json:"request_id,omitempty"`
	// LatencyMs is the time taken to generate response in milliseconds
	LatencyMs        int64       `json:"latency_ms,omitempty"`
}

// AiResponse combines the text content with response metadata.
// Use this when you need detailed information about token usage and performance.
type AiResponse struct {
	// Content is the actual text response from the AI
	Content  string           `json:"content"`
	// Metadata contains additional information about the response
	Metadata ResponseMetadata `json:"metadata"`
}

// StreamChunk represents a chunk of streaming response.
// When Finished is true, this is the final chunk and Metadata may be populated.
type StreamChunk struct {
	// Content of this chunk
	Content  string            `json:"content"`
	// Finished indicates if this is the final chunk
	Finished bool              `json:"finished"`
	// Metadata is only populated on the final chunk
	Metadata *ResponseMetadata `json:"metadata,omitempty"`
}

// RetryStrategy defines the retry behavior for failed requests.
type RetryStrategy string

const (
	// RetryStrategyFixed uses a fixed delay between retries
	RetryStrategyFixed               RetryStrategy = "fixed"
	// RetryStrategyLinear increases delay linearly with each attempt
	RetryStrategyLinear              RetryStrategy = "linear"
	// RetryStrategyExponentialBackoff doubles the delay with each attempt
	RetryStrategyExponentialBackoff  RetryStrategy = "exponential"
	// RetryStrategyExponentialWithJitter adds random jitter to prevent thundering herd
	RetryStrategyExponentialWithJitter RetryStrategy = "exponential_with_jitter"
)

// ClientConfig holds configuration options for AI clients.
// Use NewClientConfig to create a config with sensible defaults,
// then use the Set* methods to customize.
type ClientConfig struct {
	// Timeout for HTTP requests
	Timeout          time.Duration
	// Retries is the number of retry attempts for failed requests
	Retries          int
	// Temperature controls randomness (0.0-2.0), higher = more random
	Temperature      *float64
	// MaxTokens limits the response length
	MaxTokens        *int
	// TopP is nucleus sampling parameter (0.0-1.0)
	TopP             *float64
	// FrequencyPenalty reduces repetition of token sequences (-2.0 to 2.0)
	FrequencyPenalty *float64
	// PresencePenalty reduces repetition of any tokens that have appeared (-2.0 to 2.0)
	PresencePenalty  *float64
	// SystemMessage sets context for the AI assistant
	SystemMessage    *string
	// BaseURL allows custom endpoints (e.g., Azure OpenAI, local models)
	BaseURL          *string
	// RetryStrategy determines how delays are calculated between retries
	RetryStrategy    RetryStrategy
}

// NewClientConfig creates a new ClientConfig with default values
func NewClientConfig() *ClientConfig {
	return &ClientConfig{
		Timeout:       30 * time.Second,
		Retries:       3,
		RetryStrategy: RetryStrategyExponentialBackoff,
	}
}

// SetTimeout sets the timeout duration
func (c *ClientConfig) SetTimeout(timeout time.Duration) *ClientConfig {
	c.Timeout = timeout
	return c
}

// SetRetries sets the number of retries
func (c *ClientConfig) SetRetries(retries int) *ClientConfig {
	c.Retries = retries
	return c
}

// SetTemperature sets the temperature parameter
func (c *ClientConfig) SetTemperature(temperature float64) *ClientConfig {
	c.Temperature = &temperature
	return c
}

// SetMaxTokens sets the maximum number of tokens
func (c *ClientConfig) SetMaxTokens(maxTokens int) *ClientConfig {
	c.MaxTokens = &maxTokens
	return c
}

// SetTopP sets the top-p parameter
func (c *ClientConfig) SetTopP(topP float64) *ClientConfig {
	c.TopP = &topP
	return c
}

// SetFrequencyPenalty sets the frequency penalty parameter
func (c *ClientConfig) SetFrequencyPenalty(penalty float64) *ClientConfig {
	c.FrequencyPenalty = &penalty
	return c
}

// SetPresencePenalty sets the presence penalty parameter
func (c *ClientConfig) SetPresencePenalty(penalty float64) *ClientConfig {
	c.PresencePenalty = &penalty
	return c
}

// SetSystemMessage sets the system message
func (c *ClientConfig) SetSystemMessage(message string) *ClientConfig {
	c.SystemMessage = &message
	return c
}

// SetBaseURL sets the custom base URL for API endpoint
func (c *ClientConfig) SetBaseURL(url string) *ClientConfig {
	c.BaseURL = &url
	return c
}

// SetRetryStrategy sets the retry strategy
func (c *ClientConfig) SetRetryStrategy(strategy RetryStrategy) *ClientConfig {
	c.RetryStrategy = strategy
	return c
}

// AIClient defines the interface for all AI clients
type AIClient interface {
	// SendPrompt sends a single prompt and returns the response
	SendPrompt(ctx context.Context, prompt string) (string, error)

	// SendPromptWithMetadata sends a prompt and returns response with metadata
	SendPromptWithMetadata(ctx context.Context, prompt string) (*AiResponse, error)

	// SendConversation sends a conversation and returns the response
	SendConversation(ctx context.Context, conversation *Conversation) (string, error)

	// SendConversationWithMetadata sends a conversation and returns response with metadata
	SendConversationWithMetadata(ctx context.Context, conversation *Conversation) (*AiResponse, error)

	// StreamPrompt sends a prompt and returns a channel for streaming chunks
	StreamPrompt(ctx context.Context, prompt string) (<-chan StreamChunk, error)

	// StreamConversation sends a conversation and returns a channel for streaming chunks
	StreamConversation(ctx context.Context, conversation *Conversation) (<-chan StreamChunk, error)

	// SupportsStreaming returns true if the client supports streaming
	SupportsStreaming() bool

	// SupportsConversations returns true if the client supports conversations
	SupportsConversations() bool

	// Name returns the name of the client
	Name() string

	// Model returns the model identifier
	Model() string
}

// ParallelResult represents the result of a parallel execution across multiple clients.
// Either Result or Error will be populated, not both.
type ParallelResult struct {
	// ClientName identifies which client produced this result
	ClientName string
	// Result contains the successful response text
	Result     string
	// Error contains any error that occurred
	Error      error
}
