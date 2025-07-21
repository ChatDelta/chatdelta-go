package chatdelta

import (
	"context"
	"time"
)

// Message represents a single message in a conversation
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Conversation represents a collection of messages
type Conversation struct {
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

// StreamChunk represents a chunk of streaming response
type StreamChunk struct {
	Content  string `json:"content"`
	Finished bool   `json:"finished"`
}

// ClientConfig holds configuration options for AI clients
type ClientConfig struct {
	Timeout          time.Duration
	Retries          int
	Temperature      *float64
	MaxTokens        *int
	TopP             *float64
	FrequencyPenalty *float64
	PresencePenalty  *float64
	SystemMessage    *string
}

// NewClientConfig creates a new ClientConfig with default values
func NewClientConfig() *ClientConfig {
	return &ClientConfig{
		Timeout: 30 * time.Second,
		Retries: 3,
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

// AIClient defines the interface for all AI clients
type AIClient interface {
	// SendPrompt sends a single prompt and returns the response
	SendPrompt(ctx context.Context, prompt string) (string, error)

	// SendConversation sends a conversation and returns the response
	SendConversation(ctx context.Context, conversation *Conversation) (string, error)

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

// ParallelResult represents the result of a parallel execution
type ParallelResult struct {
	ClientName string
	Result     string
	Error      error
}
