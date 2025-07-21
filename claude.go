package chatdelta

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// ClaudeClient implements the AIClient interface for Anthropic's Claude API
type ClaudeClient struct {
	apiKey     string
	model      string
	config     *ClientConfig
	httpClient *http.Client
}

// Claude API request/response structures
type claudeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type claudeRequest struct {
	Model       string          `json:"model"`
	Messages    []claudeMessage `json:"messages"`
	System      string          `json:"system,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens"`
	TopP        *float64        `json:"top_p,omitempty"`
}

type claudeContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type claudeDelta struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type claudeResponse struct {
	ID      string          `json:"id"`
	Type    string          `json:"type"`
	Role    string          `json:"role,omitempty"`
	Content []claudeContent `json:"content,omitempty"`
	Model   string          `json:"model,omitempty"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
	Delta      *claudeDelta `json:"delta,omitempty"`
	StopReason *string      `json:"stop_reason,omitempty"`
}

type claudeErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type claudeErrorResponse struct {
	Type  string            `json:"type"`
	Error claudeErrorDetail `json:"error"`
}

// NewClaudeClient creates a new Claude client
func NewClaudeClient(apiKey, model string, config *ClientConfig) (*ClaudeClient, error) {
	if apiKey == "" {
		return nil, NewInvalidAPIKeyError()
	}

	if model == "" {
		model = "claude-3-haiku-20240307"
	}

	if config == nil {
		config = NewClientConfig()
	}

	return &ClaudeClient{
		apiKey: apiKey,
		model:  model,
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}, nil
}

// SendPrompt sends a single prompt to Claude
func (c *ClaudeClient) SendPrompt(ctx context.Context, prompt string) (string, error) {
	conversation := NewConversation()
	conversation.AddUserMessage(prompt)

	return c.SendConversation(ctx, conversation)
}

// SendConversation sends a conversation to Claude
func (c *ClaudeClient) SendConversation(ctx context.Context, conversation *Conversation) (string, error) {
	var result string
	var lastErr error

	operation := func() error {
		response, err := c.sendRequest(ctx, conversation, false)
		if err != nil {
			lastErr = err
			return err
		}

		if len(response.Content) == 0 {
			lastErr = NewMissingFieldError("content")
			return lastErr
		}

		result = response.Content[0].Text
		return nil
	}

	err := ExecuteWithRetry(ctx, c.config.Retries, operation)
	if err != nil {
		return "", err
	}

	return result, nil
}

// StreamPrompt streams a response for a single prompt
func (c *ClaudeClient) StreamPrompt(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	conversation := NewConversation()
	conversation.AddUserMessage(prompt)

	return c.StreamConversation(ctx, conversation)
}

// StreamConversation streams a response for a conversation
func (c *ClaudeClient) StreamConversation(ctx context.Context, conversation *Conversation) (<-chan StreamChunk, error) {
	resultChan := make(chan StreamChunk, 10)

	go func() {
		defer close(resultChan)

		operation := func() error {
			return c.streamRequest(ctx, conversation, resultChan)
		}

		err := ExecuteWithRetry(ctx, c.config.Retries, operation)
		if err != nil {
			resultChan <- StreamChunk{Content: "", Finished: true}
		}
	}()

	return resultChan, nil
}

// sendRequest sends a request to the Claude API
func (c *ClaudeClient) sendRequest(ctx context.Context, conversation *Conversation, stream bool) (*claudeResponse, error) {
	// Separate system messages from conversation messages
	var systemMessage string
	var messages []claudeMessage

	// Start with system message from config if available
	if c.config.SystemMessage != nil {
		systemMessage = *c.config.SystemMessage
	}

	for _, msg := range conversation.Messages {
		if msg.Role == "system" {
			// Append system messages to the system prompt
			if systemMessage != "" {
				systemMessage += "\n\n" + msg.Content
			} else {
				systemMessage = msg.Content
			}
		} else {
			messages = append(messages, claudeMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	maxTokens := 1024
	if c.config.MaxTokens != nil {
		maxTokens = *c.config.MaxTokens
	}

	request := claudeRequest{
		Model:       c.model,
		Messages:    messages,
		System:      systemMessage,
		Stream:      stream,
		Temperature: c.config.Temperature,
		MaxTokens:   maxTokens,
		TopP:        c.config.TopP,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, NewJSONParseError(err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, NewConnectionError(err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, NewTimeoutError(c.config.Timeout)
		}
		return nil, NewConnectionError(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, NewConnectionError(err)
	}

	if resp.StatusCode != http.StatusOK {
		var errorResp claudeErrorResponse
		if err := json.Unmarshal(body, &errorResp); err == nil {
			return nil, c.parseAPIError(resp.StatusCode, &errorResp.Error)
		}
		return nil, NewServerError(resp.StatusCode, string(body))
	}

	var response claudeResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, NewJSONParseError(err)
	}

	return &response, nil
}

// streamRequest handles streaming requests
func (c *ClaudeClient) streamRequest(ctx context.Context, conversation *Conversation, resultChan chan<- StreamChunk) error {
	// Separate system messages from conversation messages
	var systemMessage string
	var messages []claudeMessage

	// Start with system message from config if available
	if c.config.SystemMessage != nil {
		systemMessage = *c.config.SystemMessage
	}

	for _, msg := range conversation.Messages {
		if msg.Role == "system" {
			// Append system messages to the system prompt
			if systemMessage != "" {
				systemMessage += "\n\n" + msg.Content
			} else {
				systemMessage = msg.Content
			}
		} else {
			messages = append(messages, claudeMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	maxTokens := 1024
	if c.config.MaxTokens != nil {
		maxTokens = *c.config.MaxTokens
	}

	request := claudeRequest{
		Model:       c.model,
		Messages:    messages,
		System:      systemMessage,
		Stream:      true,
		Temperature: c.config.Temperature,
		MaxTokens:   maxTokens,
		TopP:        c.config.TopP,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return NewJSONParseError(err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(jsonData))
	if err != nil {
		return NewConnectionError(err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return NewTimeoutError(c.config.Timeout)
		}
		return NewConnectionError(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		var errorResp claudeErrorResponse
		if err := json.Unmarshal(body, &errorResp); err == nil {
			return c.parseAPIError(resp.StatusCode, &errorResp.Error)
		}
		return NewServerError(resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				resultChan <- StreamChunk{Content: "", Finished: true}
				return nil
			}

			var response claudeResponse
			if err := json.Unmarshal([]byte(data), &response); err != nil {
				continue // Skip malformed chunks
			}

			// Handle different event types
			switch response.Type {
			case "content_block_delta":
				if response.Delta != nil && response.Delta.Type == "text_delta" {
					resultChan <- StreamChunk{
						Content:  response.Delta.Text,
						Finished: false,
					}
				}
			case "message_stop":
				resultChan <- StreamChunk{Content: "", Finished: true}
				return nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return NewStreamReadError(err)
	}

	return nil
}

// parseAPIError parses Claude API errors
func (c *ClaudeClient) parseAPIError(statusCode int, error *claudeErrorDetail) *ClientError {
	switch statusCode {
	case http.StatusUnauthorized:
		return NewInvalidAPIKeyError()
	case http.StatusTooManyRequests:
		return NewRateLimitError(nil)
	case http.StatusBadRequest:
		if strings.Contains(strings.ToLower(error.Message), "model") {
			return NewInvalidModelError(c.model)
		}
		return NewBadRequestError(error.Message)
	case http.StatusForbidden:
		return NewPermissionDeniedError("Claude API")
	default:
		return NewServerError(statusCode, error.Message)
	}
}

// SupportsStreaming returns true (Claude supports streaming)
func (c *ClaudeClient) SupportsStreaming() bool {
	return true
}

// SupportsConversations returns true (Claude supports conversations)
func (c *ClaudeClient) SupportsConversations() bool {
	return true
}

// Name returns the client name
func (c *ClaudeClient) Name() string {
	return "Claude"
}

// Model returns the model identifier
func (c *ClaudeClient) Model() string {
	return c.model
}
