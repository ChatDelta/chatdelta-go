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

// OpenAIClient implements the AIClient interface for OpenAI's API
type OpenAIClient struct {
	apiKey     string
	model      string
	config     *ClientConfig
	httpClient *http.Client
}

// OpenAI API request/response structures
type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIRequest struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	Stream      bool            `json:"stream,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	MaxTokens   *int            `json:"max_tokens,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	FreqPenalty *float64        `json:"frequency_penalty,omitempty"`
	PresPenalty *float64        `json:"presence_penalty,omitempty"`
}

type openAIChoice struct {
	Index   int `json:"index"`
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Delta struct {
		Role    string `json:"role,omitempty"`
		Content string `json:"content,omitempty"`
	} `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type openAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

type openAIErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

type openAIErrorResponse struct {
	Error openAIErrorDetail `json:"error"`
}

// NewOpenAIClient creates a new OpenAI client
func NewOpenAIClient(apiKey, model string, config *ClientConfig) (*OpenAIClient, error) {
	if apiKey == "" {
		return nil, NewInvalidAPIKeyError()
	}

	if model == "" {
		model = "gpt-3.5-turbo"
	}

	if config == nil {
		config = NewClientConfig()
	}

	return &OpenAIClient{
		apiKey: apiKey,
		model:  model,
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}, nil
}

// SendPrompt sends a single prompt to OpenAI
func (c *OpenAIClient) SendPrompt(ctx context.Context, prompt string) (string, error) {
	conversation := NewConversation()
	if c.config.SystemMessage != nil {
		conversation.AddSystemMessage(*c.config.SystemMessage)
	}
	conversation.AddUserMessage(prompt)

	return c.SendConversation(ctx, conversation)
}

// SendConversation sends a conversation to OpenAI
func (c *OpenAIClient) SendConversation(ctx context.Context, conversation *Conversation) (string, error) {
	var result string
	var lastErr error

	operation := func() error {
		response, err := c.sendRequest(ctx, conversation, false)
		if err != nil {
			lastErr = err
			return err
		}

		if len(response.Choices) == 0 {
			lastErr = NewMissingFieldError("choices")
			return lastErr
		}

		result = response.Choices[0].Message.Content
		return nil
	}

	err := ExecuteWithRetry(ctx, c.config.Retries, operation)
	if err != nil {
		return "", err
	}

	return result, nil
}

// StreamPrompt streams a response for a single prompt
func (c *OpenAIClient) StreamPrompt(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	conversation := NewConversation()
	if c.config.SystemMessage != nil {
		conversation.AddSystemMessage(*c.config.SystemMessage)
	}
	conversation.AddUserMessage(prompt)

	return c.StreamConversation(ctx, conversation)
}

// StreamConversation streams a response for a conversation
func (c *OpenAIClient) StreamConversation(ctx context.Context, conversation *Conversation) (<-chan StreamChunk, error) {
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

// sendRequest sends a request to the OpenAI API
func (c *OpenAIClient) sendRequest(ctx context.Context, conversation *Conversation, stream bool) (*openAIResponse, error) {
	messages := make([]openAIMessage, len(conversation.Messages))
	for i, msg := range conversation.Messages {
		messages[i] = openAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	request := openAIRequest{
		Model:       c.model,
		Messages:    messages,
		Stream:      stream,
		Temperature: c.config.Temperature,
		MaxTokens:   c.config.MaxTokens,
		TopP:        c.config.TopP,
		FreqPenalty: c.config.FrequencyPenalty,
		PresPenalty: c.config.PresencePenalty,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, NewJSONParseError(err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, NewConnectionError(err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

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
		var errorResp openAIErrorResponse
		if err := json.Unmarshal(body, &errorResp); err == nil {
			return nil, c.parseAPIError(resp.StatusCode, &errorResp.Error)
		}
		return nil, NewServerError(resp.StatusCode, string(body))
	}

	var response openAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, NewJSONParseError(err)
	}

	return &response, nil
}

// streamRequest handles streaming requests
func (c *OpenAIClient) streamRequest(ctx context.Context, conversation *Conversation, resultChan chan<- StreamChunk) error {
	messages := make([]openAIMessage, len(conversation.Messages))
	for i, msg := range conversation.Messages {
		messages[i] = openAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	request := openAIRequest{
		Model:       c.model,
		Messages:    messages,
		Stream:      true,
		Temperature: c.config.Temperature,
		MaxTokens:   c.config.MaxTokens,
		TopP:        c.config.TopP,
		FreqPenalty: c.config.FrequencyPenalty,
		PresPenalty: c.config.PresencePenalty,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return NewJSONParseError(err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return NewConnectionError(err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
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
		var errorResp openAIErrorResponse
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

			var response openAIResponse
			if err := json.Unmarshal([]byte(data), &response); err != nil {
				continue // Skip malformed chunks
			}

			if len(response.Choices) > 0 {
				content := response.Choices[0].Delta.Content
				finished := response.Choices[0].FinishReason != nil

				resultChan <- StreamChunk{
					Content:  content,
					Finished: finished,
				}

				if finished {
					return nil
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return NewStreamReadError(err)
	}

	return nil
}

// parseAPIError parses OpenAI API errors
func (c *OpenAIClient) parseAPIError(statusCode int, error *openAIErrorDetail) *ClientError {
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
		return NewPermissionDeniedError("OpenAI API")
	default:
		return NewServerError(statusCode, error.Message)
	}
}

// SupportsStreaming returns true (OpenAI supports streaming)
func (c *OpenAIClient) SupportsStreaming() bool {
	return true
}

// SupportsConversations returns true (OpenAI supports conversations)
func (c *OpenAIClient) SupportsConversations() bool {
	return true
}

// Name returns the client name
func (c *OpenAIClient) Name() string {
	return "OpenAI"
}

// Model returns the model identifier
func (c *OpenAIClient) Model() string {
	return c.model
}
