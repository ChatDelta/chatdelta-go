package chatdelta

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// GeminiClient implements the AIClient interface for Google's Gemini API
type GeminiClient struct {
	apiKey     string
	model      string
	config     *ClientConfig
	httpClient *http.Client
}

// Gemini API request/response structures
type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type geminiSafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

type geminiCandidate struct {
	Content       geminiContent        `json:"content"`
	FinishReason  string               `json:"finishReason,omitempty"`
	Index         int                  `json:"index"`
	SafetyRatings []geminiSafetyRating `json:"safetyRatings,omitempty"`
}

type geminiUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type geminiGenerationConfig struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"topP,omitempty"`
	MaxTokens   *int     `json:"maxOutputTokens,omitempty"`
}

type geminiSystemInstruction struct {
	Parts []geminiPart `json:"parts"`
}

type geminiRequest struct {
	Contents           []geminiContent         `json:"contents"`
	GenerationConfig   *geminiGenerationConfig `json:"generationConfig,omitempty"`
	SystemInstruction  *geminiSystemInstruction `json:"systemInstruction,omitempty"`
}

type geminiResponse struct {
	Candidates     []geminiCandidate    `json:"candidates"`
	UsageMetadata  *geminiUsageMetadata `json:"usageMetadata,omitempty"`
	PromptFeedback struct {
		SafetyRatings []geminiSafetyRating `json:"safetyRatings,omitempty"`
	} `json:"promptFeedback,omitempty"`
}

type geminiErrorDetail struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

type geminiErrorResponse struct {
	Error geminiErrorDetail `json:"error"`
}

// NewGeminiClient creates a new Gemini client
func NewGeminiClient(apiKey, model string, config *ClientConfig) (*GeminiClient, error) {
	if apiKey == "" {
		return nil, NewInvalidAPIKeyError()
	}
	
	if model == "" {
		model = "gemini-1.5-flash"
	}
	
	if config == nil {
		config = NewClientConfig()
	}
	
	return &GeminiClient{
		apiKey: apiKey,
		model:  model,
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}, nil
}

// SendPrompt sends a single prompt to Gemini
func (c *GeminiClient) SendPrompt(ctx context.Context, prompt string) (string, error) {
	conversation := NewConversation()
	if c.config.SystemMessage != nil {
		conversation.AddSystemMessage(*c.config.SystemMessage)
	}
	conversation.AddUserMessage(prompt)
	
	return c.SendConversation(ctx, conversation)
}

// SendConversation sends a conversation to Gemini
func (c *GeminiClient) SendConversation(ctx context.Context, conversation *Conversation) (string, error) {
	var result string
	var lastErr error
	
	operation := func() error {
		response, err := c.sendRequest(ctx, conversation)
		if err != nil {
			lastErr = err
			return err
		}
		
		if len(response.Candidates) == 0 {
			lastErr = NewMissingFieldError("candidates")
			return lastErr
		}
		
		candidate := response.Candidates[0]
		if len(candidate.Content.Parts) == 0 {
			lastErr = NewMissingFieldError("parts")
			return lastErr
		}
		
		result = candidate.Content.Parts[0].Text
		return nil
	}
	
	err := ExecuteWithRetry(ctx, c.config.Retries, operation)
	if err != nil {
		return "", err
	}
	
	return result, nil
}

// StreamPrompt streams a response for a single prompt (not implemented for Gemini yet)
func (c *GeminiClient) StreamPrompt(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	// Gemini doesn't support streaming in this implementation
	// Fall back to non-streaming and emit the result as a single chunk
	resultChan := make(chan StreamChunk, 1)
	
	go func() {
		defer close(resultChan)
		
		result, err := c.SendPrompt(ctx, prompt)
		if err != nil {
			resultChan <- StreamChunk{Content: "", Finished: true}
			return
		}
		
		resultChan <- StreamChunk{Content: result, Finished: true}
	}()
	
	return resultChan, nil
}

// StreamConversation streams a response for a conversation (not implemented for Gemini yet)
func (c *GeminiClient) StreamConversation(ctx context.Context, conversation *Conversation) (<-chan StreamChunk, error) {
	// Gemini doesn't support streaming in this implementation
	// Fall back to non-streaming and emit the result as a single chunk
	resultChan := make(chan StreamChunk, 1)
	
	go func() {
		defer close(resultChan)
		
		result, err := c.SendConversation(ctx, conversation)
		if err != nil {
			resultChan <- StreamChunk{Content: "", Finished: true}
			return
		}
		
		resultChan <- StreamChunk{Content: result, Finished: true}
	}()
	
	return resultChan, nil
}

// sendRequest sends a request to the Gemini API
func (c *GeminiClient) sendRequest(ctx context.Context, conversation *Conversation) (*geminiResponse, error) {
	// Convert messages to Gemini format
	var contents []geminiContent
	var systemInstruction *geminiSystemInstruction
	
	// Handle system messages
	var systemMessages []string
	if c.config.SystemMessage != nil {
		systemMessages = append(systemMessages, *c.config.SystemMessage)
	}
	
	for _, msg := range conversation.Messages {
		if msg.Role == "system" {
			systemMessages = append(systemMessages, msg.Content)
		} else {
			// Map roles: "user" -> "user", "assistant" -> "model"
			role := msg.Role
			if role == "assistant" {
				role = "model"
			}
			
			contents = append(contents, geminiContent{
				Parts: []geminiPart{{Text: msg.Content}},
				Role:  role,
			})
		}
	}
	
	// Combine system messages
	if len(systemMessages) > 0 {
		systemInstruction = &geminiSystemInstruction{
			Parts: []geminiPart{{Text: strings.Join(systemMessages, "\n\n")}},
		}
	}
	
	// Build generation config
	var genConfig *geminiGenerationConfig
	if c.config.Temperature != nil || c.config.TopP != nil || c.config.MaxTokens != nil {
		genConfig = &geminiGenerationConfig{
			Temperature: c.config.Temperature,
			TopP:        c.config.TopP,
			MaxTokens:   c.config.MaxTokens,
		}
	}
	
	request := geminiRequest{
		Contents:          contents,
		GenerationConfig:  genConfig,
		SystemInstruction: systemInstruction,
	}
	
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, NewJSONParseError(err)
	}
	
	// Build URL with API key
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", c.model, c.apiKey)
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, NewConnectionError(err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	
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
		var errorResp geminiErrorResponse
		if err := json.Unmarshal(body, &errorResp); err == nil {
			return nil, c.parseAPIError(resp.StatusCode, &errorResp.Error)
		}
		return nil, NewServerError(resp.StatusCode, string(body))
	}
	
	var response geminiResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, NewJSONParseError(err)
	}
	
	return &response, nil
}

// parseAPIError parses Gemini API errors
func (c *GeminiClient) parseAPIError(statusCode int, error *geminiErrorDetail) *ClientError {
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
		return NewPermissionDeniedError("Gemini API")
	default:
		return NewServerError(statusCode, error.Message)
	}
}

// SupportsStreaming returns false (Gemini streaming not implemented yet)
func (c *GeminiClient) SupportsStreaming() bool {
	return false
}

// SupportsConversations returns true (Gemini supports conversations)
func (c *GeminiClient) SupportsConversations() bool {
	return true
}

// Name returns the client name
func (c *GeminiClient) Name() string {
	return "Gemini"
}

// Model returns the model identifier
func (c *GeminiClient) Model() string {
	return c.model
}