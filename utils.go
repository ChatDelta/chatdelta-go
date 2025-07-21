package chatdelta

import (
	"context"
	"math"
	"sync"
	"time"
)

// ExecuteWithRetry executes a function with retry logic and exponential backoff
func ExecuteWithRetry(ctx context.Context, retries int, operation func() error) error {
	var lastErr error
	
	for attempt := 0; attempt <= retries; attempt++ {
		// Execute the operation
		err := operation()
		if err == nil {
			return nil // Success
		}
		
		lastErr = err
		
		// Check if the error is retryable
		if !IsRetryableError(err) {
			return err // Don't retry non-retryable errors
		}
		
		// Don't sleep after the last attempt
		if attempt == retries {
			break
		}
		
		// Calculate backoff delay: 1s, 2s, 3s, etc.
		delay := time.Duration(attempt+1) * time.Second
		
		// Check if context is cancelled
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
			// Continue to next attempt
		}
	}
	
	return lastErr
}

// ExecuteWithExponentialBackoff executes a function with exponential backoff
func ExecuteWithExponentialBackoff(ctx context.Context, retries int, baseDelay time.Duration, operation func() error) error {
	var lastErr error
	
	for attempt := 0; attempt <= retries; attempt++ {
		// Execute the operation
		err := operation()
		if err == nil {
			return nil // Success
		}
		
		lastErr = err
		
		// Check if the error is retryable
		if !IsRetryableError(err) {
			return err // Don't retry non-retryable errors
		}
		
		// Don't sleep after the last attempt
		if attempt == retries {
			break
		}
		
		// Calculate exponential backoff delay
		delay := time.Duration(math.Pow(2, float64(attempt))) * baseDelay
		
		// Cap the maximum delay at 30 seconds
		if delay > 30*time.Second {
			delay = 30 * time.Second
		}
		
		// Check if context is cancelled
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
			// Continue to next attempt
		}
	}
	
	return lastErr
}

// ExecuteParallel executes multiple AI clients in parallel with the same prompt
func ExecuteParallel(ctx context.Context, clients []AIClient, prompt string) []ParallelResult {
	results := make([]ParallelResult, len(clients))
	var wg sync.WaitGroup
	
	for i, client := range clients {
		wg.Add(1)
		go func(index int, c AIClient) {
			defer wg.Done()
			
			result, err := c.SendPrompt(ctx, prompt)
			results[index] = ParallelResult{
				ClientName: c.Name(),
				Result:     result,
				Error:      err,
			}
		}(i, client)
	}
	
	wg.Wait()
	return results
}

// ExecuteParallelConversation executes multiple AI clients in parallel with the same conversation
func ExecuteParallelConversation(ctx context.Context, clients []AIClient, conversation *Conversation) []ParallelResult {
	results := make([]ParallelResult, len(clients))
	var wg sync.WaitGroup
	
	for i, client := range clients {
		wg.Add(1)
		go func(index int, c AIClient) {
			defer wg.Done()
			
			var result string
			var err error
			
			if c.SupportsConversations() {
				result, err = c.SendConversation(ctx, conversation)
			} else {
				// Fallback to sending the last user message as a prompt
				if len(conversation.Messages) > 0 {
					lastMessage := conversation.Messages[len(conversation.Messages)-1]
					if lastMessage.Role == "user" {
						result, err = c.SendPrompt(ctx, lastMessage.Content)
					} else {
						err = NewConfigError("no user message found in conversation")
					}
				} else {
					err = NewConfigError("empty conversation")
				}
			}
			
			results[index] = ParallelResult{
				ClientName: c.Name(),
				Result:     result,
				Error:      err,
			}
		}(i, client)
	}
	
	wg.Wait()
	return results
}

// NewConfigError creates a configuration error (helper for ExecuteParallelConversation)
func NewConfigError(message string) *ClientError {
	return &ClientError{
		Type:    ErrorTypeConfig,
		Code:    "config_error",
		Message: message,
	}
}

// MergeStreamChunks combines multiple stream chunks into a single string
func MergeStreamChunks(chunks <-chan StreamChunk) (string, error) {
	var result string
	
	for chunk := range chunks {
		result += chunk.Content
		if chunk.Finished {
			break
		}
	}
	
	return result, nil
}

// StreamToString converts a streaming response to a string
func StreamToString(ctx context.Context, client AIClient, prompt string) (string, error) {
	if !client.SupportsStreaming() {
		return client.SendPrompt(ctx, prompt)
	}
	
	chunks, err := client.StreamPrompt(ctx, prompt)
	if err != nil {
		return "", err
	}
	
	result, err := MergeStreamChunks(chunks)
	return result, err
}

// StreamConversationToString converts a streaming conversation response to a string
func StreamConversationToString(ctx context.Context, client AIClient, conversation *Conversation) (string, error) {
	if !client.SupportsStreaming() {
		return client.SendConversation(ctx, conversation)
	}
	
	chunks, err := client.StreamConversation(ctx, conversation)
	if err != nil {
		return "", err
	}
	
	result, err := MergeStreamChunks(chunks)
	return result, err
}

// ValidateConfig validates a ClientConfig
func ValidateConfig(config *ClientConfig) error {
	if config.Timeout <= 0 {
		return NewInvalidParameterError("timeout", config.Timeout.String())
	}
	
	if config.Retries < 0 {
		return NewInvalidParameterError("retries", string(rune(config.Retries)))
	}
	
	if config.Temperature != nil && (*config.Temperature < 0 || *config.Temperature > 2) {
		return NewInvalidParameterError("temperature", string(rune(int(*config.Temperature))))
	}
	
	if config.MaxTokens != nil && *config.MaxTokens <= 0 {
		return NewInvalidParameterError("max_tokens", string(rune(*config.MaxTokens)))
	}
	
	if config.TopP != nil && (*config.TopP < 0 || *config.TopP > 1) {
		return NewInvalidParameterError("top_p", string(rune(int(*config.TopP))))
	}
	
	if config.FrequencyPenalty != nil && (*config.FrequencyPenalty < -2 || *config.FrequencyPenalty > 2) {
		return NewInvalidParameterError("frequency_penalty", string(rune(int(*config.FrequencyPenalty))))
	}
	
	if config.PresencePenalty != nil && (*config.PresencePenalty < -2 || *config.PresencePenalty > 2) {
		return NewInvalidParameterError("presence_penalty", string(rune(int(*config.PresencePenalty))))
	}
	
	return nil
}