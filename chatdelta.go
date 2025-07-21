// Package chatdelta provides a unified interface for interacting with multiple AI APIs
// including OpenAI, Anthropic Claude, and Google Gemini.
//
// The package supports both synchronous and streaming responses, conversation handling,
// parallel execution across multiple providers, comprehensive error handling, and
// configurable retry logic with exponential backoff.
//
// Basic usage:
//
//	client, err := chatdelta.CreateClient("openai", "your-api-key", "gpt-3.5-turbo", nil)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	response, err := client.SendPrompt(context.Background(), "Hello, how are you?")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	fmt.Println(response)
//
// Advanced usage with configuration:
//
//	config := chatdelta.NewClientConfig().
//		SetTimeout(60 * time.Second).
//		SetTemperature(0.7).
//		SetMaxTokens(2048).
//		SetSystemMessage("You are a helpful assistant.")
//
//	client, err := chatdelta.CreateClient("claude", "your-api-key", "claude-3-haiku-20240307", config)
//	if err != nil {
//		log.Fatal(err)
//	}
//
// Conversation handling:
//
//	conversation := chatdelta.NewConversation()
//	conversation.AddSystemMessage("You are a helpful math tutor.")
//	conversation.AddUserMessage("What is 2 + 2?")
//	conversation.AddAssistantMessage("2 + 2 equals 4.")
//	conversation.AddUserMessage("What about 3 + 3?")
//
//	response, err := client.SendConversation(context.Background(), conversation)
//
// Streaming responses:
//
//	if client.SupportsStreaming() {
//		chunks, err := client.StreamPrompt(context.Background(), "Write a poem")
//		if err != nil {
//			log.Fatal(err)
//		}
//
//		for chunk := range chunks {
//			fmt.Print(chunk.Content)
//			if chunk.Finished {
//				break
//			}
//		}
//	}
//
// Parallel execution:
//
//	clients := []chatdelta.AIClient{client1, client2, client3}
//	results := chatdelta.ExecuteParallel(context.Background(), clients, "What is the meaning of life?")
//
//	for _, result := range results {
//		fmt.Printf("%s: %s\n", result.ClientName, result.Result)
//	}
//
// Environment variables:
// The library automatically detects API keys from environment variables:
//   - OpenAI: OPENAI_API_KEY or CHATGPT_API_KEY
//   - Anthropic: ANTHROPIC_API_KEY or CLAUDE_API_KEY
//   - Google: GOOGLE_API_KEY or GEMINI_API_KEY
//
// Error handling:
// The library provides comprehensive error handling with specific error types
// and helper functions for error classification:
//
//	_, err := client.SendPrompt(ctx, "Hello")
//	if err != nil {
//		if chatdelta.IsAuthenticationError(err) {
//			// Handle authentication error
//		} else if chatdelta.IsNetworkError(err) {
//			// Handle network error
//		} else if chatdelta.IsRetryableError(err) {
//			// Library will automatically retry retryable errors
//		}
//	}
package chatdelta

const (
	// Version of the chatdelta-go library
	Version = "1.0.0"
	
	// DefaultTimeout is the default timeout for HTTP requests
	DefaultTimeout = 30
	
	// DefaultRetries is the default number of retry attempts
	DefaultRetries = 3
)

// Package-level convenience functions

// QuickPrompt is a convenience function for sending a quick prompt to a provider
// without needing to manage client instances. It uses environment variables
// for API keys and default configurations.
func QuickPrompt(provider, prompt string) (string, error) {
	client, err := CreateClient(provider, "", "", nil)
	if err != nil {
		return "", err
	}
	
	return client.SendPrompt(nil, prompt)
}