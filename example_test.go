package chatdelta_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/chatdelta/chatdelta-go"
)

// ExampleCreateClient demonstrates how to create different AI clients
func ExampleCreateClient() {
	// Create an OpenAI client
	openaiClient, err := chatdelta.CreateClient("openai", "your-api-key", "gpt-3.5-turbo", nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Created %s client with model %s\n", openaiClient.Name(), openaiClient.Model())

	// Create a Claude client with custom configuration
	config := chatdelta.NewClientConfig().
		SetTimeout(45 * time.Second).
		SetTemperature(0.7).
		SetMaxTokens(2048).
		SetSystemMessage("You are a helpful AI assistant.")

	claudeClient, err := chatdelta.CreateClient("claude", "your-api-key", "claude-3-haiku-20240307", config)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Created %s client with model %s\n", claudeClient.Name(), claudeClient.Model())
}

// ExampleAIClient_SendPrompt demonstrates sending a simple prompt
func ExampleAIClient_SendPrompt() {
	client, err := chatdelta.CreateClient("openai", "your-api-key", "gpt-3.5-turbo", nil)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	response, err := client.SendPrompt(ctx, "What is the capital of France?")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Response: %s\n", response)
}

// ExampleAIClient_SendConversation demonstrates conversation handling
func ExampleAIClient_SendConversation() {
	client, err := chatdelta.CreateClient("claude", "your-api-key", "", nil)
	if err != nil {
		log.Fatal(err)
	}

	// Build a conversation
	conversation := chatdelta.NewConversation()
	conversation.AddSystemMessage("You are a helpful math tutor.")
	conversation.AddUserMessage("What is 2 + 2?")
	conversation.AddAssistantMessage("2 + 2 equals 4.")
	conversation.AddUserMessage("What about 3 + 3?")

	ctx := context.Background()
	response, err := client.SendConversation(ctx, conversation)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Response: %s\n", response)
}

// ExampleAIClient_StreamPrompt demonstrates streaming responses
func ExampleAIClient_StreamPrompt() {
	client, err := chatdelta.CreateClient("openai", "your-api-key", "gpt-3.5-turbo", nil)
	if err != nil {
		log.Fatal(err)
	}

	if !client.SupportsStreaming() {
		fmt.Printf("%s client doesn't support streaming\n", client.Name())
		return
	}

	ctx := context.Background()
	chunks, err := client.StreamPrompt(ctx, "Write a short poem about Go programming.")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Streaming response from %s:\n", client.Name())
	for chunk := range chunks {
		if chunk.Content != "" {
			fmt.Print(chunk.Content)
		}
		if chunk.Finished {
			break
		}
	}
	fmt.Println()
}

// ExampleExecuteParallel demonstrates parallel execution across multiple providers
func ExampleExecuteParallel() {
	// Create multiple clients
	var clients []chatdelta.AIClient

	if openaiClient, err := chatdelta.CreateClient("openai", os.Getenv("OPENAI_API_KEY"), "", nil); err == nil {
		clients = append(clients, openaiClient)
	}

	if claudeClient, err := chatdelta.CreateClient("claude", os.Getenv("CLAUDE_API_KEY"), "", nil); err == nil {
		clients = append(clients, claudeClient)
	}

	if geminiClient, err := chatdelta.CreateClient("gemini", os.Getenv("GEMINI_API_KEY"), "", nil); err == nil {
		clients = append(clients, geminiClient)
	}

	if len(clients) == 0 {
		fmt.Println("No clients available (check API keys)")
		return
	}

	// Execute the same prompt across all providers
	ctx := context.Background()
	prompt := "What is the meaning of life?"
	results := chatdelta.ExecuteParallel(ctx, clients, prompt)

	fmt.Printf("Results from %d providers:\n", len(results))
	for _, result := range results {
		fmt.Printf("\n=== %s ===\n", result.ClientName)
		if result.Error != nil {
			fmt.Printf("Error: %v\n", result.Error)
		} else {
			fmt.Printf("Response: %s\n", result.Result)
		}
	}
}

// ExampleClientConfig demonstrates configuration options
func ExampleClientConfig() {
	config := chatdelta.NewClientConfig().
		SetTimeout(60 * time.Second).      // 60 second timeout
		SetRetries(5).                     // 5 retry attempts
		SetTemperature(0.8).               // Creative temperature
		SetMaxTokens(1024).                // Limit output length
		SetTopP(0.9).                      // Nucleus sampling
		SetFrequencyPenalty(0.1).          // Reduce repetition
		SetPresencePenalty(0.1).           // Encourage topic diversity
		SetSystemMessage("You are a creative writing assistant.")

	client, err := chatdelta.CreateClient("openai", "your-api-key", "gpt-4", config)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Created %s client with custom configuration\n", client.Name())
}

// ExampleGetAvailableProviders demonstrates checking for available API keys
func ExampleGetAvailableProviders() {
	available := chatdelta.GetAvailableProviders()
	
	if len(available) == 0 {
		fmt.Println("No AI providers available (no API keys found in environment)")
		fmt.Println("Set one of these environment variables:")
		fmt.Println("  OPENAI_API_KEY or CHATGPT_API_KEY")
		fmt.Println("  ANTHROPIC_API_KEY or CLAUDE_API_KEY")
		fmt.Println("  GOOGLE_API_KEY or GEMINI_API_KEY")
		return
	}

	fmt.Printf("Available providers: %v\n", available)
	
	// Create clients for all available providers
	for _, provider := range available {
		client, err := chatdelta.CreateClient(provider, "", "", nil)
		if err != nil {
			fmt.Printf("Error creating %s client: %v\n", provider, err)
			continue
		}
		
		info := chatdelta.GetClientInfo(client)
		fmt.Printf("  %s: model=%s, streaming=%t, conversations=%t\n", 
			info.Name, info.Model, info.SupportsStreaming, info.SupportsConversations)
	}
}

// Example_errorHandling demonstrates error handling
func Example_errorHandling() {
	// Try to create a client with invalid API key
	client, err := chatdelta.CreateClient("openai", "invalid-key", "", nil)
	if err != nil {
		fmt.Printf("Error creating client: %v\n", err)
		return
	}

	ctx := context.Background()
	_, err = client.SendPrompt(ctx, "Hello")
	
	if err != nil {
		// Check error type
		if chatdelta.IsAuthenticationError(err) {
			fmt.Println("Authentication error - check your API key")
		} else if chatdelta.IsNetworkError(err) {
			fmt.Println("Network error - check your connection")
		} else if chatdelta.IsRetryableError(err) {
			fmt.Println("Retryable error - the library will automatically retry")
		} else {
			fmt.Printf("Other error: %v\n", err)
		}
	}
}

// ExampleConversation demonstrates conversation building
func ExampleConversation() {
	conversation := chatdelta.NewConversation()
	
	// Add messages to build a conversation
	conversation.AddSystemMessage("You are a helpful math tutor.")
	conversation.AddUserMessage("I need help with algebra.")
	conversation.AddAssistantMessage("I'd be happy to help you with algebra! What specific topic are you working on?")
	conversation.AddUserMessage("How do I solve linear equations?")
	
	fmt.Printf("Conversation has %d messages:\n", len(conversation.Messages))
	for i, msg := range conversation.Messages {
		fmt.Printf("%d. %s: %s\n", i+1, msg.Role, msg.Content)
	}
}