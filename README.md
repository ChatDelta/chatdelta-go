# ChatDelta Go

A complete Go library for connecting to multiple AI APIs with a unified interface. Supports OpenAI, Anthropic Claude, and Google Gemini with streaming, conversation handling, parallel execution, and comprehensive error handling.

## Features

- ✅ **Multiple AI Providers**: OpenAI, Anthropic Claude, Google Gemini
- ✅ **Unified Interface**: Same API for all providers
- ✅ **Streaming Support**: Real-time response streaming where supported
- ✅ **Conversation Handling**: Multi-turn conversations with context
- ✅ **Parallel Execution**: Execute the same prompt across multiple providers simultaneously  
- ✅ **Comprehensive Error Handling**: Detailed error types with retry logic
- ✅ **Configurable**: Timeout, retries, temperature, tokens, and more
- ✅ **Environment Integration**: Automatic API key detection from environment variables
- ✅ **Type Safe**: Full Go type safety with comprehensive interfaces
- 🆕 **Response Metadata**: Token counts, latency tracking, and request IDs (v0.3.0)
- 🆕 **Chat Sessions**: High-level conversation management with automatic history (v0.3.0)
- 🆕 **Custom Base URLs**: Support for Azure OpenAI and local models (v0.3.0)
- 🆕 **Advanced Retry Strategies**: Fixed, Linear, Exponential, and ExponentialWithJitter (v0.3.0)

## Installation

```bash
go get github.com/chatdelta/chatdelta-go
```

## Quick Start

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/chatdelta/chatdelta-go"
)

func main() {
    // Create a client (uses OPENAI_API_KEY from environment)
    client, err := chatdelta.CreateClient("openai", "", "gpt-3.5-turbo", nil)
    if err != nil {
        log.Fatal(err)
    }
    
    // Send a prompt
    response, err := client.SendPrompt(context.Background(), "What is the capital of France?")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Response:", response)
}
```

### Advanced Configuration

```go
config := chatdelta.NewClientConfig().
    SetTimeout(60 * time.Second).      // 60 second timeout
    SetRetries(5).                     // 5 retry attempts
    SetTemperature(0.7).               // Creative temperature
    SetMaxTokens(2048).                // Response length limit
    SetTopP(0.9).                      // Nucleus sampling
    SetSystemMessage("You are a helpful AI assistant.")

client, err := chatdelta.CreateClient("claude", "your-api-key", "claude-3-haiku-20240307", config)
```

## Supported Providers

| Provider | Streaming | Conversations | Environment Variable |
|----------|-----------|---------------|---------------------|
| OpenAI   | ✅        | ✅            | `OPENAI_API_KEY` or `CHATGPT_API_KEY` |
| Claude   | ✅        | ✅            | `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` |
| Gemini   | ❌*       | ✅            | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

*Gemini streaming support coming soon

## Usage Examples

### Conversation Handling

```go
// Build a conversation
conversation := chatdelta.NewConversation()
conversation.AddSystemMessage("You are a helpful math tutor.")
conversation.AddUserMessage("What is 2 + 2?")
conversation.AddAssistantMessage("2 + 2 equals 4.")
conversation.AddUserMessage("What about 3 + 3?")

// Send the conversation
response, err := client.SendConversation(context.Background(), conversation)
if err != nil {
    log.Fatal(err)
}

fmt.Println("Response:", response)
```

### Streaming Responses

```go
if client.SupportsStreaming() {
    chunks, err := client.StreamPrompt(context.Background(), "Write a short poem about Go")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Print("Streaming response: ")
    for chunk := range chunks {
        fmt.Print(chunk.Content)
        if chunk.Finished {
            break
        }
    }
    fmt.Println()
}
```

### Parallel Execution

```go
// Create multiple clients
var clients []chatdelta.AIClient

if client1, err := chatdelta.CreateClient("openai", "", "", nil); err == nil {
    clients = append(clients, client1)
}
if client2, err := chatdelta.CreateClient("claude", "", "", nil); err == nil {
    clients = append(clients, client2)
}
if client3, err := chatdelta.CreateClient("gemini", "", "", nil); err == nil {
    clients = append(clients, client3)
}

// Execute same prompt across all providers
results := chatdelta.ExecuteParallel(context.Background(), clients, "What is the meaning of life?")

for _, result := range results {
    fmt.Printf("=== %s ===\n", result.ClientName)
    if result.Error != nil {
        fmt.Printf("Error: %v\n", result.Error)
    } else {
        fmt.Printf("Response: %s\n", result.Result)
    }
}
```

### Chat Sessions (NEW in v0.3.0)

```go
// Create a session with automatic history management
session := chatdelta.NewChatSessionWithSystemMessage(
    client, 
    "You are a helpful assistant.",
)

// Send messages - history is managed automatically
response1, err := session.Send(context.Background(), "What is Go?")
response2, err := session.Send(context.Background(), "What are its main features?")

// Get response with metadata
responseMeta, err := session.SendWithMetadata(context.Background(), "Compare it to Python")
fmt.Printf("Tokens used: %d\n", responseMeta.Metadata.TotalTokens)
fmt.Printf("Latency: %dms\n", responseMeta.Metadata.LatencyMs)
```

### Response Metadata (NEW in v0.3.0)

```go
// Get detailed metadata with responses
response, err := client.SendPromptWithMetadata(context.Background(), "Explain goroutines")
if err != nil {
    log.Fatal(err)
}

fmt.Println("Content:", response.Content)
fmt.Println("Model:", response.Metadata.ModelUsed)
fmt.Println("Prompt tokens:", response.Metadata.PromptTokens)
fmt.Println("Completion tokens:", response.Metadata.CompletionTokens)
fmt.Println("Total tokens:", response.Metadata.TotalTokens)
fmt.Printf("Latency: %dms\n", response.Metadata.LatencyMs)
```

### Custom Base URLs (NEW in v0.3.0)

```go
// Support for Azure OpenAI or local models
config := chatdelta.NewClientConfig().
    SetBaseURL("https://your-instance.openai.azure.com").
    SetRetryStrategy(chatdelta.RetryStrategyExponentialWithJitter)

client, err := chatdelta.CreateClient("openai", apiKey, "gpt-4", config)
```

### Error Handling

```go
response, err := client.SendPrompt(context.Background(), "Hello")
if err != nil {
    // Check specific error types
    if chatdelta.IsAuthenticationError(err) {
        fmt.Println("Authentication error - check your API key")
    } else if chatdelta.IsNetworkError(err) {
        fmt.Println("Network error - check your connection") 
    } else if chatdelta.IsRetryableError(err) {
        fmt.Println("Retryable error - the library automatically retries")
    } else {
        fmt.Printf("Other error: %v\n", err)
    }
}
```

### Check Available Providers

```go
available := chatdelta.GetAvailableProviders()
fmt.Printf("Available providers: %v\n", available)

for _, provider := range available {
    client, err := chatdelta.CreateClient(provider, "", "", nil)
    if err != nil {
        continue
    }
    
    info := chatdelta.GetClientInfo(client)
    fmt.Printf("%s: model=%s, streaming=%t, conversations=%t\n", 
        info.Name, info.Model, info.SupportsStreaming, info.SupportsConversations)
}
```

## API Reference

### Core Types

#### AIClient Interface

```go
type AIClient interface {
    // Send a single prompt
    SendPrompt(ctx context.Context, prompt string) (string, error)
    
    // Send a conversation with multiple messages
    SendConversation(ctx context.Context, conversation *Conversation) (string, error)
    
    // Stream a single prompt response
    StreamPrompt(ctx context.Context, prompt string) (<-chan StreamChunk, error)
    
    // Stream a conversation response  
    StreamConversation(ctx context.Context, conversation *Conversation) (<-chan StreamChunk, error)
    
    // Check if streaming is supported
    SupportsStreaming() bool
    
    // Check if conversations are supported
    SupportsConversations() bool
    
    // Get client name
    Name() string
    
    // Get model identifier
    Model() string
}
```

#### ClientConfig

```go
type ClientConfig struct {
    Timeout           time.Duration
    Retries           int
    Temperature       *float64  // 0.0 - 2.0
    MaxTokens         *int      // Max response tokens
    TopP              *float64  // 0.0 - 1.0 nucleus sampling
    FrequencyPenalty  *float64  // -2.0 - 2.0
    PresencePenalty   *float64  // -2.0 - 2.0  
    SystemMessage     *string   // System instruction
}
```

#### Conversation

```go
type Conversation struct {
    Messages []Message
}

func (c *Conversation) AddMessage(role, content string)
func (c *Conversation) AddSystemMessage(content string)
func (c *Conversation) AddUserMessage(content string)
func (c *Conversation) AddAssistantMessage(content string)
```

### Functions

#### Client Creation

```go
// Create a client for a specific provider
func CreateClient(provider, apiKey, model string, config *ClientConfig) (AIClient, error)

// Get providers with available API keys
func GetAvailableProviders() []string

// Get information about a client
func GetClientInfo(client AIClient) ClientInfo
```

#### Parallel Execution

```go
// Execute same prompt across multiple clients
func ExecuteParallel(ctx context.Context, clients []AIClient, prompt string) []ParallelResult

// Execute same conversation across multiple clients  
func ExecuteParallelConversation(ctx context.Context, clients []AIClient, conversation *Conversation) []ParallelResult
```

#### Error Helpers

```go
// Check if error is network-related
func IsNetworkError(err error) bool

// Check if error is retryable  
func IsRetryableError(err error) bool

// Check if error is authentication-related
func IsAuthenticationError(err error) bool
```

### Error Types

The library provides comprehensive error handling with specific error types:

- **NetworkError**: Timeouts, connection failures, DNS issues
- **APIError**: Rate limits, quota exceeded, invalid models, server errors  
- **AuthError**: Invalid API keys, expired tokens, permission issues
- **ConfigError**: Invalid parameters, missing configuration
- **ParseError**: JSON parsing failures, missing fields
- **StreamError**: Streaming-specific issues

## Environment Variables

Set these environment variables for automatic API key detection:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"
# or
export CHATGPT_API_KEY="your-openai-key"

# Anthropic Claude  
export ANTHROPIC_API_KEY="your-anthropic-key" 
# or
export CLAUDE_API_KEY="your-anthropic-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-key"
# or  
export GEMINI_API_KEY="your-google-key"
```

## Default Models

When no model is specified, these defaults are used:

- **OpenAI**: `gpt-3.5-turbo`
- **Claude**: `claude-3-haiku-20240307`  
- **Gemini**: `gemini-1.5-flash`

## Demo CLI

A demo CLI application is included to showcase the library:

```bash
# Build the demo
go build -o chatdelta-demo ./cmd/demo

# Run with default settings (OpenAI)
./chatdelta-demo

# Use a different provider
./chatdelta-demo -provider claude -prompt "Explain quantum computing"

# Enable streaming
./chatdelta-demo -stream -prompt "Write a haiku about programming"

# Run in parallel across all available providers
./chatdelta-demo -parallel -prompt "What is the meaning of life?"

# Customize parameters
./chatdelta-demo -provider gemini -temperature 0.9 -max-tokens 2048
```

## Testing

Run the test suite:

```bash
go test ./...
```

Run tests with coverage:

```bash
go test -v -race -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Continuous Integration

This repository includes a GitHub Actions workflow that automatically runs `go fmt`, `go vet`, and the test suite on every push and pull request.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the [chatdelta-rs](https://github.com/chatdelta/chatdelta-rs) Rust library
- Built with idiomatic Go practices and comprehensive error handling
- Designed for production use with retry logic and timeout handling

---

**Note**: This library focuses on defensive security applications only. It should not be used to create, modify, or improve code that may be used maliciously.
