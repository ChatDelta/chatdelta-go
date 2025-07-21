package chatdelta

import (
	"os"
	"strings"
)

// SupportedProviders lists all supported AI providers
var SupportedProviders = []string{"openai", "anthropic", "claude", "google", "gemini"}

// CreateClient creates a new AI client based on the provider string
func CreateClient(provider, apiKey, model string, config *ClientConfig) (AIClient, error) {
	if config == nil {
		config = NewClientConfig()
	}
	
	if err := ValidateConfig(config); err != nil {
		return nil, err
	}
	
	// Normalize provider name
	provider = strings.ToLower(strings.TrimSpace(provider))
	
	// If no API key provided, try to get from environment
	if apiKey == "" {
		apiKey = getAPIKeyFromEnv(provider)
	}
	
	if apiKey == "" {
		return nil, NewMissingConfigError("API key for provider: " + provider)
	}
	
	// If no model provided, use default
	if model == "" {
		model = getDefaultModel(provider)
	}
	
	switch provider {
	case "openai":
		return NewOpenAIClient(apiKey, model, config)
	case "anthropic", "claude":
		return NewClaudeClient(apiKey, model, config)
	case "google", "gemini":
		return NewGeminiClient(apiKey, model, config)
	default:
		return nil, NewInvalidParameterError("provider", provider)
	}
}

// getAPIKeyFromEnv retrieves the API key from environment variables
func getAPIKeyFromEnv(provider string) string {
	switch provider {
	case "openai":
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			return key
		}
		return os.Getenv("CHATGPT_API_KEY")
	case "anthropic", "claude":
		if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
			return key
		}
		return os.Getenv("CLAUDE_API_KEY")
	case "google", "gemini":
		if key := os.Getenv("GOOGLE_API_KEY"); key != "" {
			return key
		}
		return os.Getenv("GEMINI_API_KEY")
	default:
		return ""
	}
}

// getDefaultModel returns the default model for a provider
func getDefaultModel(provider string) string {
	switch provider {
	case "openai":
		return "gpt-3.5-turbo"
	case "anthropic", "claude":
		return "claude-3-haiku-20240307"
	case "google", "gemini":
		return "gemini-1.5-flash"
	default:
		return ""
	}
}

// GetAvailableProviders returns a list of providers with available API keys
func GetAvailableProviders() []string {
	var available []string
	
	for _, provider := range SupportedProviders {
		if getAPIKeyFromEnv(provider) != "" {
			available = append(available, provider)
		}
	}
	
	return available
}

// ClientInfo holds information about a client
type ClientInfo struct {
	Name                   string `json:"name"`
	Model                  string `json:"model"`
	SupportsStreaming      bool   `json:"supports_streaming"`
	SupportsConversations  bool   `json:"supports_conversations"`
}

// GetClientInfo returns information about a client
func GetClientInfo(client AIClient) ClientInfo {
	return ClientInfo{
		Name:                   client.Name(),
		Model:                  client.Model(),
		SupportsStreaming:      client.SupportsStreaming(),
		SupportsConversations:  client.SupportsConversations(),
	}
}