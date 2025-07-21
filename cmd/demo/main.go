package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/chatdelta/chatdelta-go"
)

func main() {
	var (
		provider    = flag.String("provider", "openai", "AI provider (openai, claude, gemini)")
		model       = flag.String("model", "", "Model to use (defaults to provider default)")
		prompt      = flag.String("prompt", "Hello! How are you?", "Prompt to send")
		temperature = flag.Float64("temperature", 0.7, "Temperature parameter")
		maxTokens   = flag.Int("max-tokens", 1024, "Maximum tokens in response")
		stream      = flag.Bool("stream", false, "Use streaming response")
		parallel    = flag.Bool("parallel", false, "Execute on all available providers in parallel")
		timeout     = flag.Duration("timeout", 30*time.Second, "Request timeout")
	)
	flag.Parse()

	if *parallel {
		runParallel(*prompt, *timeout)
	} else {
		runSingle(*provider, *model, *prompt, *temperature, *maxTokens, *stream, *timeout)
	}
}

func runSingle(provider, model, prompt string, temperature float64, maxTokens int, useStreaming bool, timeout time.Duration) {
	// Create configuration
	config := chatdelta.NewClientConfig().
		SetTimeout(timeout).
		SetTemperature(temperature).
		SetMaxTokens(maxTokens).
		SetRetries(3)

	// Create client
	client, err := chatdelta.CreateClient(provider, "", model, config)
	if err != nil {
		log.Fatalf("Failed to create %s client: %v", provider, err)
	}

	fmt.Printf("Using %s with model %s\n", client.Name(), client.Model())
	fmt.Printf("Prompt: %s\n", prompt)
	fmt.Println("---")

	ctx := context.Background()

	if useStreaming && client.SupportsStreaming() {
		fmt.Println("Streaming response:")
		chunks, err := client.StreamPrompt(ctx, prompt)
		if err != nil {
			log.Fatalf("Failed to stream prompt: %v", err)
		}

		for chunk := range chunks {
			fmt.Print(chunk.Content)
			if chunk.Finished {
				break
			}
		}
		fmt.Println()
	} else {
		if useStreaming && !client.SupportsStreaming() {
			fmt.Printf("Note: %s doesn't support streaming, using regular response\n", client.Name())
		}
		
		response, err := client.SendPrompt(ctx, prompt)
		if err != nil {
			log.Fatalf("Failed to send prompt: %v", err)
		}

		fmt.Printf("Response: %s\n", response)
	}
}

func runParallel(prompt string, timeout time.Duration) {
	available := chatdelta.GetAvailableProviders()
	if len(available) == 0 {
		fmt.Println("No AI providers available.")
		fmt.Println("Set one of these environment variables:")
		fmt.Println("  OPENAI_API_KEY or CHATGPT_API_KEY")
		fmt.Println("  ANTHROPIC_API_KEY or CLAUDE_API_KEY")
		fmt.Println("  GOOGLE_API_KEY or GEMINI_API_KEY")
		os.Exit(1)
	}

	fmt.Printf("Available providers: %v\n", available)
	fmt.Printf("Prompt: %s\n", prompt)
	fmt.Println("---")

	// Create clients for all available providers
	var clients []chatdelta.AIClient
	for _, provider := range available {
		client, err := chatdelta.CreateClient(provider, "", "", nil)
		if err != nil {
			fmt.Printf("Warning: Failed to create %s client: %v\n", provider, err)
			continue
		}
		clients = append(clients, client)
	}

	if len(clients) == 0 {
		log.Fatal("No clients could be created")
	}

	// Execute in parallel
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	results := chatdelta.ExecuteParallel(ctx, clients, prompt)

	// Display results
	for _, result := range results {
		fmt.Printf("\n=== %s ===\n", result.ClientName)
		if result.Error != nil {
			fmt.Printf("Error: %v\n", result.Error)
		} else {
			fmt.Printf("Response: %s\n", result.Result)
		}
	}
}