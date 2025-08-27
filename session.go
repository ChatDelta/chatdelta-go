package chatdelta

import (
	"context"
)

// ChatSession manages multi-turn conversations with an AI client.
// It automatically maintains conversation history and handles context.
//
// Example:
//
//	session := NewChatSessionWithSystemMessage(client, "You are a helpful assistant.")
//	response1, err := session.Send(ctx, "What is Go?")
//	response2, err := session.Send(ctx, "What are its benefits?") // Remembers context
type ChatSession struct {
	client       AIClient
	conversation *Conversation
}

// NewChatSession creates a new chat session with the given client.
// The conversation starts empty with no system message.
func NewChatSession(client AIClient) *ChatSession {
	return &ChatSession{
		client:       client,
		conversation: NewConversation(),
	}
}

// NewChatSessionWithSystemMessage creates a new chat session with a system message.
// The system message sets the context and behavior for the AI assistant.
func NewChatSessionWithSystemMessage(client AIClient, message string) *ChatSession {
	session := &ChatSession{
		client:       client,
		conversation: NewConversation(),
	}
	session.conversation.AddSystemMessage(message)
	return session
}

// Send sends a message and gets a response.
// The message is added to the conversation history as a user message,
// and the response is added as an assistant message.
// If an error occurs, the user message is removed from history.
func (s *ChatSession) Send(ctx context.Context, message string) (string, error) {
	s.conversation.AddUserMessage(message)
	
	response, err := s.client.SendConversation(ctx, s.conversation)
	if err != nil {
		// Remove the user message if the request failed
		if len(s.conversation.Messages) > 0 {
			s.conversation.Messages = s.conversation.Messages[:len(s.conversation.Messages)-1]
		}
		return "", err
	}
	
	s.conversation.AddAssistantMessage(response)
	return response, nil
}

// SendWithMetadata sends a message and gets a response with metadata.
// This includes token counts, latency, and other provider-specific information.
// The conversation history is updated the same as Send.
func (s *ChatSession) SendWithMetadata(ctx context.Context, message string) (*AiResponse, error) {
	s.conversation.AddUserMessage(message)
	
	response, err := s.client.SendConversationWithMetadata(ctx, s.conversation)
	if err != nil {
		// Remove the user message if the request failed
		if len(s.conversation.Messages) > 0 {
			s.conversation.Messages = s.conversation.Messages[:len(s.conversation.Messages)-1]
		}
		return nil, err
	}
	
	s.conversation.AddAssistantMessage(response.Content)
	return response, nil
}

// Stream sends a message and returns a channel for streaming chunks.
// The complete response is assembled and added to history when streaming completes.
// The returned channel is buffered and will be closed when streaming ends.
func (s *ChatSession) Stream(ctx context.Context, message string) (<-chan StreamChunk, error) {
	s.conversation.AddUserMessage(message)
	
	chunks, err := s.client.StreamConversation(ctx, s.conversation)
	if err != nil {
		// Remove the user message if the request failed
		if len(s.conversation.Messages) > 0 {
			s.conversation.Messages = s.conversation.Messages[:len(s.conversation.Messages)-1]
		}
		return nil, err
	}
	
	// Create a wrapper channel to collect the full response
	wrapped := make(chan StreamChunk, 100)
	go func() {
		defer close(wrapped)
		var fullContent string
		for chunk := range chunks {
			fullContent += chunk.Content
			wrapped <- chunk
			if chunk.Finished {
				// Add the complete response to conversation
				s.conversation.AddAssistantMessage(fullContent)
			}
		}
	}()
	
	return wrapped, nil
}

// AddMessage adds a message to the conversation without sending it.
// Use this to manually construct conversation history.
func (s *ChatSession) AddMessage(message Message) {
	s.conversation.Messages = append(s.conversation.Messages, message)
}

// History returns the conversation history.
// The returned conversation can be modified directly if needed.
func (s *ChatSession) History() *Conversation {
	return s.conversation
}

// Clear removes all messages from the conversation history.
func (s *ChatSession) Clear() {
	s.conversation.Messages = make([]Message, 0)
}

// ResetWithSystem clears the conversation and sets a new system message.
// This is useful for changing the AI's behavior mid-session.
func (s *ChatSession) ResetWithSystem(message string) {
	s.conversation = NewConversation()
	s.conversation.AddSystemMessage(message)
}

// Len returns the number of messages in the conversation.
func (s *ChatSession) Len() int {
	return len(s.conversation.Messages)
}

// IsEmpty returns true if the conversation has no messages.
func (s *ChatSession) IsEmpty() bool {
	return len(s.conversation.Messages) == 0
}