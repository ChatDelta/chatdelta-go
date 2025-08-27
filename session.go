package chatdelta

import (
	"context"
)

// ChatSession manages multi-turn conversations with an AI client
// Ported from chatdelta-rs/src/lib.rs:341-428
type ChatSession struct {
	client       AIClient
	conversation *Conversation
}

// NewChatSession creates a new chat session with the given client
func NewChatSession(client AIClient) *ChatSession {
	return &ChatSession{
		client:       client,
		conversation: NewConversation(),
	}
}

// NewChatSessionWithSystemMessage creates a new chat session with a system message
func NewChatSessionWithSystemMessage(client AIClient, message string) *ChatSession {
	session := &ChatSession{
		client:       client,
		conversation: NewConversation(),
	}
	session.conversation.AddSystemMessage(message)
	return session
}

// Send sends a message and gets a response
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

// SendWithMetadata sends a message and gets a response with metadata
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

// Stream sends a message and returns a channel for streaming chunks
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

// AddMessage adds a message to the conversation without sending
func (s *ChatSession) AddMessage(message Message) {
	s.conversation.Messages = append(s.conversation.Messages, message)
}

// History returns the conversation history
func (s *ChatSession) History() *Conversation {
	return s.conversation
}

// Clear clears the conversation history
func (s *ChatSession) Clear() {
	s.conversation.Messages = make([]Message, 0)
}

// ResetWithSystem resets the session with a new system message
func (s *ChatSession) ResetWithSystem(message string) {
	s.conversation = NewConversation()
	s.conversation.AddSystemMessage(message)
}

// Len returns the number of messages in the conversation
func (s *ChatSession) Len() int {
	return len(s.conversation.Messages)
}

// IsEmpty checks if the conversation is empty
func (s *ChatSession) IsEmpty() bool {
	return len(s.conversation.Messages) == 0
}