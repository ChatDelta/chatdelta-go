// Package chatdelta provides a unified interface for interacting with multiple AI APIs.
// mock.go implements a mock AIClient for use in unit tests.
// Responses are pre-loaded into a queue and dequeued in order; when the queue is
// exhausted a default fallback response is returned.
package chatdelta

import (
	"context"
	"fmt"
	"sync"
)

// MockResponse is a pre-configured response held in a MockClient's queue.
type MockResponse struct {
	// Content is the text to return on success.
	Content string
	// Error, if non-nil, is returned instead of Content.
	Error error
}

// MockClient implements AIClient using a pre-loaded response queue.
// It is safe for concurrent use.
type MockClient struct {
	mu        sync.Mutex
	name      string
	model     string
	responses []MockResponse
}

// NewMockClient creates a new MockClient with the given name and model.
// If model is empty it defaults to "mock-model".
func NewMockClient(name, model string) *MockClient {
	if model == "" {
		model = "mock-model"
	}
	return &MockClient{
		name:      name,
		model:     model,
		responses: make([]MockResponse, 0),
	}
}

// QueueResponse enqueues a successful text response.
func (m *MockClient) QueueResponse(content string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.responses = append(m.responses, MockResponse{Content: content})
}

// QueueError enqueues an error response.
func (m *MockClient) QueueError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.responses = append(m.responses, MockResponse{Error: err})
}

// dequeue pops the next response, or returns a generic fallback when the queue is empty.
func (m *MockClient) dequeue() MockResponse {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.responses) == 0 {
		return MockResponse{Content: fmt.Sprintf("mock response from %s", m.name)}
	}
	resp := m.responses[0]
	m.responses = m.responses[1:]
	return resp
}

// SendPrompt returns the next queued response.
func (m *MockClient) SendPrompt(_ context.Context, _ string) (string, error) {
	resp := m.dequeue()
	return resp.Content, resp.Error
}

// SendPromptWithMetadata returns the next queued response with basic metadata.
func (m *MockClient) SendPromptWithMetadata(_ context.Context, _ string) (*AiResponse, error) {
	resp := m.dequeue()
	if resp.Error != nil {
		return nil, resp.Error
	}
	return &AiResponse{
		Content:  resp.Content,
		Metadata: ResponseMetadata{ModelUsed: m.model},
	}, nil
}

// SendConversation returns the next queued response.
func (m *MockClient) SendConversation(_ context.Context, _ *Conversation) (string, error) {
	resp := m.dequeue()
	return resp.Content, resp.Error
}

// SendConversationWithMetadata returns the next queued response with basic metadata.
func (m *MockClient) SendConversationWithMetadata(_ context.Context, _ *Conversation) (*AiResponse, error) {
	resp := m.dequeue()
	if resp.Error != nil {
		return nil, resp.Error
	}
	return &AiResponse{
		Content:  resp.Content,
		Metadata: ResponseMetadata{ModelUsed: m.model},
	}, nil
}

// StreamPrompt dequeues a response and delivers it as a two-chunk stream.
// If the dequeued item is an error it is returned immediately.
func (m *MockClient) StreamPrompt(_ context.Context, _ string) (<-chan StreamChunk, error) {
	resp := m.dequeue()
	if resp.Error != nil {
		return nil, resp.Error
	}
	ch := make(chan StreamChunk, 2)
	go func() {
		defer close(ch)
		ch <- StreamChunk{Content: resp.Content, Finished: false}
		ch <- StreamChunk{Content: "", Finished: true}
	}()
	return ch, nil
}

// StreamConversation delegates to StreamPrompt.
func (m *MockClient) StreamConversation(ctx context.Context, _ *Conversation) (<-chan StreamChunk, error) {
	return m.StreamPrompt(ctx, "")
}

// SupportsStreaming returns true.
func (m *MockClient) SupportsStreaming() bool { return true }

// SupportsConversations returns true.
func (m *MockClient) SupportsConversations() bool { return true }

// Name returns the client name.
func (m *MockClient) Name() string { return m.name }

// Model returns the model identifier.
func (m *MockClient) Model() string { return m.model }
