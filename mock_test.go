package chatdelta

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMockClient_Interface(t *testing.T) {
	// Compile-time check that MockClient satisfies AIClient.
	var _ AIClient = (*MockClient)(nil)
}

func TestMockClient_Defaults(t *testing.T) {
	m := NewMockClient("test", "")
	assert.Equal(t, "test", m.Name())
	assert.Equal(t, "mock-model", m.Model())
	assert.True(t, m.SupportsStreaming())
	assert.True(t, m.SupportsConversations())
}

func TestMockClient_QueueResponse(t *testing.T) {
	m := NewMockClient("test", "model")
	m.QueueResponse("hello")
	m.QueueResponse("world")

	r1, err := m.SendPrompt(context.Background(), "q")
	require.NoError(t, err)
	assert.Equal(t, "hello", r1)

	r2, err := m.SendPrompt(context.Background(), "q")
	require.NoError(t, err)
	assert.Equal(t, "world", r2)

	// Queue exhausted: fallback response
	r3, err := m.SendPrompt(context.Background(), "q")
	require.NoError(t, err)
	assert.Contains(t, r3, "mock response")
}

func TestMockClient_QueueError(t *testing.T) {
	m := NewMockClient("test", "model")
	m.QueueError(errors.New("boom"))

	_, err := m.SendPrompt(context.Background(), "q")
	assert.EqualError(t, err, "boom")
}

func TestMockClient_SendConversation(t *testing.T) {
	m := NewMockClient("test", "model")
	m.QueueResponse("conv response")

	conv := NewConversation()
	conv.AddUserMessage("hi")

	r, err := m.SendConversation(context.Background(), conv)
	require.NoError(t, err)
	assert.Equal(t, "conv response", r)
}

func TestMockClient_SendPromptWithMetadata(t *testing.T) {
	m := NewMockClient("test", "mymodel")
	m.QueueResponse("meta response")

	resp, err := m.SendPromptWithMetadata(context.Background(), "q")
	require.NoError(t, err)
	assert.Equal(t, "meta response", resp.Content)
	assert.Equal(t, "mymodel", resp.Metadata.ModelUsed)
}

func TestMockClient_SendPromptWithMetadata_Error(t *testing.T) {
	m := NewMockClient("test", "model")
	m.QueueError(errors.New("api error"))

	_, err := m.SendPromptWithMetadata(context.Background(), "q")
	assert.EqualError(t, err, "api error")
}

func TestMockClient_SendConversationWithMetadata(t *testing.T) {
	m := NewMockClient("test", "mymodel")
	m.QueueResponse("conv meta response")

	conv := NewConversation()
	conv.AddUserMessage("hi")

	resp, err := m.SendConversationWithMetadata(context.Background(), conv)
	require.NoError(t, err)
	assert.Equal(t, "conv meta response", resp.Content)
	assert.Equal(t, "mymodel", resp.Metadata.ModelUsed)
}

func TestMockClient_StreamPrompt(t *testing.T) {
	m := NewMockClient("test", "model")
	m.QueueResponse("streamed")

	ch, err := m.StreamPrompt(context.Background(), "q")
	require.NoError(t, err)

	var chunks []StreamChunk
	for c := range ch {
		chunks = append(chunks, c)
	}
	require.Len(t, chunks, 2)
	assert.Equal(t, "streamed", chunks[0].Content)
	assert.False(t, chunks[0].Finished)
	assert.True(t, chunks[1].Finished)
}

func TestMockClient_StreamPrompt_Error(t *testing.T) {
	m := NewMockClient("test", "model")
	m.QueueError(errors.New("stream error"))

	ch, err := m.StreamPrompt(context.Background(), "q")
	assert.EqualError(t, err, "stream error")
	assert.Nil(t, ch)
}
