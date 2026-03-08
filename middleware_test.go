package chatdelta

import (
	"context"
	"errors"
	"log"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateJSONResponse_Valid(t *testing.T) {
	data := []byte(`{"choices": [], "model": "gpt-4"}`)
	err := validateJSONResponse(data, "choices", "model")
	assert.NoError(t, err)
}

func TestValidateJSONResponse_MissingField(t *testing.T) {
	data := []byte(`{"model": "gpt-4"}`)
	err := validateJSONResponse(data, "choices")
	assert.Error(t, err)
	ce, ok := err.(*ClientError)
	assert.True(t, ok)
	assert.Equal(t, "missing_field", ce.Code)
}

func TestValidateJSONResponse_InvalidJSON(t *testing.T) {
	data := []byte(`not json`)
	err := validateJSONResponse(data, "field")
	assert.Error(t, err)
	ce, ok := err.(*ClientError)
	assert.True(t, ok)
	assert.Equal(t, ErrorTypeParse, ce.Type)
}

func TestExtractAPIError_OpenAIShape(t *testing.T) {
	data := []byte(`{"error":{"message":"invalid api key","code":"invalid_api_key"}}`)
	msg := extractAPIError(data)
	assert.Equal(t, "invalid_api_key: invalid api key", msg)
}

func TestExtractAPIError_MessageShape(t *testing.T) {
	data := []byte(`{"message":"resource not found"}`)
	msg := extractAPIError(data)
	assert.Equal(t, "resource not found", msg)
}

func TestExtractAPIError_DetailShape(t *testing.T) {
	data := []byte(`{"detail":"validation error"}`)
	msg := extractAPIError(data)
	assert.Equal(t, "validation error", msg)
}

func TestExtractAPIError_FallbackRawBody(t *testing.T) {
	data := []byte(`unexpected plain text error`)
	msg := extractAPIError(data)
	assert.Equal(t, "unexpected plain text error", msg)
}

func TestExtractAPIError_ErrorWithoutCode(t *testing.T) {
	data := []byte(`{"error":{"message":"rate limit exceeded"}}`)
	msg := extractAPIError(data)
	assert.Equal(t, "rate limit exceeded", msg)
}

// ---------------------------------------------------------------------------
// MiddlewareClient tests
// ---------------------------------------------------------------------------

func TestMiddlewareClient_Interface(t *testing.T) {
	var _ AIClient = (*MiddlewareClient)(nil)
}

func TestMiddlewareClient_Passthrough(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("hello")
	mc := NewMiddlewareClient(inner)

	resp, err := mc.SendPrompt(context.Background(), "hi")
	require.NoError(t, err)
	assert.Equal(t, "hello", resp)
}

func TestMiddlewareClient_DelegatesMetadata(t *testing.T) {
	inner := NewMockClient("test", "mymodel")
	inner.QueueResponse("response")
	mc := NewMiddlewareClient(inner)

	resp, err := mc.SendPromptWithMetadata(context.Background(), "hi")
	require.NoError(t, err)
	assert.Equal(t, "response", resp.Content)
	assert.Equal(t, "mymodel", resp.Metadata.ModelUsed)
}

func TestMiddlewareClient_SingleMiddleware(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("original")

	var intercepted string
	mw := Middleware(func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
		intercepted = prompt
		return next(ctx, prompt)
	})
	mc := NewMiddlewareClient(inner, mw)

	_, err := mc.SendPrompt(context.Background(), "my prompt")
	require.NoError(t, err)
	assert.Equal(t, "my prompt", intercepted)
}

func TestMiddlewareClient_ChainOrder(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("ok")

	var order []int
	makeMW := func(id int) Middleware {
		return func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
			order = append(order, id)
			return next(ctx, prompt)
		}
	}

	mc := NewMiddlewareClient(inner, makeMW(1), makeMW(2), makeMW(3))
	_, err := mc.SendPrompt(context.Background(), "q")
	require.NoError(t, err)
	assert.Equal(t, []int{1, 2, 3}, order)
}

func TestMiddlewareClient_Use(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("ok")

	var seen []string
	mc := NewMiddlewareClient(inner)
	mc.Use(func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
		seen = append(seen, "added")
		return next(ctx, prompt)
	})

	_, err := mc.SendPrompt(context.Background(), "q")
	require.NoError(t, err)
	assert.Equal(t, []string{"added"}, seen)
}

func TestMiddlewareClient_MiddlewareCanShortCircuit(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("should not reach")

	mc := NewMiddlewareClient(inner, func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
		return "shortcut", nil
	})

	resp, err := mc.SendPrompt(context.Background(), "q")
	require.NoError(t, err)
	assert.Equal(t, "shortcut", resp)
}

func TestMiddlewareClient_MiddlewareCanTransformPrompt(t *testing.T) {
	var received string
	inner := NewMockClient("test", "model")
	inner.QueueResponse("ok")

	mc := NewMiddlewareClient(inner,
		func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
			received = prompt
			return next(ctx, "TRANSFORMED: "+prompt)
		},
		func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
			received = prompt // will be the transformed value
			return next(ctx, prompt)
		},
	)
	_, _ = mc.SendPrompt(context.Background(), "original")
	assert.Equal(t, "TRANSFORMED: original", received)
}

func TestMiddlewareClient_SendConversation(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("conv response")

	var interceptedPrompt string
	mc := NewMiddlewareClient(inner, func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
		interceptedPrompt = prompt
		return next(ctx, prompt)
	})

	conv := NewConversation()
	conv.AddUserMessage("user question")

	resp, err := mc.SendConversation(context.Background(), conv)
	require.NoError(t, err)
	assert.Equal(t, "conv response", resp)
	assert.Equal(t, "user question", interceptedPrompt)
}

func TestMiddlewareClient_DelegatesCapabilities(t *testing.T) {
	inner := NewMockClient("MyMock", "mock-model")
	mc := NewMiddlewareClient(inner)
	assert.Equal(t, "MyMock", mc.Name())
	assert.Equal(t, "mock-model", mc.Model())
	assert.True(t, mc.SupportsStreaming())
	assert.True(t, mc.SupportsConversations())
}

func TestMiddlewareClient_StreamPrompt(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("streamed")
	mc := NewMiddlewareClient(inner)

	ch, err := mc.StreamPrompt(context.Background(), "q")
	require.NoError(t, err)

	var chunks []StreamChunk
	for c := range ch {
		chunks = append(chunks, c)
	}
	assert.NotEmpty(t, chunks)
}

// ---------------------------------------------------------------------------
// Built-in middleware tests
// ---------------------------------------------------------------------------

func TestLoggingMiddleware(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("logged response")

	var buf strings.Builder
	logger := log.New(&buf, "", 0)
	mc := NewMiddlewareClient(inner, LoggingMiddleware(logger))

	_, err := mc.SendPrompt(context.Background(), "my question")
	require.NoError(t, err)
	assert.Contains(t, buf.String(), "request")
	assert.Contains(t, buf.String(), "response")
}

func TestLoggingMiddleware_NilLogger(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("ok")
	mc := NewMiddlewareClient(inner, LoggingMiddleware(nil))
	// Should not panic.
	_, err := mc.SendPrompt(context.Background(), "q")
	require.NoError(t, err)
}

func TestRetryMiddleware_SuccessOnFirstTry(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("ok")
	mc := NewMiddlewareClient(inner, RetryMiddleware(3))

	resp, err := mc.SendPrompt(context.Background(), "q")
	require.NoError(t, err)
	assert.Equal(t, "ok", resp)
}

func TestRetryMiddleware_NonRetryableErrorNotRetried(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueError(NewInvalidAPIKeyError()) // not retryable
	mc := NewMiddlewareClient(inner, RetryMiddleware(3))

	_, err := mc.SendPrompt(context.Background(), "q")
	assert.Error(t, err)
}

func TestTimeoutMiddleware_TimesOut(t *testing.T) {
	inner := NewMockClient("test", "model")
	// Simulate a slow call by blocking until context cancels.
	mc := NewMiddlewareClient(inner,
		TimeoutMiddleware(10*time.Millisecond),
		func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		},
	)
	_, err := mc.SendPrompt(context.Background(), "q")
	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded))
}

func TestTimeoutMiddleware_PassesWithinDeadline(t *testing.T) {
	inner := NewMockClient("test", "model")
	inner.QueueResponse("fast")
	mc := NewMiddlewareClient(inner, TimeoutMiddleware(5*time.Second))

	resp, err := mc.SendPrompt(context.Background(), "q")
	require.NoError(t, err)
	assert.Equal(t, "fast", resp)
}

// ---------------------------------------------------------------------------
// StreamToChannel tests
// ---------------------------------------------------------------------------

func TestStreamToChannel_Passthrough(t *testing.T) {
	src := make(chan StreamChunk, 3)
	src <- StreamChunk{Content: "a"}
	src <- StreamChunk{Content: "b"}
	src <- StreamChunk{Content: "", Finished: true}
	close(src)

	out := StreamToChannel(context.Background(), src, nil)
	var got []StreamChunk
	for c := range out {
		got = append(got, c)
	}
	require.Len(t, got, 3)
	assert.Equal(t, "a", got[0].Content)
	assert.Equal(t, "b", got[1].Content)
	assert.True(t, got[2].Finished)
}

func TestStreamToChannel_Transform(t *testing.T) {
	src := make(chan StreamChunk, 2)
	src <- StreamChunk{Content: "hello"}
	src <- StreamChunk{Content: "", Finished: true}
	close(src)

	upper := func(c StreamChunk) StreamChunk {
		c.Content = strings.ToUpper(c.Content)
		return c
	}
	out := StreamToChannel(context.Background(), src, upper)

	var content string
	for c := range out {
		content += c.Content
	}
	assert.Equal(t, "HELLO", content)
}

func TestStreamToChannel_ContextCancel(t *testing.T) {
	src := make(chan StreamChunk) // never sends
	ctx, cancel := context.WithCancel(context.Background())
	out := StreamToChannel(ctx, src, nil)
	cancel()

	// Should receive a terminal Finished chunk and then close.
	chunk := <-out
	assert.True(t, chunk.Finished)
}

func TestTruncate(t *testing.T) {
	assert.Equal(t, "hello", truncate("hello", 10))
	assert.Equal(t, "hello…", truncate("hello world", 5))
}
