// Package chatdelta provides a unified interface for interacting with multiple AI APIs.
// middleware.go provides composable request-interceptor middleware for AIClient, plus
// JSON response validation and API-error extraction helpers shared across provider
// implementations.
//
// The Middleware / MiddlewareClient types model the same interceptor chain pattern as
// the Rust MiddlewareClient, adapted to Go idioms: instead of async traits, each
// Middleware is a plain function that calls next() to continue the chain. The
// lower-level retry loop from utils.go is still available; MiddlewareClient adds an
// orthogonal, higher-level interception layer (logging, per-call timeouts, etc.).
package chatdelta

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// ---------------------------------------------------------------------------
// Middleware chain types
// ---------------------------------------------------------------------------

// Middleware is an interceptor around a prompt-based AIClient call.
//
// ctx is the request context.
// prompt is the current prompt string (may have been modified by an earlier interceptor).
// next invokes the next middleware in the chain, or the inner AIClient when at the end.
//
// A Middleware can:
//   - Inspect or transform prompt before calling next.
//   - Inspect or transform the response returned by next.
//   - Short-circuit by returning an error without calling next.
//   - Add retry, logging, timeout, or other cross-cutting behaviour.
type Middleware func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error)

// MiddlewareClient wraps any AIClient with an ordered chain of Middleware interceptors.
// Execution order is middleware[0] → middleware[1] → … → inner client.
//
// For SendPrompt, the full chain is applied.
// For SendConversation, the last user message in the conversation is extracted as the
// "prompt" passed to middleware. The inner AIClient always receives the unmodified
// Conversation, so prompt-mutating middleware affects the returned string but not the
// conversation history. Logging and validation middleware work transparently for both.
// For streaming methods the middleware chain is bypassed and calls are forwarded to
// the inner client directly; use StreamToChannel to post-process chunks.
type MiddlewareClient struct {
	inner AIClient
	chain []Middleware
}

// NewMiddlewareClient creates a MiddlewareClient wrapping inner with the given
// middleware chain. Additional middleware can be appended later via Use.
func NewMiddlewareClient(inner AIClient, mw ...Middleware) *MiddlewareClient {
	return &MiddlewareClient{inner: inner, chain: append([]Middleware(nil), mw...)}
}

// Use appends one or more middleware to the end of the chain.
func (m *MiddlewareClient) Use(mw ...Middleware) {
	m.chain = append(m.chain, mw...)
}

// buildChain constructs the composed handler from the middleware chain and the
// provided innermost base function. Middleware is applied outermost-first so that
// middleware[0] runs first and middleware[last] is closest to the inner client.
func (m *MiddlewareClient) buildChain(base func(context.Context, string) (string, error)) func(context.Context, string) (string, error) {
	h := base
	for i := len(m.chain) - 1; i >= 0; i-- {
		mw := m.chain[i]
		next := h
		h = func(ctx context.Context, prompt string) (string, error) {
			return mw(ctx, prompt, next)
		}
	}
	return h
}

// lastUserMessage extracts the content of the last message with role "user" from conv,
// or an empty string when no user message is present.
func lastUserMessage(conv *Conversation) string {
	for i := len(conv.Messages) - 1; i >= 0; i-- {
		if conv.Messages[i].Role == "user" {
			return conv.Messages[i].Content
		}
	}
	return ""
}

// ---------------------------------------------------------------------------
// AIClient interface implementation
// ---------------------------------------------------------------------------

// SendPrompt applies the middleware chain then delegates to the inner client.
func (m *MiddlewareClient) SendPrompt(ctx context.Context, prompt string) (string, error) {
	return m.buildChain(m.inner.SendPrompt)(ctx, prompt)
}

// SendPromptWithMetadata applies the middleware chain and wraps the result with metadata.
func (m *MiddlewareClient) SendPromptWithMetadata(ctx context.Context, prompt string) (*AiResponse, error) {
	chain := m.buildChain(func(ctx context.Context, p string) (string, error) {
		resp, err := m.inner.SendPromptWithMetadata(ctx, p)
		if err != nil {
			return "", err
		}
		return resp.Content, nil
	})
	content, err := chain(ctx, prompt)
	if err != nil {
		return nil, err
	}
	return &AiResponse{Content: content, Metadata: ResponseMetadata{ModelUsed: m.inner.Model()}}, nil
}

// SendConversation applies the middleware chain using the last user message as the
// interception point, then delegates the full conversation to the inner client.
func (m *MiddlewareClient) SendConversation(ctx context.Context, conv *Conversation) (string, error) {
	proxy := lastUserMessage(conv)
	return m.buildChain(func(ctx context.Context, _ string) (string, error) {
		return m.inner.SendConversation(ctx, conv)
	})(ctx, proxy)
}

// SendConversationWithMetadata applies the middleware chain then delegates to the inner client.
func (m *MiddlewareClient) SendConversationWithMetadata(ctx context.Context, conv *Conversation) (*AiResponse, error) {
	proxy := lastUserMessage(conv)
	chain := m.buildChain(func(ctx context.Context, _ string) (string, error) {
		resp, err := m.inner.SendConversationWithMetadata(ctx, conv)
		if err != nil {
			return "", err
		}
		return resp.Content, nil
	})
	content, err := chain(ctx, proxy)
	if err != nil {
		return nil, err
	}
	return &AiResponse{Content: content, Metadata: ResponseMetadata{ModelUsed: m.inner.Model()}}, nil
}

// StreamPrompt forwards directly to the inner client. Use StreamToChannel to post-process.
func (m *MiddlewareClient) StreamPrompt(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	return m.inner.StreamPrompt(ctx, prompt)
}

// StreamConversation forwards directly to the inner client. Use StreamToChannel to post-process.
func (m *MiddlewareClient) StreamConversation(ctx context.Context, conv *Conversation) (<-chan StreamChunk, error) {
	return m.inner.StreamConversation(ctx, conv)
}

// SupportsStreaming delegates to the inner client.
func (m *MiddlewareClient) SupportsStreaming() bool { return m.inner.SupportsStreaming() }

// SupportsConversations delegates to the inner client.
func (m *MiddlewareClient) SupportsConversations() bool { return m.inner.SupportsConversations() }

// Name delegates to the inner client.
func (m *MiddlewareClient) Name() string { return m.inner.Name() }

// Model delegates to the inner client.
func (m *MiddlewareClient) Model() string { return m.inner.Model() }

// ---------------------------------------------------------------------------
// Built-in Middleware factories
// ---------------------------------------------------------------------------

// LoggingMiddleware returns a Middleware that logs each request prompt and the
// corresponding response (or error) using logger. Pass nil to use the default
// stdlib logger writing to stderr.
func LoggingMiddleware(logger *log.Logger) Middleware {
	if logger == nil {
		logger = log.Default()
	}
	return func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
		logger.Printf("[chatdelta] request prompt=%q", truncate(prompt, 80))
		resp, err := next(ctx, prompt)
		if err != nil {
			logger.Printf("[chatdelta] response error=%v", err)
		} else {
			logger.Printf("[chatdelta] response content=%q", truncate(resp, 80))
		}
		return resp, err
	}
}

// RetryMiddleware returns a Middleware that retries retryable errors up to
// maxRetries additional times with exponential backoff. This is the interceptor-level
// complement to utils.go's ExecuteWithRetry loop: use this when you want retries
// applied by a composable layer rather than baked into a single operation.
func RetryMiddleware(maxRetries int) Middleware {
	return func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
		var lastErr error
		for attempt := 0; attempt <= maxRetries; attempt++ {
			result, err := next(ctx, prompt)
			if err == nil {
				return result, nil
			}
			lastErr = err
			if !IsRetryableError(err) || attempt == maxRetries {
				return "", err
			}
			delay := time.Duration(1<<uint(attempt)) * 100 * time.Millisecond
			if delay > 10*time.Second {
				delay = 10 * time.Second
			}
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(delay):
			}
		}
		return "", lastErr
	}
}

// TimeoutMiddleware returns a Middleware that enforces a per-call timeout by
// deriving a child context with the given deadline. If the inner call exceeds
// the deadline the context's error is returned.
func TimeoutMiddleware(timeout time.Duration) Middleware {
	return func(ctx context.Context, prompt string, next func(context.Context, string) (string, error)) (string, error) {
		ctx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		return next(ctx, prompt)
	}
}

// truncate returns s truncated to maxLen characters, appending "…" when truncated.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "…"
}

// ---------------------------------------------------------------------------
// StreamToChannel
// ---------------------------------------------------------------------------

// StreamToChannel reads chunks from src, applies transform to each one, and
// forwards the result to the returned channel. The returned channel is closed when
// src closes or a Finished chunk is forwarded, or when ctx is cancelled.
//
// This is the Go equivalent of the Rust stream_to_channel utility: it converts one
// channel into another, allowing callers to post-process streaming chunks without
// consuming the source directly.
//
// Pass nil transform for a transparent passthrough.
func StreamToChannel(ctx context.Context, src <-chan StreamChunk, transform func(StreamChunk) StreamChunk) <-chan StreamChunk {
	if transform == nil {
		transform = func(c StreamChunk) StreamChunk { return c }
	}
	out := make(chan StreamChunk, 10)
	go func() {
		defer close(out)
		for {
			select {
			case <-ctx.Done():
				out <- StreamChunk{Content: "", Finished: true}
				return
			case chunk, ok := <-src:
				if !ok {
					return
				}
				out <- transform(chunk)
				if chunk.Finished {
					return
				}
			}
		}
	}()
	return out
}

// ---------------------------------------------------------------------------
// JSON validation helpers (used by provider client implementations)
// ---------------------------------------------------------------------------

// validateJSONResponse checks that every required field name is present as a
// top-level key in the JSON object encoded in data before full deserialization
// is attempted. It returns NewMissingFieldError for the first absent field, or
// NewJSONParseError if data is not valid JSON.
func validateJSONResponse(data []byte, required ...string) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return NewJSONParseError(err)
	}
	for _, field := range required {
		if _, ok := raw[field]; !ok {
			return NewMissingFieldError(field)
		}
	}
	return nil
}

// extractAPIError attempts to extract a human-readable error message from a
// JSON response body. It recognises several common shapes used by AI provider APIs:
//
//   - {"error": {"message": "…", "code": "…"}}  (OpenAI, Claude)
//   - {"message": "…"}
//   - {"detail": "…"}                            (some REST conventions)
//
// When no recognised shape is matched the raw body bytes are returned as-is.
func extractAPIError(data []byte) string {
	// Shape: {"error": {"message": "…"[, "code": "…"]}}
	var wrapper struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &wrapper); err == nil && wrapper.Error.Message != "" {
		if wrapper.Error.Code != "" {
			return fmt.Sprintf("%s: %s", wrapper.Error.Code, wrapper.Error.Message)
		}
		return wrapper.Error.Message
	}

	// Shape: {"message": "…"}
	var simple struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(data, &simple); err == nil && simple.Message != "" {
		return simple.Message
	}

	// Shape: {"detail": "…"}
	var detail struct {
		Detail string `json:"detail"`
	}
	if err := json.Unmarshal(data, &detail); err == nil && detail.Detail != "" {
		return detail.Detail
	}

	return string(data)
}
