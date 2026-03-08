// Package chatdelta provides a unified interface for interacting with multiple AI APIs.
// metrics.go implements thread-safe performance metrics collection for AI client
// interactions. Counters are backed by sync/atomic operations so ClientMetrics can
// be shared across goroutines without additional locking.
package chatdelta

import (
	"fmt"
	"sync/atomic"
	"time"
)

// ClientMetrics tracks cumulative performance statistics for an AI client.
// All fields are updated atomically; the zero value is ready to use.
type ClientMetrics struct {
	requestsTotal      atomic.Uint64
	requestsSuccessful atomic.Uint64
	requestsFailed     atomic.Uint64
	totalLatencyMs     atomic.Uint64
	totalTokensUsed    atomic.Uint64
	cacheHits          atomic.Uint64
	cacheMisses        atomic.Uint64
}

// NewClientMetrics creates a new, zeroed ClientMetrics.
func NewClientMetrics() *ClientMetrics {
	return &ClientMetrics{}
}

// RecordRequest records the outcome of a single request.
// latencyMs should be the wall-clock time in milliseconds; tokensUsed is the
// total token count (prompt + completion). Negative values are ignored.
func (m *ClientMetrics) RecordRequest(success bool, latencyMs int64, tokensUsed int) {
	m.requestsTotal.Add(1)
	if success {
		m.requestsSuccessful.Add(1)
	} else {
		m.requestsFailed.Add(1)
	}
	if latencyMs > 0 {
		m.totalLatencyMs.Add(uint64(latencyMs))
	}
	if tokensUsed > 0 {
		m.totalTokensUsed.Add(uint64(tokensUsed))
	}
}

// RecordCacheHit increments the cache-hit counter.
func (m *ClientMetrics) RecordCacheHit() {
	m.cacheHits.Add(1)
}

// RecordCacheMiss increments the cache-miss counter.
func (m *ClientMetrics) RecordCacheMiss() {
	m.cacheMisses.Add(1)
}

// Snapshot returns a consistent, serialisable point-in-time copy of the metrics.
// Derived statistics (success rate, average latency, cache hit ratio) are computed
// at snapshot time.
func (m *ClientMetrics) Snapshot() MetricsSnapshot {
	total := m.requestsTotal.Load()
	successful := m.requestsSuccessful.Load()
	failed := m.requestsFailed.Load()
	latencyMs := m.totalLatencyMs.Load()
	tokens := m.totalTokensUsed.Load()
	hits := m.cacheHits.Load()
	misses := m.cacheMisses.Load()

	var successRate float64
	if total > 0 {
		successRate = float64(successful) / float64(total) * 100
	}

	var avgLatencyMs float64
	if successful > 0 {
		avgLatencyMs = float64(latencyMs) / float64(successful)
	}

	var cacheHitRatio float64
	if total := hits + misses; total > 0 {
		cacheHitRatio = float64(hits) / float64(total)
	}

	return MetricsSnapshot{
		RequestsTotal:      total,
		RequestsSuccessful: successful,
		RequestsFailed:     failed,
		TotalLatencyMs:     latencyMs,
		TotalTokensUsed:    tokens,
		CacheHits:          hits,
		CacheMisses:        misses,
		SuccessRate:        successRate,
		AvgLatencyMs:       avgLatencyMs,
		CacheHitRatio:      cacheHitRatio,
	}
}

// Reset clears all counters to zero.
func (m *ClientMetrics) Reset() {
	m.requestsTotal.Store(0)
	m.requestsSuccessful.Store(0)
	m.requestsFailed.Store(0)
	m.totalLatencyMs.Store(0)
	m.totalTokensUsed.Store(0)
	m.cacheHits.Store(0)
	m.cacheMisses.Store(0)
}

// MetricsSnapshot is a serialisable, point-in-time view of ClientMetrics.
type MetricsSnapshot struct {
	// Raw counters.
	RequestsTotal      uint64 `json:"requests_total"`
	RequestsSuccessful uint64 `json:"requests_successful"`
	RequestsFailed     uint64 `json:"requests_failed"`
	TotalLatencyMs     uint64 `json:"total_latency_ms"`
	TotalTokensUsed    uint64 `json:"total_tokens_used"`
	CacheHits          uint64 `json:"cache_hits"`
	CacheMisses        uint64 `json:"cache_misses"`

	// Derived statistics computed at snapshot time.
	SuccessRate   float64 `json:"success_rate"`    // percentage (0–100)
	AvgLatencyMs  float64 `json:"avg_latency_ms"`  // per successful request
	CacheHitRatio float64 `json:"cache_hit_ratio"` // fraction (0–1)
}

// Summary returns a human-readable one-line summary of the snapshot.
func (s MetricsSnapshot) Summary() string {
	return fmt.Sprintf(
		"Requests: %d total, %d ok (%.1f%%), %d failed | avg latency: %.1fms | tokens: %d | cache: %d hits / %d misses (%.1f%%)",
		s.RequestsTotal,
		s.RequestsSuccessful,
		s.SuccessRate,
		s.RequestsFailed,
		s.AvgLatencyMs,
		s.TotalTokensUsed,
		s.CacheHits,
		s.CacheMisses,
		s.CacheHitRatio*100,
	)
}

// RequestTimer measures elapsed wall-clock time for a single operation.
type RequestTimer struct {
	start time.Time
}

// NewRequestTimer creates and immediately starts a RequestTimer.
func NewRequestTimer() *RequestTimer {
	return &RequestTimer{start: time.Now()}
}

// ElapsedMs returns milliseconds elapsed since the timer was started.
func (t *RequestTimer) ElapsedMs() int64 {
	return time.Since(t.start).Milliseconds()
}
