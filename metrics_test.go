package chatdelta

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNewClientMetrics_ZeroValue(t *testing.T) {
	m := NewClientMetrics()
	s := m.Snapshot()
	assert.Equal(t, uint64(0), s.RequestsTotal)
	assert.Equal(t, float64(0), s.SuccessRate)
	assert.Equal(t, float64(0), s.AvgLatencyMs)
	assert.Equal(t, float64(0), s.CacheHitRatio)
}

func TestClientMetrics_RecordSuccessfulRequests(t *testing.T) {
	m := NewClientMetrics()
	m.RecordRequest(true, 100, 500)
	m.RecordRequest(true, 200, 300)

	s := m.Snapshot()
	assert.Equal(t, uint64(2), s.RequestsTotal)
	assert.Equal(t, uint64(2), s.RequestsSuccessful)
	assert.Equal(t, uint64(0), s.RequestsFailed)
	assert.Equal(t, float64(100), s.SuccessRate)
	assert.Equal(t, float64(150), s.AvgLatencyMs) // (100+200)/2
	assert.Equal(t, uint64(800), s.TotalTokensUsed)
}

func TestClientMetrics_RecordFailedRequests(t *testing.T) {
	m := NewClientMetrics()
	m.RecordRequest(true, 50, 100)
	m.RecordRequest(false, 0, 0)

	s := m.Snapshot()
	assert.Equal(t, uint64(2), s.RequestsTotal)
	assert.Equal(t, uint64(1), s.RequestsSuccessful)
	assert.Equal(t, uint64(1), s.RequestsFailed)
	assert.Equal(t, float64(50), s.SuccessRate)
}

func TestClientMetrics_CacheCounters(t *testing.T) {
	m := NewClientMetrics()
	m.RecordCacheHit()
	m.RecordCacheHit()
	m.RecordCacheMiss()

	s := m.Snapshot()
	assert.Equal(t, uint64(2), s.CacheHits)
	assert.Equal(t, uint64(1), s.CacheMisses)
	assert.InDelta(t, 0.667, s.CacheHitRatio, 0.001)
}

func TestClientMetrics_Reset(t *testing.T) {
	m := NewClientMetrics()
	m.RecordRequest(true, 100, 200)
	m.RecordCacheHit()
	m.Reset()

	s := m.Snapshot()
	assert.Equal(t, uint64(0), s.RequestsTotal)
	assert.Equal(t, uint64(0), s.CacheHits)
}

func TestMetricsSnapshot_Summary(t *testing.T) {
	m := NewClientMetrics()
	m.RecordRequest(true, 42, 100)
	s := m.Snapshot()
	summary := s.Summary()
	assert.Contains(t, summary, "1 total")
	assert.Contains(t, summary, "42.0ms")
}

func TestRequestTimer(t *testing.T) {
	timer := NewRequestTimer()
	time.Sleep(5 * time.Millisecond)
	elapsed := timer.ElapsedMs()
	assert.GreaterOrEqual(t, elapsed, int64(1))
}
