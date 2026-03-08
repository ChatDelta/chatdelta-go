package chatdelta

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTextExporter_Interface(t *testing.T) {
	var _ MetricsExporter = (*TextExporter)(nil)
}

func TestTextExporter_Name(t *testing.T) {
	e := NewTextExporter()
	assert.Equal(t, "text", e.Name())
}

func TestTextExporter_Format(t *testing.T) {
	e := NewTextExporter()
	m := NewClientMetrics()
	m.RecordRequest(true, 100, 200)
	m.RecordCacheHit()
	m.RecordCacheMiss()

	output := e.Format(m.Snapshot())
	assert.Contains(t, output, "requests_total=1")
	assert.Contains(t, output, "requests_successful=1")
	assert.Contains(t, output, "requests_failed=0")
	assert.Contains(t, output, "cache_hits=1")
	assert.Contains(t, output, "cache_misses=1")
}

func TestTextExporter_Export(t *testing.T) {
	e := NewTextExporter()
	m := NewClientMetrics()
	m.RecordRequest(false, 0, 0)

	// Export should not return an error.
	err := e.Export(m.Snapshot())
	assert.NoError(t, err)
}

func TestInitTracing_Default(t *testing.T) {
	os.Unsetenv("CHATDELTA_LOG")
	level := InitTracing()
	assert.Equal(t, LogLevelInfo, level)
}

func TestInitTracing_ValidValues(t *testing.T) {
	cases := []struct {
		env   string
		level LogLevel
	}{
		{"error", LogLevelError},
		{"warn", LogLevelWarn},
		{"info", LogLevelInfo},
		{"debug", LogLevelDebug},
	}
	for _, tc := range cases {
		t.Run(tc.env, func(t *testing.T) {
			os.Setenv("CHATDELTA_LOG", tc.env)
			defer os.Unsetenv("CHATDELTA_LOG")
			assert.Equal(t, tc.level, InitTracing())
		})
	}
}

func TestInitTracing_InvalidFallsBackToInfo(t *testing.T) {
	os.Setenv("CHATDELTA_LOG", "verbose")
	defer os.Unsetenv("CHATDELTA_LOG")
	assert.Equal(t, LogLevelInfo, InitTracing())
}

func TestObservabilityContext(t *testing.T) {
	ctx := NewObservabilityContext("req-1", "openai", "gpt-4")
	assert.Equal(t, "req-1", ctx.RequestID)
	assert.Equal(t, "openai", ctx.Provider)
	assert.Equal(t, "gpt-4", ctx.Model)

	fields := ctx.LogFields()
	assert.Equal(t, "req-1", fields["request_id"])
	assert.Equal(t, "openai", fields["provider"])
	assert.Equal(t, "gpt-4", fields["model"])
}

func TestTextExporter_Format_ContainsAllFields(t *testing.T) {
	e := NewTextExporter()
	snap := MetricsSnapshot{
		RequestsTotal:      10,
		RequestsSuccessful: 8,
		RequestsFailed:     2,
		SuccessRate:        80.0,
		AvgLatencyMs:       150.5,
		TotalTokensUsed:    5000,
		CacheHits:          3,
		CacheMisses:        7,
		CacheHitRatio:      0.3,
	}
	out := e.Format(snap)
	for _, key := range []string{
		"requests_total=10",
		"success_rate=80.00%",
		"avg_latency_ms=150.50",
		"total_tokens=5000",
		"cache_hit_ratio=30.00%",
	} {
		assert.True(t, strings.Contains(out, key), "missing %q in %q", key, out)
	}
}
