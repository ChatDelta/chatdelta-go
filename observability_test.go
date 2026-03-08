package chatdelta

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

// ---------------------------------------------------------------------------
// PrometheusExporter tests
// ---------------------------------------------------------------------------

func TestPrometheusExporter_Interface(t *testing.T) {
	var _ MetricsExporter = (*PrometheusExporter)(nil)
}

func TestPrometheusExporter_Name(t *testing.T) {
	e := NewPrometheusExporter()
	assert.Equal(t, "prometheus", e.Name())
}

func TestPrometheusExporter_Export_NoError(t *testing.T) {
	e := NewPrometheusExporter()
	m := NewClientMetrics()
	m.RecordRequest(true, 150, 300)
	m.RecordCacheHit()
	m.RecordCacheMiss()

	err := e.Export(m.Snapshot())
	assert.NoError(t, err)
}

func TestPrometheusExporter_Registry_NotNil(t *testing.T) {
	e := NewPrometheusExporter()
	assert.NotNil(t, e.Registry())
}

func scrapeMetrics(t *testing.T, e *PrometheusExporter) string {
	t.Helper()
	srv := httptest.NewServer(e.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return string(body)
}

func TestPrometheusExporter_Handler_ServesMetrics(t *testing.T) {
	e := NewPrometheusExporter()
	m := NewClientMetrics()
	m.RecordRequest(true, 100, 50)
	require.NoError(t, e.Export(m.Snapshot()))

	body := scrapeMetrics(t, e)
	assert.Contains(t, body, "chatdelta_requests_total")
}

func TestPrometheusExporter_MetricNames(t *testing.T) {
	e := NewPrometheusExporter()
	require.NoError(t, e.Export(MetricsSnapshot{}))

	body := scrapeMetrics(t, e)
	for _, name := range []string{
		"chatdelta_requests_total",
		"chatdelta_requests_successful_total",
		"chatdelta_requests_failed_total",
		"chatdelta_request_duration_ms",
		"chatdelta_tokens_total",
		"chatdelta_cache_hits_total",
		"chatdelta_cache_misses_total",
	} {
		assert.Contains(t, body, name, "metric %q missing from output", name)
	}
}

func TestPrometheusExporter_ValuesReflectSnapshot(t *testing.T) {
	e := NewPrometheusExporter()
	m := NewClientMetrics()
	m.RecordRequest(true, 200, 1000)
	m.RecordRequest(false, 0, 0)
	m.RecordCacheHit()
	m.RecordCacheHit()
	m.RecordCacheMiss()
	require.NoError(t, e.Export(m.Snapshot()))

	body := scrapeMetrics(t, e)
	// requests_total should be 2, successful 1, failed 1, cache_hits 2, cache_misses 1.
	assert.Contains(t, body, "chatdelta_requests_total 2")
	assert.Contains(t, body, "chatdelta_requests_successful_total 1")
	assert.Contains(t, body, "chatdelta_requests_failed_total 1")
	assert.Contains(t, body, "chatdelta_cache_hits_total 2")
	assert.Contains(t, body, "chatdelta_cache_misses_total 1")
}

func TestPrometheusExporter_UpdatedOnSubsequentExport(t *testing.T) {
	e := NewPrometheusExporter()

	m := NewClientMetrics()
	m.RecordRequest(true, 50, 100)
	require.NoError(t, e.Export(m.Snapshot()))

	// Record a second request and export again.
	m.RecordRequest(true, 80, 200)
	require.NoError(t, e.Export(m.Snapshot()))

	body := scrapeMetrics(t, e)
	assert.Contains(t, body, "chatdelta_requests_total 2")
}

func TestPrometheusExporter_IsolatedRegistry(t *testing.T) {
	// Two exporters must not share state (each has its own registry).
	e1 := NewPrometheusExporter()
	e2 := NewPrometheusExporter()

	m1 := NewClientMetrics()
	m1.RecordRequest(true, 10, 0)
	require.NoError(t, e1.Export(m1.Snapshot()))

	// e2 has seen no requests.
	body2 := scrapeMetrics(t, e2)
	assert.Contains(t, body2, "chatdelta_requests_total 0")

	body1 := scrapeMetrics(t, e1)
	assert.Contains(t, body1, "chatdelta_requests_total 1")
}
