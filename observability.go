// Package chatdelta provides a unified interface for interacting with multiple AI APIs.
// observability.go provides metrics export and structured logging infrastructure for
// monitoring AI client performance.
//
// Two MetricsExporter implementations are provided:
//   - TextExporter: always available, writes human-readable key=value lines to stderr.
//   - PrometheusExporter: exports metrics in Prometheus format via a dedicated registry.
//     It mirrors the seven metrics tracked by the Rust observability module.
package chatdelta

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// MetricsExporter is implemented by backends that receive periodic metrics snapshots.
type MetricsExporter interface {
	// Export sends the snapshot to the configured backend.
	// Implementations should be non-blocking where possible.
	Export(snapshot MetricsSnapshot) error
	// Name returns a short identifier for the exporter (e.g. "text", "prometheus").
	Name() string
}

// TextExporter writes human-readable metrics to a standard logger.
// It is always available and has no external dependencies.
type TextExporter struct {
	logger *log.Logger
}

// NewTextExporter creates a TextExporter that writes to stderr.
func NewTextExporter() *TextExporter {
	return &TextExporter{
		logger: log.New(os.Stderr, "[chatdelta] ", log.LstdFlags),
	}
}

// Export logs a formatted summary of the snapshot.
func (e *TextExporter) Export(snapshot MetricsSnapshot) error {
	e.logger.Println(e.Format(snapshot))
	return nil
}

// Name returns "text".
func (e *TextExporter) Name() string { return "text" }

// Format returns a key=value formatted metrics string without logging it.
func (e *TextExporter) Format(snapshot MetricsSnapshot) string {
	return fmt.Sprintf(
		"requests_total=%d requests_successful=%d requests_failed=%d "+
			"success_rate=%.2f%% avg_latency_ms=%.2f total_tokens=%d "+
			"cache_hits=%d cache_misses=%d cache_hit_ratio=%.2f%%",
		snapshot.RequestsTotal,
		snapshot.RequestsSuccessful,
		snapshot.RequestsFailed,
		snapshot.SuccessRate,
		snapshot.AvgLatencyMs,
		snapshot.TotalTokensUsed,
		snapshot.CacheHits,
		snapshot.CacheMisses,
		snapshot.CacheHitRatio*100,
	)
}

// LogLevel controls the verbosity of chatdelta's internal log output.
type LogLevel string

const (
	// LogLevelError only logs errors.
	LogLevelError LogLevel = "error"
	// LogLevelWarn logs warnings and errors.
	LogLevelWarn LogLevel = "warn"
	// LogLevelInfo logs informational messages and above.
	LogLevelInfo LogLevel = "info"
	// LogLevelDebug logs all messages including diagnostic details.
	LogLevelDebug LogLevel = "debug"
)

// InitTracing reads the CHATDELTA_LOG environment variable and returns the
// corresponding LogLevel. Valid values are "error", "warn", "info", "debug".
// Unrecognised or empty values default to LogLevelInfo.
func InitTracing() LogLevel {
	switch level := LogLevel(os.Getenv("CHATDELTA_LOG")); level {
	case LogLevelError, LogLevelWarn, LogLevelInfo, LogLevelDebug:
		return level
	default:
		return LogLevelInfo
	}
}

// ObservabilityContext associates an in-flight request with a unique ID and
// provider/model information for structured logging and distributed tracing.
type ObservabilityContext struct {
	// RequestID is a unique identifier for this request.
	RequestID string
	// Provider is the AI provider name (e.g. "openai", "claude").
	Provider string
	// Model is the model identifier used for this request.
	Model string
}

// NewObservabilityContext creates a new ObservabilityContext.
func NewObservabilityContext(requestID, provider, model string) *ObservabilityContext {
	return &ObservabilityContext{
		RequestID: requestID,
		Provider:  provider,
		Model:     model,
	}
}

// LogFields returns a map of key-value pairs suitable for structured logging.
func (o *ObservabilityContext) LogFields() map[string]string {
	return map[string]string{
		"request_id": o.RequestID,
		"provider":   o.Provider,
		"model":      o.Model,
	}
}

// ---------------------------------------------------------------------------
// PrometheusExporter
// ---------------------------------------------------------------------------

// PrometheusExporter exports ChatDelta metrics in Prometheus format.
//
// It mirrors the seven metrics exported by the Rust observability module:
//   - chatdelta_requests_total          (gauge — cumulative total requests)
//   - chatdelta_requests_successful_total (gauge)
//   - chatdelta_requests_failed_total     (gauge)
//   - chatdelta_request_duration_ms       (gauge — average latency per request)
//   - chatdelta_tokens_total              (gauge — cumulative tokens consumed)
//   - chatdelta_cache_hits_total          (gauge)
//   - chatdelta_cache_misses_total        (gauge)
//
// All counters are modelled as Gauges because Export receives a full
// MetricsSnapshot with absolute cumulative values (not per-call deltas).
// This is the idiomatic Go approach when syncing from an external metrics source.
//
// The Rust implementation uses a Histogram for request duration with nine
// buckets (10 ms – 10 s). Because the snapshot only carries an average, that
// information is surfaced here as a Gauge. If you need the full histogram,
// record individual observations with a prometheus.Histogram directly.
//
// Use NewPrometheusExporter to construct an instance, Export to push a snapshot,
// and Handler to serve a /metrics endpoint.
type PrometheusExporter struct {
	mu       sync.Mutex
	registry *prometheus.Registry

	requestsTotal      prometheus.Gauge
	requestsSuccessful prometheus.Gauge
	requestsFailed     prometheus.Gauge
	avgLatencyMs       prometheus.Gauge
	totalTokens        prometheus.Gauge
	cacheHits          prometheus.Gauge
	cacheMisses        prometheus.Gauge
}

// NewPrometheusExporter creates a PrometheusExporter with its own isolated
// prometheus.Registry (not the global default). Metrics are registered once at
// construction time; call Export to update their values.
func NewPrometheusExporter() *PrometheusExporter {
	reg := prometheus.NewRegistry()

	newGauge := func(name, help string) prometheus.Gauge {
		g := prometheus.NewGauge(prometheus.GaugeOpts{
			Name: name,
			Help: help,
		})
		reg.MustRegister(g)
		return g
	}

	return &PrometheusExporter{
		registry: reg,

		requestsTotal: newGauge(
			"chatdelta_requests_total",
			"Cumulative number of AI requests made.",
		),
		requestsSuccessful: newGauge(
			"chatdelta_requests_successful_total",
			"Cumulative number of successful AI requests.",
		),
		requestsFailed: newGauge(
			"chatdelta_requests_failed_total",
			"Cumulative number of failed AI requests.",
		),
		avgLatencyMs: newGauge(
			"chatdelta_request_duration_ms",
			"Average latency per successful request in milliseconds.",
		),
		totalTokens: newGauge(
			"chatdelta_tokens_total",
			"Cumulative number of tokens consumed across all requests.",
		),
		cacheHits: newGauge(
			"chatdelta_cache_hits_total",
			"Cumulative number of response cache hits.",
		),
		cacheMisses: newGauge(
			"chatdelta_cache_misses_total",
			"Cumulative number of response cache misses.",
		),
	}
}

// Export synchronises the Prometheus gauges with the given snapshot.
// It implements MetricsExporter and is safe for concurrent use.
func (e *PrometheusExporter) Export(snapshot MetricsSnapshot) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.requestsTotal.Set(float64(snapshot.RequestsTotal))
	e.requestsSuccessful.Set(float64(snapshot.RequestsSuccessful))
	e.requestsFailed.Set(float64(snapshot.RequestsFailed))
	e.avgLatencyMs.Set(snapshot.AvgLatencyMs)
	e.totalTokens.Set(float64(snapshot.TotalTokensUsed))
	e.cacheHits.Set(float64(snapshot.CacheHits))
	e.cacheMisses.Set(float64(snapshot.CacheMisses))

	return nil
}

// Name returns "prometheus".
func (e *PrometheusExporter) Name() string { return "prometheus" }

// Registry returns the underlying prometheus.Registry so callers can register
// additional metrics or integrate with existing instrumentation.
func (e *PrometheusExporter) Registry() *prometheus.Registry { return e.registry }

// Handler returns an http.Handler that serves Prometheus metrics on the
// standard /metrics path. Mount it on your HTTP mux:
//
//	http.Handle("/metrics", exporter.Handler())
func (e *PrometheusExporter) Handler() http.Handler {
	return promhttp.HandlerFor(e.registry, promhttp.HandlerOpts{})
}
