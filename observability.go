// Package chatdelta provides a unified interface for interacting with multiple AI APIs.
// observability.go provides metrics export and structured logging infrastructure for
// monitoring AI client performance. The Rust implementation's optional Prometheus
// exporter is omitted here to keep the dependency footprint minimal; a TextExporter
// that writes formatted metrics to stderr is provided instead. Additional exporters
// can be wired in by implementing the MetricsExporter interface.
package chatdelta

import (
	"fmt"
	"log"
	"os"
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
