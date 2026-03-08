// Package chatdelta provides a unified interface for interacting with multiple AI APIs.
// sse.go implements a Server-Sent Events (SSE) parser for reading streaming HTTP
// responses. It uses only the standard library (bufio, strings) — no external SSE
// dependency is required.
//
// Usage:
//
//	reader := chatdelta.NewSseReader(httpResp.Body)
//	for {
//	    event, err := reader.Next()
//	    if event == nil || err != nil {
//	        break
//	    }
//	    fmt.Println(event.Data)
//	}
package chatdelta

import (
	"bufio"
	"io"
	"strconv"
	"strings"
)

// SseEvent represents a single parsed Server-Sent Event.
// Fields correspond directly to the SSE wire format defined in the
// WHATWG specification (https://html.spec.whatwg.org/multipage/server-sent-events.html).
type SseEvent struct {
	// Event is the optional event-type field (defaults to "message" per spec).
	Event string
	// Data is the event payload; multiple consecutive data lines are joined with "\n".
	Data string
	// ID is the optional last-event-ID value.
	ID string
	// Retry, when non-nil, is the reconnection time hint in milliseconds.
	Retry *int
}

// SseReader reads and parses SSE events from an io.Reader.
// It buffers the underlying reader and is NOT safe for concurrent use.
type SseReader struct {
	scanner *bufio.Scanner
}

// NewSseReader wraps r in an SseReader.
func NewSseReader(r io.Reader) *SseReader {
	return &SseReader{scanner: bufio.NewScanner(r)}
}

// Next reads the next complete SSE event from the stream.
// It returns (nil, nil) when the stream is exhausted, (nil, err) on I/O
// errors, and a populated *SseEvent when an event boundary (blank line or
// end-of-stream) is reached and at least one data line was seen.
func (s *SseReader) Next() (*SseEvent, error) {
	var ev SseEvent
	var dataLines []string
	hasData := false

	for s.scanner.Scan() {
		line := s.scanner.Text()

		// Blank line signals end of current event.
		if line == "" {
			if hasData {
				ev.Data = strings.Join(dataLines, "\n")
				return &ev, nil
			}
			// Reset for the next event.
			ev = SseEvent{}
			dataLines = dataLines[:0]
			continue
		}

		// Lines beginning with ':' are comments; ignore them.
		if strings.HasPrefix(line, ":") {
			continue
		}

		field, value, _ := strings.Cut(line, ":")
		// The spec allows an optional single space after the colon.
		value = strings.TrimPrefix(value, " ")

		switch field {
		case "event":
			ev.Event = value
		case "data":
			dataLines = append(dataLines, value)
			hasData = true
		case "id":
			ev.ID = value
		case "retry":
			if ms, err := strconv.Atoi(value); err == nil {
				ev.Retry = &ms
			}
		}
	}

	if err := s.scanner.Err(); err != nil {
		return nil, err
	}

	// Handle a final event not terminated by a trailing blank line.
	if hasData {
		ev.Data = strings.Join(dataLines, "\n")
		return &ev, nil
	}

	return nil, nil
}

// ParseSseData extracts the payload from a raw "data: …" SSE line.
// Returns the trimmed data string and true if the line is a data line;
// otherwise returns an empty string and false.
func ParseSseData(line string) (string, bool) {
	if !strings.HasPrefix(line, "data: ") {
		return "", false
	}
	return strings.TrimPrefix(line, "data: "), true
}
