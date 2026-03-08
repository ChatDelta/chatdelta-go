package chatdelta

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSseReader_SingleEvent(t *testing.T) {
	raw := "data: hello world\n\n"
	r := NewSseReader(strings.NewReader(raw))

	ev, err := r.Next()
	require.NoError(t, err)
	require.NotNil(t, ev)
	assert.Equal(t, "hello world", ev.Data)
}

func TestSseReader_EventWithType(t *testing.T) {
	raw := "event: update\ndata: payload\n\n"
	r := NewSseReader(strings.NewReader(raw))

	ev, err := r.Next()
	require.NoError(t, err)
	require.NotNil(t, ev)
	assert.Equal(t, "update", ev.Event)
	assert.Equal(t, "payload", ev.Data)
}

func TestSseReader_EventWithID(t *testing.T) {
	raw := "id: 42\ndata: body\n\n"
	r := NewSseReader(strings.NewReader(raw))

	ev, err := r.Next()
	require.NoError(t, err)
	require.NotNil(t, ev)
	assert.Equal(t, "42", ev.ID)
	assert.Equal(t, "body", ev.Data)
}

func TestSseReader_RetryField(t *testing.T) {
	raw := "retry: 3000\ndata: x\n\n"
	r := NewSseReader(strings.NewReader(raw))

	ev, err := r.Next()
	require.NoError(t, err)
	require.NotNil(t, ev)
	require.NotNil(t, ev.Retry)
	assert.Equal(t, 3000, *ev.Retry)
}

func TestSseReader_MultiLineData(t *testing.T) {
	raw := "data: line1\ndata: line2\ndata: line3\n\n"
	r := NewSseReader(strings.NewReader(raw))

	ev, err := r.Next()
	require.NoError(t, err)
	require.NotNil(t, ev)
	assert.Equal(t, "line1\nline2\nline3", ev.Data)
}

func TestSseReader_MultipleEvents(t *testing.T) {
	raw := "data: first\n\ndata: second\n\n"
	r := NewSseReader(strings.NewReader(raw))

	ev1, err := r.Next()
	require.NoError(t, err)
	require.NotNil(t, ev1)
	assert.Equal(t, "first", ev1.Data)

	ev2, err := r.Next()
	require.NoError(t, err)
	require.NotNil(t, ev2)
	assert.Equal(t, "second", ev2.Data)

	ev3, err := r.Next()
	require.NoError(t, err)
	assert.Nil(t, ev3)
}

func TestSseReader_IgnoresComments(t *testing.T) {
	raw := ": this is a comment\ndata: real data\n\n"
	r := NewSseReader(strings.NewReader(raw))

	ev, err := r.Next()
	require.NoError(t, err)
	require.NotNil(t, ev)
	assert.Equal(t, "real data", ev.Data)
}

func TestSseReader_EmptyStream(t *testing.T) {
	r := NewSseReader(strings.NewReader(""))
	ev, err := r.Next()
	require.NoError(t, err)
	assert.Nil(t, ev)
}

func TestSseReader_NoTrailingBlankLine(t *testing.T) {
	// Final event without trailing blank line should still be returned.
	raw := "data: final"
	r := NewSseReader(strings.NewReader(raw))

	ev, err := r.Next()
	require.NoError(t, err)
	require.NotNil(t, ev)
	assert.Equal(t, "final", ev.Data)
}

func TestParseSseData(t *testing.T) {
	data, ok := ParseSseData("data: hello")
	assert.True(t, ok)
	assert.Equal(t, "hello", data)

	_, ok = ParseSseData("event: update")
	assert.False(t, ok)

	data, ok = ParseSseData("data: [DONE]")
	assert.True(t, ok)
	assert.Equal(t, "[DONE]", data)
}
