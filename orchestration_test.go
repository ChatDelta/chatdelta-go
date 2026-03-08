package chatdelta

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestOrchestrator creates an orchestrator backed by two MockClients.
func newTestOrchestrator(strategy OrchestrationStrategy) (*AiOrchestrator, *MockClient, *MockClient) {
	m1 := NewMockClient("model-a", "mock-a")
	m2 := NewMockClient("model-b", "mock-b")
	o := NewAiOrchestrator([]AIClient{m1, m2}, strategy)
	return o, m1, m2
}

func TestNewAiOrchestrator(t *testing.T) {
	m := NewMockClient("m", "")
	o := NewAiOrchestrator([]AIClient{m}, StrategyParallel)
	assert.NotNil(t, o)
}

func TestResponseCache_GetSet(t *testing.T) {
	cache := NewResponseCache(time.Minute, 10)
	resp := &FusedResponse{Content: "cached"}

	assert.Nil(t, cache.Get("key"))
	cache.Set("key", resp)
	got := cache.Get("key")
	require.NotNil(t, got)
	assert.Equal(t, "cached", got.Content)
}

func TestResponseCache_Expired(t *testing.T) {
	cache := NewResponseCache(1, 10) // 1 nanosecond TTL
	cache.Set("key", &FusedResponse{Content: "stale"})
	// Wait for expiry.
	time.Sleep(2 * time.Millisecond)
	assert.Nil(t, cache.Get("key"))
}

func TestAiOrchestrator_Parallel(t *testing.T) {
	o, m1, m2 := newTestOrchestrator(StrategyParallel)
	m1.QueueResponse("response from model-a")
	m2.QueueResponse("response from model-b")

	resp, err := o.Query(context.Background(), "hello")
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Content)
	assert.Len(t, resp.Contributions, 2)
	assert.Equal(t, 2, resp.Metrics.ModelsConsulted)
	assert.Equal(t, 2, resp.Metrics.SuccessfulResponses)
}

func TestAiOrchestrator_AllErrors(t *testing.T) {
	o, m1, m2 := newTestOrchestrator(StrategyParallel)
	m1.QueueError(errors.New("err1"))
	m2.QueueError(errors.New("err2"))

	_, err := o.Query(context.Background(), "hello")
	assert.Error(t, err)
}

func TestAiOrchestrator_Sequential(t *testing.T) {
	o, m1, m2 := newTestOrchestrator(StrategySequential)
	m1.QueueResponse("first pass")
	m2.QueueResponse("refined")

	resp, err := o.Query(context.Background(), "question")
	require.NoError(t, err)
	assert.Equal(t, "refined", resp.Content)
}

func TestAiOrchestrator_Specialized(t *testing.T) {
	o, m1, _ := newTestOrchestrator(StrategySpecialized)
	m1.QueueResponse("specialized answer")

	resp, err := o.Query(context.Background(), "implement a function in code")
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Content)
	assert.Equal(t, 1, resp.Metrics.ModelsConsulted)
}

func TestAiOrchestrator_Consensus(t *testing.T) {
	o, m1, m2 := newTestOrchestrator(StrategyConsensus)
	m1.QueueResponse("the sky is blue and beautiful")
	m2.QueueResponse("the sky appears blue during daytime")

	resp, err := o.Query(context.Background(), "color of sky")
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Content)
	assert.NotNil(t, resp.Consensus)
}

func TestAiOrchestrator_WeightedFusion(t *testing.T) {
	o, m1, m2 := newTestOrchestrator(StrategyWeightedFusion)
	m1.QueueResponse("A detailed comprehensive answer with many words covering the topic thoroughly")
	m2.QueueResponse("Short answer")

	resp, err := o.Query(context.Background(), "question")
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Content)
	assert.Greater(t, resp.Confidence, 0.0)
}

func TestAiOrchestrator_Tournament(t *testing.T) {
	o, m1, m2 := newTestOrchestrator(StrategyTournament)
	m1.QueueResponse("highly relevant detailed answer about the question at hand")
	m2.QueueResponse("x")

	resp, err := o.Query(context.Background(), "question")
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Content)
}

func TestAiOrchestrator_Adaptive_SingleClient(t *testing.T) {
	m := NewMockClient("only", "")
	m.QueueResponse("solo response")
	o := NewAiOrchestrator([]AIClient{m}, StrategyAdaptive)

	resp, err := o.Query(context.Background(), "hi")
	require.NoError(t, err)
	assert.Equal(t, "solo response", resp.Content)
	assert.Equal(t, StrategySpecialized, resp.Metrics.Strategy)
}

func TestAiOrchestrator_CacheHit(t *testing.T) {
	o, m1, m2 := newTestOrchestrator(StrategyParallel)
	m1.QueueResponse("cached response")
	m2.QueueResponse("cached response 2")

	// First call populates cache.
	resp1, err := o.Query(context.Background(), "cached prompt")
	require.NoError(t, err)

	// Second call should hit cache; no responses queued so mock fallback would differ.
	resp2, err := o.Query(context.Background(), "cached prompt")
	require.NoError(t, err)
	assert.Equal(t, resp1.Content, resp2.Content)

	snap := o.OrchestratorMetrics()
	assert.Equal(t, uint64(1), snap.CacheHits)
}

func TestAiOrchestrator_SetCapabilities(t *testing.T) {
	m := NewMockClient("specialist", "")
	o := NewAiOrchestrator([]AIClient{m}, StrategySpecialized)
	o.SetCapabilities("specialist", ModelCapabilities{
		Name:      "specialist",
		Strengths: []Strength{StrengthCodeGeneration},
	})
	m.QueueResponse("code answer")

	resp, err := o.Query(context.Background(), "implement an algorithm")
	require.NoError(t, err)
	assert.Equal(t, "code answer", resp.Content)
}

func TestCalculateConfidence(t *testing.T) {
	o := &AiOrchestrator{}
	assert.Equal(t, float64(0), o.calculateConfidence(""))

	short := o.calculateConfidence("hi")
	assert.Greater(t, short, 0.0)
	assert.LessOrEqual(t, short, 1.0)

	long := o.calculateConfidence(strings.Repeat("word ", 110))
	assert.Greater(t, long, short)
}

func TestClassifyTask(t *testing.T) {
	o := &AiOrchestrator{}
	assert.Equal(t, OrchestratorTaskCode, o.classifyTask("implement a function"))
	assert.Equal(t, OrchestratorTaskMathematics, o.classifyTask("calculate the integral"))
	assert.Equal(t, OrchestratorTaskAnalysis, o.classifyTask("analyze the data"))
	assert.Equal(t, OrchestratorTaskCreative, o.classifyTask("write a poem"))
	assert.Equal(t, OrchestratorTaskGeneral, o.classifyTask("what time is it"))
}
