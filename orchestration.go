// Package chatdelta provides a unified interface for interacting with multiple AI APIs.
// orchestration.go implements multi-model AI orchestration with seven coordination
// strategies, confidence-based response fusion, consensus analysis, task-type routing,
// and a TTL-based in-memory response cache backed by sync.Map. No external libraries
// beyond the standard library are used.
package chatdelta

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// OrchestrationStrategy selects how multiple AI clients are coordinated.
type OrchestrationStrategy int

const (
	// StrategyParallel queries all clients simultaneously and picks the highest-
	// confidence response as the primary content.
	StrategyParallel OrchestrationStrategy = iota
	// StrategySequential passes output through each client in order, with each
	// client refining the previous response.
	StrategySequential
	// StrategySpecialized routes the prompt to the single best-suited client
	// based on task type and declared model strengths.
	StrategySpecialized
	// StrategyConsensus queries all clients and adds agreement analysis.
	StrategyConsensus
	// StrategyWeightedFusion merges all responses weighted by confidence scores.
	StrategyWeightedFusion
	// StrategyTournament scores every response and returns the winner.
	StrategyTournament
	// StrategyAdaptive selects a strategy at runtime based on the number of
	// configured clients and prompt characteristics.
	StrategyAdaptive
)

// Strength describes a capability area in which a model excels.
type Strength string

const (
	StrengthReasoning      Strength = "reasoning"
	StrengthCreativity     Strength = "creativity"
	StrengthCodeGeneration Strength = "code_generation"
	StrengthMathematics    Strength = "mathematics"
	StrengthLanguage       Strength = "language"
	StrengthAnalysis       Strength = "analysis"
	StrengthVision         Strength = "vision"
	StrengthSpeed          Strength = "speed"
)

// ModelCapabilities describes a client's characteristics and areas of strength.
type ModelCapabilities struct {
	// Name is the client/model identifier.
	Name string
	// Strengths lists areas where this model performs best.
	Strengths []Strength
	// AvgLatencyMs is the historical average latency in milliseconds.
	AvgLatencyMs int64
	// CostPer1kTokens is the approximate cost in USD per 1,000 tokens.
	CostPer1kTokens float32
	// MaxContextLength is the maximum number of tokens the model accepts.
	MaxContextLength int
	// SupportsStreaming indicates streaming capability.
	SupportsStreaming bool
	// SupportsVision indicates multi-modal image understanding.
	SupportsVision bool
	// SupportsFunctionCalling indicates function/tool-call capability.
	SupportsFunctionCalling bool
}

// OrchestratorTaskType classifies a prompt for routing decisions.
type OrchestratorTaskType string

const (
	OrchestratorTaskCode        OrchestratorTaskType = "code"
	OrchestratorTaskCreative    OrchestratorTaskType = "creative"
	OrchestratorTaskAnalysis    OrchestratorTaskType = "analysis"
	OrchestratorTaskMathematics OrchestratorTaskType = "mathematics"
	OrchestratorTaskGeneral     OrchestratorTaskType = "general"
)

// ModelContribution holds one client's response within a FusedResponse.
type ModelContribution struct {
	// ClientName identifies the contributing client.
	ClientName string
	// Response is the raw text returned by the client.
	Response string
	// Confidence is a 0–1 quality estimate computed from response characteristics.
	Confidence float64
	// LatencyMs is the wall-clock time to receive this response.
	LatencyMs int64
	// Error holds any error from this client, if applicable.
	Error error
}

// FactCheck is a simple consistency verdict for a single claim.
type FactCheck struct {
	// Claim is the assertion being verified.
	Claim string
	// Consistent is true when a majority of models agree on the claim.
	Consistent bool
	// SupportCount is the number of models that expressed the claim.
	SupportCount int
}

// ConsensusAnalysis summarises agreement across model responses.
type ConsensusAnalysis struct {
	// AgreementScore is the fraction of models that produced similar responses (0–1).
	AgreementScore float64
	// CommonThemes are terms appearing in a majority of responses.
	CommonThemes []string
	// FactChecks contains consistency verdicts for notable claims.
	FactChecks []FactCheck
}

// OrchestrationMetrics records performance data for one orchestration call.
type OrchestrationMetrics struct {
	// TotalLatencyMs is the wall-clock time from call start to result.
	TotalLatencyMs int64
	// ModelsConsulted is the number of clients that were queried.
	ModelsConsulted int
	// SuccessfulResponses is the number of clients that returned without error.
	SuccessfulResponses int
	// Strategy is the strategy that was ultimately applied.
	Strategy OrchestrationStrategy
}

// FusedResponse is the aggregated output of an orchestration call.
type FusedResponse struct {
	// Content is the final merged or selected response text.
	Content string
	// Contributions holds each model's individual result.
	Contributions []ModelContribution
	// Confidence is the overall confidence in the fused answer (0–1).
	Confidence float64
	// Consensus contains agreement analysis when applicable.
	Consensus *ConsensusAnalysis
	// Metrics records performance information for this call.
	Metrics OrchestrationMetrics
}

// cacheEntry wraps a FusedResponse with its expiry time.
type cacheEntry struct {
	response  *FusedResponse
	expiresAt time.Time
}

// ResponseCache is a TTL-based in-memory cache backed by sync.Map.
// It is safe for concurrent use. Expired entries are removed lazily on Get.
type ResponseCache struct {
	store sync.Map
	ttl   time.Duration
}

// NewResponseCache creates a ResponseCache with the specified TTL.
// maxItems is accepted for API compatibility but not enforced (entries expire via TTL).
func NewResponseCache(ttl time.Duration, maxItems int) *ResponseCache {
	_ = maxItems
	return &ResponseCache{ttl: ttl}
}

// Get retrieves a cached response. Returns nil if absent or expired.
func (c *ResponseCache) Get(key string) *FusedResponse {
	val, ok := c.store.Load(key)
	if !ok {
		return nil
	}
	entry := val.(*cacheEntry)
	if time.Now().After(entry.expiresAt) {
		c.store.Delete(key)
		return nil
	}
	return entry.response
}

// Set stores a response in the cache under key.
func (c *ResponseCache) Set(key string, resp *FusedResponse) {
	c.store.Store(key, &cacheEntry{
		response:  resp,
		expiresAt: time.Now().Add(c.ttl),
	})
}

// AiOrchestrator coordinates multiple AI clients using a configurable strategy.
// Use NewAiOrchestrator to create an instance.
type AiOrchestrator struct {
	clients      []AIClient
	capabilities map[string]ModelCapabilities
	strategy     OrchestrationStrategy
	metrics      *ClientMetrics
	cache        *ResponseCache
}

// NewAiOrchestrator creates an orchestrator wrapping the given clients.
// Basic ModelCapabilities are inferred from each client at construction time.
// A five-minute response cache is created automatically.
func NewAiOrchestrator(clients []AIClient, strategy OrchestrationStrategy) *AiOrchestrator {
	caps := make(map[string]ModelCapabilities, len(clients))
	for _, c := range clients {
		caps[c.Name()] = ModelCapabilities{
			Name:             c.Name(),
			SupportsStreaming: c.SupportsStreaming(),
		}
	}
	return &AiOrchestrator{
		clients:      clients,
		capabilities: caps,
		strategy:     strategy,
		metrics:      NewClientMetrics(),
		cache:        NewResponseCache(5*time.Minute, 100),
	}
}

// SetCapabilities registers detailed capability metadata for a named model.
// Call this after NewAiOrchestrator to enable strength-based routing.
func (o *AiOrchestrator) SetCapabilities(name string, caps ModelCapabilities) {
	o.capabilities[name] = caps
}

// Query sends prompt to the configured clients using the orchestration strategy.
// Successful responses are cached by prompt text for the configured TTL.
func (o *AiOrchestrator) Query(ctx context.Context, prompt string) (*FusedResponse, error) {
	if cached := o.cache.Get(prompt); cached != nil {
		o.metrics.RecordCacheHit()
		return cached, nil
	}
	o.metrics.RecordCacheMiss()

	strategy := o.strategy
	if strategy == StrategyAdaptive {
		strategy = o.selectAdaptiveStrategy(prompt)
	}

	timer := NewRequestTimer()

	var (
		resp *FusedResponse
		err  error
	)
	switch strategy {
	case StrategyParallel:
		resp, err = o.executeParallel(ctx, prompt)
	case StrategySequential:
		resp, err = o.executeSequential(ctx, prompt)
	case StrategySpecialized:
		resp, err = o.executeSpecialized(ctx, prompt)
	case StrategyConsensus:
		resp, err = o.executeConsensus(ctx, prompt)
	case StrategyWeightedFusion:
		resp, err = o.executeWeightedFusion(ctx, prompt)
	case StrategyTournament:
		resp, err = o.executeTournament(ctx, prompt)
	default:
		resp, err = o.executeParallel(ctx, prompt)
	}

	elapsed := timer.ElapsedMs()
	if err != nil {
		o.metrics.RecordRequest(false, elapsed, 0)
		return nil, err
	}

	resp.Metrics.TotalLatencyMs = elapsed
	resp.Metrics.Strategy = strategy
	o.metrics.RecordRequest(true, elapsed, 0)
	o.cache.Set(prompt, resp)
	return resp, nil
}

// gatherContributions queries every client in parallel and collects results.
func (o *AiOrchestrator) gatherContributions(ctx context.Context, prompt string) []ModelContribution {
	contribs := make([]ModelContribution, len(o.clients))
	var wg sync.WaitGroup
	for i, client := range o.clients {
		wg.Add(1)
		go func(idx int, c AIClient) {
			defer wg.Done()
			t := NewRequestTimer()
			response, err := c.SendPrompt(ctx, prompt)
			contrib := ModelContribution{
				ClientName: c.Name(),
				Response:   response,
				LatencyMs:  t.ElapsedMs(),
				Error:      err,
			}
			if err == nil {
				contrib.Confidence = o.calculateConfidence(response)
			}
			contribs[idx] = contrib
		}(i, client)
	}
	wg.Wait()
	return contribs
}

// executeParallel queries all clients and selects the highest-confidence response.
func (o *AiOrchestrator) executeParallel(ctx context.Context, prompt string) (*FusedResponse, error) {
	return o.buildFusedFromContribs(o.gatherContributions(ctx, prompt))
}

// executeSequential passes the prompt through each client in order, chaining output.
// The last successful response is used as the final content.
func (o *AiOrchestrator) executeSequential(ctx context.Context, prompt string) (*FusedResponse, error) {
	if len(o.clients) == 0 {
		return nil, fmt.Errorf("no clients configured")
	}
	var contribs []ModelContribution
	current := prompt
	var lastSuccessful *ModelContribution
	for _, client := range o.clients {
		t := NewRequestTimer()
		response, err := client.SendPrompt(ctx, current)
		contrib := ModelContribution{
			ClientName: client.Name(),
			Response:   response,
			LatencyMs:  t.ElapsedMs(),
			Error:      err,
		}
		if err == nil {
			contrib.Confidence = o.calculateConfidence(response)
			current = response
			c := contrib
			lastSuccessful = &c
		}
		contribs = append(contribs, contrib)
	}
	if lastSuccessful == nil {
		return nil, fmt.Errorf("all %d client(s) returned errors", len(contribs))
	}
	successful := 0
	var totalConf float64
	for _, c := range contribs {
		if c.Error == nil {
			successful++
			totalConf += c.Confidence
		}
	}
	return &FusedResponse{
		Content:       lastSuccessful.Response,
		Contributions: contribs,
		Confidence:    totalConf / float64(successful),
		Metrics: OrchestrationMetrics{
			ModelsConsulted:     len(contribs),
			SuccessfulResponses: successful,
		},
	}, nil
}

// executeSpecialized routes the prompt to the best-suited single client.
func (o *AiOrchestrator) executeSpecialized(ctx context.Context, prompt string) (*FusedResponse, error) {
	client := o.selectSpecialist(o.classifyTask(prompt))
	t := NewRequestTimer()
	response, err := client.SendPrompt(ctx, prompt)
	if err != nil {
		return nil, err
	}
	contrib := ModelContribution{
		ClientName: client.Name(),
		Response:   response,
		LatencyMs:  t.ElapsedMs(),
		Confidence: o.calculateConfidence(response),
	}
	return &FusedResponse{
		Content:       response,
		Contributions: []ModelContribution{contrib},
		Confidence:    contrib.Confidence,
		Metrics:       OrchestrationMetrics{ModelsConsulted: 1, SuccessfulResponses: 1},
	}, nil
}

// executeConsensus queries all clients and adds ConsensusAnalysis to the result.
func (o *AiOrchestrator) executeConsensus(ctx context.Context, prompt string) (*FusedResponse, error) {
	contribs := o.gatherContributions(ctx, prompt)
	resp, err := o.buildFusedFromContribs(contribs)
	if err != nil {
		return nil, err
	}
	resp.Consensus = o.analyzeConsensus(contribs)
	return resp, nil
}

// executeWeightedFusion merges all successful responses weighted by confidence.
func (o *AiOrchestrator) executeWeightedFusion(ctx context.Context, prompt string) (*FusedResponse, error) {
	contribs := o.gatherContributions(ctx, prompt)
	var parts []string
	var totalWeight float64
	for _, c := range contribs {
		if c.Error == nil && c.Confidence > 0 {
			parts = append(parts, fmt.Sprintf("[%.0f%%] %s", c.Confidence*100, c.Response))
			totalWeight += c.Confidence
		}
	}
	resp, err := o.buildFusedFromContribs(contribs)
	if err != nil {
		return nil, err
	}
	if len(parts) > 1 {
		resp.Content = strings.Join(parts, "\n\n---\n\n")
	}
	if totalWeight > 0 && len(contribs) > 0 {
		resp.Confidence = totalWeight / float64(len(contribs))
	}
	return resp, nil
}

// executeTournament scores all responses and returns the best.
func (o *AiOrchestrator) executeTournament(ctx context.Context, prompt string) (*FusedResponse, error) {
	contribs := o.gatherContributions(ctx, prompt)
	var best *ModelContribution
	for i := range contribs {
		c := &contribs[i]
		if c.Error != nil {
			continue
		}
		c.Confidence = o.scoreResponse(c.Response, prompt) / 100.0
		if best == nil || c.Confidence > best.Confidence {
			best = c
		}
	}
	if best == nil {
		return nil, fmt.Errorf("all clients returned errors")
	}
	successful := 0
	for _, c := range contribs {
		if c.Error == nil {
			successful++
		}
	}
	return &FusedResponse{
		Content:       best.Response,
		Contributions: contribs,
		Confidence:    best.Confidence,
		Metrics:       OrchestrationMetrics{ModelsConsulted: len(contribs), SuccessfulResponses: successful},
	}, nil
}

// buildFusedFromContribs assembles a FusedResponse from a slice of contributions,
// selecting the highest-confidence successful response as the primary content.
func (o *AiOrchestrator) buildFusedFromContribs(contribs []ModelContribution) (*FusedResponse, error) {
	var successful []ModelContribution
	for _, c := range contribs {
		if c.Error == nil {
			successful = append(successful, c)
		}
	}
	if len(successful) == 0 {
		return nil, fmt.Errorf("all %d client(s) returned errors", len(contribs))
	}
	best := successful[0]
	for _, c := range successful[1:] {
		if c.Confidence > best.Confidence {
			best = c
		}
	}
	var total float64
	for _, c := range successful {
		total += c.Confidence
	}
	return &FusedResponse{
		Content:       best.Response,
		Contributions: contribs,
		Confidence:    total / float64(len(successful)),
		Metrics: OrchestrationMetrics{
			ModelsConsulted:     len(contribs),
			SuccessfulResponses: len(successful),
		},
	}, nil
}

// calculateConfidence estimates response quality on a 0–1 scale using heuristics.
func (o *AiOrchestrator) calculateConfidence(response string) float64 {
	if response == "" {
		return 0
	}
	score := 0.5
	words := len(strings.Fields(response))
	switch {
	case words > 100:
		score += 0.3
	case words > 50:
		score += 0.2
	case words > 20:
		score += 0.1
	}
	if strings.Contains(response, "\n") {
		score += 0.1
	}
	if strings.Contains(response, "- ") || strings.Contains(response, "1.") {
		score += 0.1
	}
	if score > 1.0 {
		score = 1.0
	}
	return score
}

// scoreResponse scores a response against a prompt on a 0–100 scale.
func (o *AiOrchestrator) scoreResponse(response, prompt string) float64 {
	score := 50.0
	promptWords := strings.Fields(strings.ToLower(prompt))
	responseL := strings.ToLower(response)
	matched := 0
	for _, word := range promptWords {
		if len(word) > 3 && strings.Contains(responseL, word) {
			matched++
		}
	}
	if len(promptWords) > 0 {
		score += float64(matched) / float64(len(promptWords)) * 30
	}
	switch words := len(strings.Fields(response)); {
	case words > 100:
		score += 20
	case words > 50:
		score += 10
	case words > 20:
		score += 5
	}
	if score > 100 {
		score = 100
	}
	return score
}

// classifyTask determines the task type from prompt keywords.
func (o *AiOrchestrator) classifyTask(prompt string) OrchestratorTaskType {
	lower := strings.ToLower(prompt)
	for _, kw := range []string{"code", "function", "class", "implement", "debug", "algorithm", "program"} {
		if strings.Contains(lower, kw) {
			return OrchestratorTaskCode
		}
	}
	for _, kw := range []string{"math", "calculate", "equation", "formula", "solve", "integral", "derivative"} {
		if strings.Contains(lower, kw) {
			return OrchestratorTaskMathematics
		}
	}
	for _, kw := range []string{"analyze", "analysis", "compare", "evaluate", "assess"} {
		if strings.Contains(lower, kw) {
			return OrchestratorTaskAnalysis
		}
	}
	for _, kw := range []string{"write", "story", "poem", "creative", "compose", "imagine"} {
		if strings.Contains(lower, kw) {
			return OrchestratorTaskCreative
		}
	}
	return OrchestratorTaskGeneral
}

// selectSpecialist returns the best client for the given task type.
// Falls back to the first client when no client declares the matching strength.
func (o *AiOrchestrator) selectSpecialist(taskType OrchestratorTaskType) AIClient {
	if len(o.clients) == 0 {
		panic("no clients configured")
	}
	target := o.taskToStrength(taskType)
	best := o.clients[0]
	bestScore := 0
	for _, client := range o.clients {
		caps, ok := o.capabilities[client.Name()]
		if !ok {
			continue
		}
		for _, s := range caps.Strengths {
			if s == target {
				bestScore = 2
				best = client
				break
			}
		}
		_ = bestScore
	}
	return best
}

func (o *AiOrchestrator) taskToStrength(taskType OrchestratorTaskType) Strength {
	switch taskType {
	case OrchestratorTaskCode:
		return StrengthCodeGeneration
	case OrchestratorTaskCreative:
		return StrengthCreativity
	case OrchestratorTaskAnalysis:
		return StrengthAnalysis
	case OrchestratorTaskMathematics:
		return StrengthMathematics
	default:
		return StrengthReasoning
	}
}

// analyzeConsensus builds agreement statistics across a slice of contributions.
func (o *AiOrchestrator) analyzeConsensus(contribs []ModelContribution) *ConsensusAnalysis {
	var responses []string
	for _, c := range contribs {
		if c.Error == nil {
			responses = append(responses, strings.ToLower(c.Response))
		}
	}
	if len(responses) == 0 {
		return &ConsensusAnalysis{}
	}

	// Word frequency analysis for common themes.
	wordCount := make(map[string]int)
	for _, r := range responses {
		for _, word := range strings.Fields(r) {
			if len(word) > 4 {
				wordCount[word]++
			}
		}
	}
	threshold := len(responses)/2 + 1
	type wf struct {
		word  string
		count int
	}
	var freqs []wf
	for w, c := range wordCount {
		if c >= threshold {
			freqs = append(freqs, wf{w, c})
		}
	}
	sort.Slice(freqs, func(i, j int) bool { return freqs[i].count > freqs[j].count })

	var themes []string
	for i, f := range freqs {
		if i >= 5 {
			break
		}
		themes = append(themes, f.word)
	}

	agreementScore := float64(len(themes)) / 5.0
	if agreementScore > 1 {
		agreementScore = 1
	}
	if len(responses) > 1 {
		agreementScore *= float64(len(responses)-1) / float64(len(responses))
	}

	return &ConsensusAnalysis{
		AgreementScore: agreementScore,
		CommonThemes:   themes,
	}
}

// selectAdaptiveStrategy chooses a strategy based on the number of available clients.
func (o *AiOrchestrator) selectAdaptiveStrategy(_ string) OrchestrationStrategy {
	switch {
	case len(o.clients) == 1:
		return StrategySpecialized
	case len(o.clients) <= 3:
		return StrategyWeightedFusion
	default:
		return StrategyTournament
	}
}

// OrchestratorMetrics returns a snapshot of the orchestrator's accumulated metrics.
func (o *AiOrchestrator) OrchestratorMetrics() MetricsSnapshot {
	return o.metrics.Snapshot()
}
