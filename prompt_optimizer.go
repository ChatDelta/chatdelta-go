// Package chatdelta provides a unified interface for interacting with multiple AI APIs.
// prompt_optimizer.go implements a context-aware prompt engineering engine that
// analyses incoming prompts, classifies their task type and target expertise level,
// then applies a set of configurable enhancement techniques to improve response quality.
package chatdelta

import (
	"fmt"
	"strings"
)

// PromptTaskType classifies the nature of a prompt for optimization routing.
type PromptTaskType string

const (
	// PromptTaskAnalysis covers evaluation, comparison, and data-analysis tasks.
	PromptTaskAnalysis PromptTaskType = "analysis"
	// PromptTaskGeneration covers text generation tasks.
	PromptTaskGeneration PromptTaskType = "generation"
	// PromptTaskSummarization covers condensing or abstracting existing content.
	PromptTaskSummarization PromptTaskType = "summarization"
	// PromptTaskTranslation covers language translation tasks.
	PromptTaskTranslation PromptTaskType = "translation"
	// PromptTaskQA covers question-answering and explanation tasks.
	PromptTaskQA PromptTaskType = "qa"
	// PromptTaskReasoning covers logical inference and multi-step reasoning.
	PromptTaskReasoning PromptTaskType = "reasoning"
	// PromptTaskCreative covers open-ended creative writing tasks.
	PromptTaskCreative PromptTaskType = "creative"
	// PromptTaskTechnical covers code, algorithms, and technical documentation.
	PromptTaskTechnical PromptTaskType = "technical"
)

// ExpertiseLevel represents the estimated expertise of the intended audience.
type ExpertiseLevel string

const (
	// ExpertiseBeginner targets users with minimal domain knowledge.
	ExpertiseBeginner ExpertiseLevel = "beginner"
	// ExpertiseIntermediate targets users with moderate domain knowledge.
	ExpertiseIntermediate ExpertiseLevel = "intermediate"
	// ExpertiseExpert targets domain experts expecting technical depth.
	ExpertiseExpert ExpertiseLevel = "expert"
)

// PromptContext holds characteristics detected in the input prompt.
type PromptContext struct {
	// TaskType is the detected task category.
	TaskType PromptTaskType
	// ExpertiseLevel is the estimated target expertise.
	ExpertiseLevel ExpertiseLevel
	// Keywords are notable terms found in the prompt.
	Keywords []string
}

// OptimizationTechnique names a single enhancement applied to a prompt.
type OptimizationTechnique string

const (
	// TechniqueClarity adds punctuation and direct phrasing.
	TechniqueClarity OptimizationTechnique = "clarity_enhancement"
	// TechniqueContextInjection prepends task-specific framing.
	TechniqueContextInjection OptimizationTechnique = "context_injection"
	// TechniqueChainOfThought introduces step-by-step reasoning language.
	TechniqueChainOfThought OptimizationTechnique = "chain_of_thought"
	// TechniqueFewShot appends a request for illustrative examples.
	TechniqueFewShot OptimizationTechnique = "few_shot_learning"
	// TechniqueRoleSpecification prefixes a persona assignment.
	TechniqueRoleSpecification OptimizationTechnique = "role_specification"
)

// PromptVariation is an alternative phrasing of the prompt with an estimated improvement.
type PromptVariation struct {
	// Prompt is the alternative phrasing.
	Prompt string
	// EstimatedImprovement is the estimated percentage gain over the original.
	EstimatedImprovement float64
}

// OptimizedPrompt is the result of running PromptOptimizer.Optimize.
type OptimizedPrompt struct {
	// Original is the unmodified input.
	Original string
	// Optimized is the enhanced prompt.
	Optimized string
	// Techniques lists each enhancement applied, in order.
	Techniques []OptimizationTechnique
	// Context holds the detected characteristics of the input.
	Context PromptContext
	// Confidence is a 0–1 quality estimate for the optimized prompt.
	Confidence float64
	// Variations are alternative phrasings worth considering.
	Variations []PromptVariation
}

// PromptOptimizer analyses prompts and applies context-appropriate enhancements.
type PromptOptimizer struct {
	history []optimizerRecord
}

type optimizerRecord struct {
	original  string
	optimized string
	score     float64
}

// NewPromptOptimizer creates a PromptOptimizer with default settings.
func NewPromptOptimizer() *PromptOptimizer {
	return &PromptOptimizer{}
}

// Optimize analyses prompt, applies enhancement techniques, and returns an
// OptimizedPrompt containing the improved text and diagnostic metadata.
func (o *PromptOptimizer) Optimize(prompt string) *OptimizedPrompt {
	ctx := o.analyzeContext(prompt)
	techniques, optimized := o.applyStrategies(prompt, ctx)
	variations := generateVariations(prompt)
	confidence := float64(len(techniques)) / 5.0
	if confidence > 1.0 {
		confidence = 1.0
	}
	return &OptimizedPrompt{
		Original:   prompt,
		Optimized:  optimized,
		Techniques: techniques,
		Context:    ctx,
		Confidence: confidence,
		Variations: variations,
	}
}

// analyzeContext detects the task type, expertise level, and keywords in prompt.
func (o *PromptOptimizer) analyzeContext(prompt string) PromptContext {
	lower := strings.ToLower(prompt)
	return PromptContext{
		TaskType:       detectPromptTaskType(lower),
		ExpertiseLevel: detectExpertiseLevel(lower),
		Keywords:       extractPromptKeywords(lower),
	}
}

// promptTaskKeywords maps each PromptTaskType to its associated keywords.
var promptTaskKeywords = map[PromptTaskType][]string{
	PromptTaskAnalysis:      {"analyze", "analysis", "compare", "evaluate", "assess", "review"},
	PromptTaskGeneration:    {"generate", "write", "create", "draft", "compose", "produce"},
	PromptTaskSummarization: {"summarize", "summary", "tldr", "condense", "brief"},
	PromptTaskTranslation:   {"translate", "translation", "convert to", "in spanish", "in french", "in german"},
	PromptTaskQA:            {"what is", "how does", "why", "when", "who", "explain", "describe"},
	PromptTaskReasoning:     {"reason", "logic", "deduce", "infer", "prove", "conclude"},
	PromptTaskCreative:      {"story", "poem", "creative", "imagine", "fiction", "narrative"},
	PromptTaskTechnical:     {"code", "function", "algorithm", "implement", "debug", "program", "class", "api"},
}

func detectPromptTaskType(lower string) PromptTaskType {
	counts := make(map[PromptTaskType]int, len(promptTaskKeywords))
	for tt, keywords := range promptTaskKeywords {
		for _, kw := range keywords {
			if strings.Contains(lower, kw) {
				counts[tt]++
			}
		}
	}
	best := PromptTaskQA
	bestCount := 0
	for tt, count := range counts {
		if count > bestCount {
			bestCount = count
			best = tt
		}
	}
	return best
}

var technicalTermsList = []string{
	"algorithm", "complexity", "recursive", "async", "concurrent",
	"api", "interface", "abstract", "polymorphism", "heuristic",
	"gradient", "tensor", "latency", "throughput", "idempotent",
}

func detectExpertiseLevel(lower string) ExpertiseLevel {
	count := 0
	for _, term := range technicalTermsList {
		if strings.Contains(lower, term) {
			count++
		}
	}
	switch {
	case count >= 3:
		return ExpertiseExpert
	case count >= 1:
		return ExpertiseIntermediate
	default:
		return ExpertiseBeginner
	}
}

func extractPromptKeywords(lower string) []string {
	seen := make(map[string]struct{})
	var keywords []string
	for _, terms := range promptTaskKeywords {
		for _, kw := range terms {
			if _, found := seen[kw]; !found && strings.Contains(lower, kw) {
				seen[kw] = struct{}{}
				keywords = append(keywords, kw)
			}
		}
	}
	return keywords
}

// applyStrategies returns the list of techniques applied and the enhanced prompt.
func (o *PromptOptimizer) applyStrategies(prompt string, ctx PromptContext) ([]OptimizationTechnique, string) {
	var techniques []OptimizationTechnique
	result := strings.TrimSpace(prompt)

	// Clarity: ensure prompt ends with punctuation.
	last := result[len(result)-1]
	if last != '?' && last != '.' && last != '!' {
		result += "."
		techniques = append(techniques, TechniqueClarity)
	}

	// Context injection based on task type.
	if prefix := taskContextPrefix(ctx.TaskType); prefix != "" {
		result = prefix + " " + result
		techniques = append(techniques, TechniqueContextInjection)
	}

	// Chain of thought for reasoning and analysis tasks.
	if ctx.TaskType == PromptTaskReasoning || ctx.TaskType == PromptTaskAnalysis {
		result += " Please think step by step."
		techniques = append(techniques, TechniqueChainOfThought)
	}

	// Role specification for technical tasks or expert-level prompts.
	if ctx.TaskType == PromptTaskTechnical || ctx.ExpertiseLevel == ExpertiseExpert {
		result = "As an expert software engineer: " + result
		techniques = append(techniques, TechniqueRoleSpecification)
	}

	return techniques, result
}

func taskContextPrefix(tt PromptTaskType) string {
	switch tt {
	case PromptTaskAnalysis:
		return "Please provide a detailed analysis:"
	case PromptTaskGeneration:
		return "Please generate the following:"
	case PromptTaskSummarization:
		return "Please provide a concise summary:"
	case PromptTaskTranslation:
		return "Please translate the following:"
	case PromptTaskReasoning:
		return "Please reason through the following:"
	case PromptTaskCreative:
		return "Please create the following:"
	case PromptTaskTechnical:
		return "Please provide a technical solution:"
	default:
		return ""
	}
}

// generateVariations returns three alternative phrasings with estimated improvements.
func generateVariations(prompt string) []PromptVariation {
	return []PromptVariation{
		{
			Prompt:               fmt.Sprintf("Be very specific and detailed: %s", prompt),
			EstimatedImprovement: 15.0,
		},
		{
			Prompt:               fmt.Sprintf("%s Please provide examples.", prompt),
			EstimatedImprovement: 10.0,
		},
		{
			Prompt:               fmt.Sprintf("%s Format your response as a structured list.", prompt),
			EstimatedImprovement: 8.0,
		},
	}
}
