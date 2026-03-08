package chatdelta

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewPromptOptimizer(t *testing.T) {
	o := NewPromptOptimizer()
	assert.NotNil(t, o)
}

func TestOptimize_RetainsOriginal(t *testing.T) {
	o := NewPromptOptimizer()
	original := "What is Go?"
	result := o.Optimize(original)
	assert.Equal(t, original, result.Original)
}

func TestOptimize_ProducesOptimized(t *testing.T) {
	o := NewPromptOptimizer()
	result := o.Optimize("What is Go?")
	assert.NotEmpty(t, result.Optimized)
}

func TestOptimize_AppliesClarity(t *testing.T) {
	o := NewPromptOptimizer()
	// Prompt without trailing punctuation should trigger clarity technique.
	result := o.Optimize("Tell me about Go")
	assert.Contains(t, result.Techniques, TechniqueClarity)
	assert.True(t, strings.HasSuffix(result.Optimized, ".") ||
		strings.HasSuffix(result.Optimized, "?") ||
		strings.HasSuffix(result.Optimized, "!"))
}

func TestOptimize_NoClarityWhenPunctuated(t *testing.T) {
	o := NewPromptOptimizer()
	result := o.Optimize("What is Go?")
	assert.NotContains(t, result.Techniques, TechniqueClarity)
}

func TestOptimize_ChainOfThoughtForReasoning(t *testing.T) {
	o := NewPromptOptimizer()
	result := o.Optimize("Please reason through this logic problem.")
	assert.Contains(t, result.Techniques, TechniqueChainOfThought)
	assert.Contains(t, result.Optimized, "step by step")
}

func TestOptimize_RoleSpecificationForTechnical(t *testing.T) {
	o := NewPromptOptimizer()
	result := o.Optimize("Write a function to sort a list.")
	assert.Contains(t, result.Techniques, TechniqueRoleSpecification)
	assert.Contains(t, result.Optimized, "expert")
}

func TestOptimize_ConfidenceRange(t *testing.T) {
	o := NewPromptOptimizer()
	result := o.Optimize("Analyze the algorithm complexity.")
	assert.GreaterOrEqual(t, result.Confidence, 0.0)
	assert.LessOrEqual(t, result.Confidence, 1.0)
}

func TestOptimize_GeneratesVariations(t *testing.T) {
	o := NewPromptOptimizer()
	result := o.Optimize("What is Go?")
	require.Len(t, result.Variations, 3)
	for _, v := range result.Variations {
		assert.NotEmpty(t, v.Prompt)
		assert.Greater(t, v.EstimatedImprovement, 0.0)
	}
}

func TestDetectPromptTaskType_Technical(t *testing.T) {
	tt := detectPromptTaskType("write a function to implement a binary search algorithm")
	assert.Equal(t, PromptTaskTechnical, tt)
}

func TestDetectPromptTaskType_Creative(t *testing.T) {
	tt := detectPromptTaskType("write a short story about a dragon")
	// "write" matches generation; "story" matches creative — creative has more hits
	assert.Equal(t, PromptTaskCreative, tt)
}

func TestDetectExpertiseLevel_Expert(t *testing.T) {
	level := detectExpertiseLevel("optimize the algorithm complexity using a recursive concurrent api interface with polymorphism")
	assert.Equal(t, ExpertiseExpert, level)
}

func TestDetectExpertiseLevel_Intermediate(t *testing.T) {
	level := detectExpertiseLevel("explain the api endpoints")
	assert.Equal(t, ExpertiseIntermediate, level)
}

func TestDetectExpertiseLevel_Beginner(t *testing.T) {
	level := detectExpertiseLevel("how do I say hello in python")
	assert.Equal(t, ExpertiseBeginner, level)
}

func TestOptimize_ContextInjection(t *testing.T) {
	o := NewPromptOptimizer()
	result := o.Optimize("Analyze this dataset.")
	assert.Contains(t, result.Techniques, TechniqueContextInjection)
}
