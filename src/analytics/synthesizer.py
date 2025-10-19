"""Business insight synthesizer - converts statistical results to business insights."""

from typing import Dict, List, Optional, Any
from .base_models import (
    AnalysisResult,
    ForecastResult,
    CausalResult,
    VarianceDecompositionResult,
    AnomalyDetectionResult,
    ImpactEstimationResult,
    ConfidenceLevel
)
from src.utils.llm_client import get_llm
from langchain.prompts import PromptTemplate
import json


class BusinessInsightSynthesizer:
    """
    Converts statistical analysis results into human-readable business insights.

    Takes calculated numbers from fixed code and generates:
    - Plain English explanations
    - Business context
    - Actionable recommendations
    - "Why it matters" sections
    """

    def __init__(self, temperature: float = 0.3):
        """
        Initialize synthesizer.

        Args:
            temperature: LLM temperature for insight generation
        """
        self.llm = get_llm(temperature=temperature, model="gemini-2.5-flash")

    def synthesize_forecast(self, result: ForecastResult) -> str:
        """
        Generate business insight from forecast results.

        Args:
            result: ForecastResult with statistical forecasts

        Returns:
            Markdown-formatted business insight
        """
        # Extract key data
        results = result.results
        validation = result.validation
        context = result.context

        prompt = PromptTemplate.from_template("""
You are a business analyst converting forecasting results into actionable insights.

## Statistical Results (DO NOT show these raw numbers to users):
- Forecast: {predictions}
- Current Value: {current_value}
- Growth Rate: {growth_rate}%
- Confidence Interval: {confidence_interval}
- Model: {model_name}
- Accuracy (MAPE): {mape}%
- Confidence Level: {confidence_level}
- Sample Size: {sample_size}

## Business Context:
{context}

## Your Task:
Write a business insight in markdown format with:

### Insight Title: [Clear, Business-Friendly Title]

[2-3 sentences explaining what the forecast shows in plain English]

**Confidence Level**: [HIGH/MEDIUM/LOW] - [One sentence explaining why]
- [If LOW confidence, explain sample size or data quality issues]

**Why It Matters**: [2 sentences on business impact]

**Recommendation**: [Specific actionable next steps]

**Key Numbers**:
- Forecast: [value with context]
- Expected Range: [min]-[max] (95% confidence)
- Growth Rate: [X%] vs current period

IMPORTANT:
- Write for business executives, not statisticians
- NO p-values, NO ARIMA parameters, NO technical jargon
- Focus on "what does this mean?" and "what should we do?"
- Use specific numbers but explain what they mean
- Be concise (max 200 words)
""")

        # Format confidence interval
        ci = results.get('confidence_intervals', [(0, 0)])[0] if results.get('confidence_intervals') else (0, 0)

        input_data = {
            'predictions': results.get('predictions', []),
            'current_value': results.get('current_value', 0),
            'growth_rate': results.get('growth_rate', 0),
            'confidence_interval': ci,
            'model_name': results.get('model_name', 'UNKNOWN'),
            'mape': results.get('accuracy_metrics', {}).get('mape', 0),
            'confidence_level': validation.confidence_level.value,
            'sample_size': validation.sample_size,
            'context': json.dumps(context, indent=2) if context else "No additional context"
        }

        response = self.llm.invoke(prompt.format(**input_data))
        return response.content

    def synthesize_causal(self, result: CausalResult) -> str:
        """Generate business insight from causal analysis results."""
        results = result.results
        validation = result.validation
        context = result.context

        relationships = results.get('relationships', [])
        if not relationships:
            return "### No Causal Relationships Found\n\nNo statistically significant causal relationships were detected in the data."

        prompt = PromptTemplate.from_template("""
You are a business analyst converting causal analysis results into actionable insights.

## Statistical Results (DO NOT show raw statistics):
{relationships}

## Confidence Level: {confidence_level}
## Sample Size: {sample_size}

## Business Context:
{context}

## Your Task:
Write a business insight explaining the causal relationship:

### Insight Title: [What Causes What?]

[2-3 sentences explaining the causal relationship in plain English]

**Strength**: [{strength}] - [Explain what this means]

**Time Lag**: [Explain how long it takes for the effect to appear]

**Why It Matters**: [Business implications]

**Recommendation**: [How to use this insight - specific actions]

**Key Finding**:
- When [cause] changes, [effect] changes [time lag] later
- Effect size: [quantify if possible]

IMPORTANT:
- Explain causality clearly ("X causes Y", not "X correlates with Y")
- NO p-values or F-statistics in the insight
- Focus on business implications and actions
- Be specific about timing (lag period)
""")

        rel = relationships[0]
        input_data = {
            'relationships': json.dumps(relationships, indent=2),
            'confidence_level': validation.confidence_level.value,
            'sample_size': validation.sample_size,
            'context': json.dumps(context, indent=2) if context else "No additional context",
            'strength': rel.get('strength', 'unknown').upper()
        }

        response = self.llm.invoke(prompt.format(**input_data))
        return response.content

    def synthesize_variance_decomposition(self, result: VarianceDecompositionResult) -> str:
        """Generate business insight from variance decomposition results."""
        results = result.results
        validation = result.validation
        context = result.context

        contributions = results.get('feature_contributions', [])
        top_3 = contributions[:3] if len(contributions) >= 3 else contributions

        prompt = PromptTemplate.from_template("""
You are a business analyst explaining what drives outcomes in the business.

## Statistical Results (DO NOT show raw percentages):
Top Contributing Factors:
{top_factors}

Total Variance Explained: {variance_explained}%
Method: {method}

## Confidence Level: {confidence_level}

## Business Context:
{context}

## Your Task:
Write a business insight explaining what drives the outcome:

### Insight Title: Top Drivers of [Outcome]

[Opening paragraph explaining what the analysis reveals]

**Top 3 Factors**:
1. **[Factor 1]** - [Explain its contribution and what it means]
2. **[Factor 2]** - [Explain its contribution and what it means]
3. **[Factor 3]** - [Explain its contribution and what it means]

Together, these three factors explain [X]% of why [outcome] varies.

**Why It Matters**: [Business implications]

**Recommendation**: [Which factors to focus on and why]

IMPORTANT:
- Write for business stakeholders
- Explain what each factor means in practical terms
- NO SHAP values, NO technical method details
- Focus on actionable insights
- Use analogies if helpful (e.g., "like a pie chart of influence")
""")

        input_data = {
            'top_factors': json.dumps(top_3, indent=2),
            'variance_explained': results.get('total_variance_explained', 0) * 100,
            'method': results.get('method', 'UNKNOWN'),
            'confidence_level': validation.confidence_level.value,
            'context': json.dumps(context, indent=2) if context else "No additional context"
        }

        response = self.llm.invoke(prompt.format(**input_data))
        return response.content

    def synthesize_anomalies(self, result: AnomalyDetectionResult) -> str:
        """Generate business insight from anomaly detection results."""
        results = result.results
        validation = result.validation
        context = result.context

        anomalies = results.get('anomalies', [])
        total = results.get('total_anomalies', 0)

        if total == 0:
            return "### No Anomalies Detected\n\nNo unusual patterns or outliers were found in the data."

        # Get top 5 anomalies by severity
        sorted_anomalies = sorted(
            anomalies,
            key=lambda x: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(x['severity'], 0),
            reverse=True
        )[:5]

        prompt = PromptTemplate.from_template("""
You are a business analyst explaining unusual patterns in the data.

## Statistical Results:
Total Anomalies: {total_anomalies}
Anomaly Rate: {anomaly_rate}%
Detection Method: {method}

Top Anomalies:
{top_anomalies}

Severity Breakdown: {severity_breakdown}

## Confidence Level: {confidence_level}

## Business Context:
{context}

## Your Task:
Write a business insight about the anomalies:

### Insight Title: [X] Unusual Patterns Detected

[Opening paragraph explaining what was found]

**Most Significant Anomalies**:
[List 3-5 most important anomalies with:
- When they occurred
- What the unusual value was
- How far from normal
- Possible business explanation]

**Why It Matters**: [What these anomalies might indicate]

**Recommendation**: [What to investigate or do next]

IMPORTANT:
- Focus on business implications, not statistical methods
- Suggest possible business explanations (campaign, holiday, error, etc.)
- Prioritize by severity
- Be specific about dates/times
- NO z-scores or technical details
""")

        input_data = {
            'total_anomalies': total,
            'anomaly_rate': results.get('anomaly_rate', 0) * 100,
            'method': results.get('detection_method', 'UNKNOWN'),
            'top_anomalies': json.dumps(sorted_anomalies, indent=2),
            'severity_breakdown': json.dumps(results.get('severity_breakdown', {}), indent=2),
            'confidence_level': validation.confidence_level.value,
            'context': json.dumps(context, indent=2) if context else "No additional context"
        }

        response = self.llm.invoke(prompt.format(**input_data))
        return response.content

    def synthesize_impact(self, result: ImpactEstimationResult) -> str:
        """Generate business insight from impact estimation results."""
        results = result.results
        validation = result.validation
        context = result.context

        prompt = PromptTemplate.from_template("""
You are a business analyst evaluating the impact of an intervention or change.

## Statistical Results:
Intervention: {intervention_name}
Estimated Impact: {impact}
Percent Change: {percent_change}%
Confidence Interval: {confidence_interval}
Statistical Significance: {is_significant}
Effect Size: {effect_size}

Before: Mean={before_mean}, N={n_before}
After: Mean={after_mean}, N={n_after}

P-value: {p_value}
Cohen's d: {cohens_d}

## Confidence Level: {confidence_level}

## Business Context:
{context}

## Your Task:
Write a business insight about the impact:

### Insight Title: [Intervention Name] Impact: [Positive/Negative/No Change]

[Opening paragraph explaining what happened after the intervention]

**Impact Size**: {effect_size_word} effect
- Before: [average value]
- After: [average value]
- Change: [X] ({percent_change}%)

**Confidence**: [{confidence_level}]
- {confidence_explanation}

**Statistical Significance**: {significance_explanation}

**Why It Matters**: [Business implications of this impact]

**Recommendation**: [Should we continue/scale/modify the intervention?]

IMPORTANT:
- Write for business decision-makers
- Explain if the change is meaningful (beyond just statistical significance)
- Be honest about confidence level and sample size limitations
- NO p-values, NO Cohen's d in the insight
- Focus on "was it worth it?" and "should we do it again?"
""")

        # Prepare explanations
        is_significant = results.get('is_significant', False)
        significance_explanation = (
            "Yes - This change is statistically significant and unlikely due to chance"
            if is_significant else
            "No - This change could be due to random variation"
        )

        confidence_explanation = self._get_confidence_explanation(
            validation.confidence_level,
            validation.sample_size
        )

        effect_size_word = results.get('effect_size', 'unknown').capitalize()

        input_data = {
            'intervention_name': results.get('intervention_name', 'Intervention'),
            'impact': results.get('estimated_impact', 0),
            'percent_change': results.get('percent_change', 0),
            'confidence_interval': results.get('confidence_interval', (0, 0)),
            'is_significant': is_significant,
            'effect_size': results.get('effect_size', 'unknown'),
            'effect_size_word': effect_size_word,
            'before_mean': results.get('before_mean', 0),
            'after_mean': results.get('after_mean', 0),
            'n_before': results.get('n_before', 0),
            'n_after': results.get('n_after', 0),
            'p_value': results.get('p_value', 1.0),
            'cohens_d': results.get('cohens_d', 0),
            'confidence_level': validation.confidence_level.value,
            'confidence_explanation': confidence_explanation,
            'significance_explanation': significance_explanation,
            'context': json.dumps(context, indent=2) if context else "No additional context"
        }

        response = self.llm.invoke(prompt.format(**input_data))
        return response.content

    def _get_confidence_explanation(
        self,
        confidence_level: ConfidenceLevel,
        sample_size: int
    ) -> str:
        """Get human-readable explanation of confidence level."""
        if confidence_level == ConfidenceLevel.VERY_HIGH:
            return f"Very high confidence based on {sample_size} data points"
        elif confidence_level == ConfidenceLevel.HIGH:
            return f"High confidence with adequate sample size ({sample_size} points)"
        elif confidence_level == ConfidenceLevel.MEDIUM:
            return f"Moderate confidence - results are reasonable but sample size could be larger ({sample_size} points)"
        elif confidence_level == ConfidenceLevel.LOW:
            return f"Low confidence due to small sample size ({sample_size} points) - interpret with caution"
        else:
            return f"Very low confidence - sample size is too small ({sample_size} points) for robust conclusions"
