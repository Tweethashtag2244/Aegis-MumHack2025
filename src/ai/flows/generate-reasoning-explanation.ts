'use server';

import { ai } from '@/ai/genkit';
import {
  GenerateReasoningExplanationInputSchema,
  type GenerateReasoningExplanationInput,
  GenerateReasoningExplanationOutputSchema,
  type GenerateReasoningExplanationOutput,
} from '@/ai/types';

export async function generateReasoningExplanation(
  input: GenerateReasoningExplanationInput
): Promise<GenerateReasoningExplanationOutput> {
  return generateReasoningExplanationFlow(input);
}

const generateReasoningExplanationPrompt = ai.definePrompt({
  name: 'generateReasoningExplanationPrompt',
  input: { schema: GenerateReasoningExplanationInputSchema },
  output: { schema: GenerateReasoningExplanationOutputSchema },
  model: 'googleai/gemini-2.5-flash',
  config: { temperature: 0.4, topP: 0.9, maxOutputTokens: 1024 },
  prompt: `
You are the "Aegis" Digital Forensics AI. synthesize data from multiple detection agents into a clear verdict.

### Input Data:
1. **Perceptual Analysis (Visual Artifacts):**
   - Score: {{perceptualScore}} / 1.0 (Higher is Real)
   - Details: {{perceptualExplanation}}

2. **Semantic Coherence (Audio/Visual Logic):**
   - Score: {{semanticScore}} / 1.0 (Higher is Real)
   - Details: {{semanticExplanation}}

3. **Advanced Forensic Signals:**
   {{#if additionalContext}}
   - **Provenance:** Score {{additionalContext.provenance.provenanceScore}}/1.0. Encoder: {{additionalContext.provenance.encoder}}.
   - **Model Attribution:** Detected "{{additionalContext.attribution.detectedModel}}" (Confidence: {{additionalContext.attribution.confidence}}).
   - **Adversarial Robustness:** Score {{additionalContext.adversarial.robustnessScore}}/1.0. (High = Robust, Low = Fragile).
   - **Collective Consensus:** Score {{additionalContext.consensus.score}}/1.0. Note: {{additionalContext.consensus.explanation}}.
   {{/if}}

### Final Verdict:
- **Classification:** {{classification}}
- **Authenticity Score:** {{finalScore}} / 1.0

### Instructions:
Write a **concise, 2-paragraph reasoning summary**.
- **Paragraph 1:** Explain the classification. Mention the strongest evidence (e.g., "Specific artifacts from the Sora model were detected" or "Semantic lighting inconsistencies").
- **Paragraph 2:** Comment on the metadata/provenance and consensus. Is the file header suspicious? Did online consensus verify it?
- **Tone:** Professional, objective, and authoritative.

  `,
});

const generateReasoningExplanationFlow = ai.defineFlow(
  {
    name: 'generateReasoningExplanationFlow',
    inputSchema: GenerateReasoningExplanationInputSchema,
    outputSchema: GenerateReasoningExplanationOutputSchema,
  },
  async (input) => {
    try {
      const result = await generateReasoningExplanationPrompt(input);
      if (!result || !result.output) throw new Error('LLM response missing.');
      return result.output;
    } catch (error) {
      console.error('LLM Error:', error);
      return {
        reasoningSummary: `Analysis complete. Classification: ${input.classification}. (Detailed summary unavailable due to API timeout).`,
      };
    }
  }
);