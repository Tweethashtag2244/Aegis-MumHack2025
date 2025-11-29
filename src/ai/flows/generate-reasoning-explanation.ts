'use server';

/**
 * Reasoning Explanation Agent — synthesizes weighted consensus.
 */

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
  prompt: `
You are the Reasoning Explanation Agent for Aegis.

Your task is to provide a clear, user-friendly explanation for the video's authenticity classification based on the provided analysis from multiple expert agents.

### Instructions:
1.  State the final classification provided (Likely Real, AI-Generated, or Uncertain).
2.  Synthesize the key findings from the Perceptual and Semantic agents into a cohesive narrative.
3.  Start with the most impactful finding. For example, if perceptual analysis found strong artifacts, lead with that. If the video was perceptually clean but semantically incoherent, explain that contrast.
4.  Keep the reasoning concise (2-4 sentences), factual, and easy for a non-expert to understand. Do not simply repeat the agent explanations; interpret and summarize them.

### Inputs:
- Classification: {{classification}} (based on a final score of {{finalScore}})
- Perceptual Analysis: {{perceptualScore}}, "{{perceptualExplanation}}"
- Semantic Analysis: {{semanticScore}}, "{{semanticExplanation}}"

Generate the 'reasoningSummary' accordingly.
`,
  config: {
    temperature: 0.3,
    topP: 0.9,
    maxOutputTokens: 1024,
    safetySettings: [
      {
        category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
        threshold: 'BLOCK_ONLY_HIGH',
      },
    ],
  },
});

const generateReasoningExplanationFlow = ai.defineFlow(
  {
    name: 'generateReasoningExplanationFlow',
    inputSchema: GenerateReasoningExplanationInputSchema,
    outputSchema: GenerateReasoningExplanationOutputSchema,
  },
  async input => {
    try {
      // Safely assign the result of the prompt call.
      const result = await generateReasoningExplanationPrompt(input);
      
      // Check if the result object is null/undefined OR if the structured 'output' property is missing.
      if (!result || !result.output) {
        throw new Error('LLM response was null or missing structured output after prompt execution.');
      }
      
      // If successful, return the structured output.
      return result.output;
    } catch (error) {
      // Catch any error thrown by the Genkit runtime (API failure, timeout, etc.)
      console.error('Error during LLM call for reasoning explanation:', error);
      
      // FIX: Instead of re-throwing, return a fallback object that meets the schema requirement.
      return {
        reasoningSummary: `[System Failure] Could not generate a detailed summary due to an internal API or schema validation error. The final classification is ${input.classification}.`,
      } as GenerateReasoningExplanationOutput;
    }
  }
);