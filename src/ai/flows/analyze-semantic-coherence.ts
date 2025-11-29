'use server';

/**
 * @fileOverview Analyzes the semantic coherence between video content and its audio track.
 */

import {ai} from '@/ai/genkit';
import {
  AnalyzeSemanticCoherenceInputSchema,
  type AnalyzeSemanticCoherenceInput,
  AnalyzeSemanticCoherenceOutputSchema,
  type AnalyzeSemanticCoherenceOutput,
} from '@/ai/types';

export async function analyzeSemanticCoherence(
  input: AnalyzeSemanticCoherenceInput
): Promise<AnalyzeSemanticCoherenceOutput> {
  return analyzeSemanticCoherenceFlow(input);
}

const analyzeSemanticCoherencePrompt = ai.definePrompt({
  name: 'analyzeSemanticCoherencePrompt',
  input: { schema: AnalyzeSemanticCoherenceInputSchema },
  output: { schema: AnalyzeSemanticCoherenceOutputSchema },
  model: 'googleai/gemini-2.5-flash',
  config: { temperature: 0.3, topP: 0.9, maxOutputTokens: 1024 },
  prompt: `
You are an AI expert analyzing the semantic coherence between a video's visual content and its complete audio track (speech, ambient sounds, etc.).

### Instructions:
1.  **Assess Overall Coherence:** Evaluate if the sounds (both speech and environmental) logically match the visual events.
2.  **Handle Real-World Noise:** **Crucially, distinguish between "contradiction" and "background noise."**
    - *Real videos* often have unrelated background audio (TV playing, people talking off-camera, wind, traffic). This is NORMAL. Do NOT penalize this unless it defies physics (e.g., a person on screen moving lips but no sound comes out).
    - *AI videos* often have "hallucinated" sounds or eerie silence.
3.  **Generate Score (0.0 - 1.0):**
    - **0.8 - 1.0:** Clear match or plausible background noise.
    - **0.5 - 0.7:** Audio is messy or unrelated (e.g., music overlay), but not physically impossible.
    - **0.0 - 0.4:** Direct Physical Contradiction (e.g., dog barking but visual is a cat, lip-sync completely broken).

### Required Output Format (JSON):
{
  "semanticScore": <0.0–1.0>,
  "explanation": "<your concise explanation>"
}

### Video to Analyze:
{{media url=videoDataUri}}
`,
});

const analyzeSemanticCoherenceFlow = ai.defineFlow(
  {
    name: 'analyzeSemanticCoherenceFlow',
    inputSchema: AnalyzeSemanticCoherenceInputSchema,
    outputSchema: AnalyzeSemanticCoherenceOutputSchema,
  },
  async input => {
    const {output} = await analyzeSemanticCoherencePrompt(input);
    return output!;
  }
);