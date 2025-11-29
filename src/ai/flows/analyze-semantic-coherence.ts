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
2.  **Assess Audio-Visual Sync:** Pay close attention to lip-sync for speech and the timing of environmental sounds with visual cues.
3.  **Generate Score:** Based on your assessment, provide a 'semanticScore' from 0.0 to 1.0.
4.  **Generate Explanation:** Write a brief explanation for your score, referencing specific visual and audio cues.

### Scoring Rubric:
- **0.8 - 1.0 (High Coherence):** Excellent sync, speech is relevant to the visuals, and ambient sounds are perfectly appropriate for the environment.
- **0.5 - 0.7 (Moderate Coherence):** Minor issues. Sync might be slightly off, or some background sounds may seem out of place, but it's not overtly contradictory.
- **0.0 - 0.4 (Low Coherence):** Clear contradictions. Speech does not match the scene, lip-sync is obviously wrong, or environmental sounds are illogical (e.g., city traffic in a forest). Never output 0 unless there is a blatant and undeniable contradiction.

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
