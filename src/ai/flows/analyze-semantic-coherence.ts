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
  config: { 
    temperature: 0.2, // Lower temperature for more consistent/conservative scoring
    topP: 0.8, 
    maxOutputTokens: 1024 
  },
  prompt: `
You are an expert digital forensics AI analyzing the coherence between a video's visual content and its audio track.

### YOUR GOAL:
Determine if the audio and video serve as evidence of *AI Generation* or *Real-World Recording*.

### CRITICAL RULES (READ CAREFULLY):
1. **Real World is messy:** Real videos often have background TV, traffic, wind, or conversations that are *unrelated* to the visual subject (e.g., filming a cat while a news report plays on TV). This is **NORMAL** and should receive a HIGH score (0.8+).
2. **AI Contradictions are physical:** A video is likely AI-generated only if there is a **physical violation of causality**.
   - Example (FAKE): A person on screen is clearly moving their lips to speak, but the audio is silent or playing music.
   - Example (FAKE): A dog opens its mouth to bark, but a human voice comes out.
   - Example (REAL): A cat is playing, and a human voice is heard talking about something unrelated. (This is just background noise).

### Scoring Rubric (0.0 - 1.0):
- **0.8 - 1.0 (Likely Real):** Perfect sync OR plausible background noise. Even if the audio is "unrelated" to the visual action, if it sounds like a real environment (TV, street noise), score it high.
- **0.5 - 0.7 (Uncertain):** Audio is ambiguous, low quality, or generic music overlay.
- **0.0 - 0.4 (Likely Fake):** **Explicit Mismatch.** Lip-sync failure, wrong object sounds (car making horse noise), or eerie silence in a loud visual environment.

### Video to Analyze:
{{media url=videoDataUri}}

### Output:
Provide a JSON object with:
- "semanticScore": <number between 0.0 and 1.0>
- "explanation": "<concise reasoning>"
`,
});

const analyzeSemanticCoherenceFlow = ai.defineFlow(
  {
    name: 'analyzeSemanticCoherenceFlow',
    inputSchema: AnalyzeSemanticCoherenceInputSchema,
    outputSchema: AnalyzeSemanticCoherenceOutputSchema,
  },
  async input => {
    try {
      const response = await analyzeSemanticCoherencePrompt(input);
      
      // Check if output is null or undefined (Fix for INVALID_ARGUMENT error)
      if (!response || !response.output) {
        console.warn("Semantic Agent returned null output. Using fallback.");
        return {
          semanticScore: 0.5,
          explanation: "Semantic analysis inconclusive (AI model returned empty response). Defaulting to neutral score."
        };
      }
      
      return response.output;
    } catch (e) {
      console.error("Semantic Analysis Failed:", e);
      // Fallback to neutral if analysis fails completely
      return {
        semanticScore: 0.5,
        explanation: "Semantic analysis failed due to a service error. Defaulting to neutral score."
      };
    }
  }
);