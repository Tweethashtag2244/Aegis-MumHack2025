
'use server';

import { ai } from '@/ai/genkit';
import { analyzeSemanticCoherence } from './analyze-semantic-coherence';
import { generateReasoningExplanation } from './generate-reasoning-explanation';
import {
  FullAnalysisInputSchema,
  type FullAnalysisInput,
  type FullAnalysisOutput,
  FullAnalysisOutputSchema,
} from '@/ai/types';

export async function runFullAnalysis(
  input: FullAnalysisInput
): Promise<FullAnalysisOutput> {
  return runFullAnalysisFlow(input);
}

const runFullAnalysisFlow = ai.defineFlow(
  {
    name: 'runFullAnalysisFlow',
    inputSchema: FullAnalysisInputSchema,
    outputSchema: FullAnalysisOutputSchema,
  },
  async (input) => {
    // --- Step 0: Configurable weights ---
    // Balances the sophisticated perceptual analysis from the Python script
    // with the semantic analysis from the Genkit flow.
    const WEIGHT_PERCEPTUAL = 0.85;
    const WEIGHT_SEMANTIC = 0.15;

    // --- Step 1: Perceptual analysis is passed in from the client ---
    // This result comes directly from the local Python/Flask server.
    const {
      perceptualScore,
      explanation: perceptualExplanation,
      forensicIndicators,
      anomalyFrames,
    } = input.perceptualAnalysis;

    // --- Step 2: Run semantic coherence analysis ---
    const { semanticScore, explanation: semanticExplanation } =
      await analyzeSemanticCoherence({
        videoDataUri: input.videoDataUri,
      });

    // --- Step 3: Compute final authenticity score ---
    const authenticityScore =
      WEIGHT_PERCEPTUAL * perceptualScore +
      WEIGHT_SEMANTIC * semanticScore;

    // --- Step 4: Classify based on authenticity likelihood ---
    let classification: FullAnalysisOutput['classification'];
    if (authenticityScore >= 0.7) classification = 'Likely Real';
    else if (authenticityScore < 0.5) classification = 'Likely AI-Generated';
    else classification = 'Uncertain / Mixed Evidence';

    // --- Step 5: Generate reasoning explanation ---
    const reasoningResult = await generateReasoningExplanation({
      perceptualScore,
      perceptualExplanation,
      semanticScore,
      semanticExplanation,
      classification,
      finalScore: authenticityScore,
    });

    // --- Step 6: Return final structured output ---
    return {
      authenticityScore,
      classification,
      reasoningSummary: reasoningResult.reasoningSummary,
      perceptualScore,
      perceptualExplanation,
      semanticScore,
      semanticExplanation,
      forensicIndicators,
      anomalyFrames,
    };
  }
);

export { FullAnalysisOutput };
