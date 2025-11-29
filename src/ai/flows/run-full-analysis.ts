'use server';

import { ai } from '@/ai/genkit';
import { analyzeSemanticCoherence } from './analyze-semantic-coherence';
import { generateReasoningExplanation } from './generate-reasoning-explanation';
import { runCollectiveConsensus } from './collective-consensus';
import {
  FullAnalysisInputSchema,
  type FullAnalysisInput,
  type FullAnalysisOutput,
  FullAnalysisOutputSchema,
  type ComputerVisionOutput
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
    // 1. Parse Python Results
    const cvResult = input.perceptualAnalysis as ComputerVisionOutput;
    
    const {
      perceptualScore,
      explanation: perceptualExplanation,
      forensicIndicators,
      anomalyFrames,
      provenance,
      attribution,
      adversarial
    } = cvResult;

    // 2. Run Semantic Analysis
    const { semanticScore, explanation: semanticExplanation } =
      await analyzeSemanticCoherence({
        videoDataUri: input.videoDataUri,
      });

    // 3. Run Collective Consensus
    const { consensusScore, explanation: consensusExplanation } = 
      await runCollectiveConsensus(semanticExplanation);

    // 4. Weighted Fusion Logic
    const W_PERCEPTUAL = 0.40;
    const W_SEMANTIC = 0.20;
    const W_PROVENANCE = 0.15;
    const W_ROBUSTNESS = 0.15;
    const W_CONSENSUS = 0.10;

    let authenticityScore =
      (perceptualScore * W_PERCEPTUAL) +
      (semanticScore * W_SEMANTIC) +
      (provenance.provenanceScore * W_PROVENANCE) +
      (adversarial.robustnessScore * W_ROBUSTNESS) +
      (consensusScore * W_CONSENSUS);

    // --- OVERRIDE LOGIC START ---

    // A. "The Sora Breaker" (Perceptual Veto)
    // ONLY trigger if score is extremely low (< 0.20) AND we don't have mobile provenance.
    // If it's a mobile file, low perceptual score usually just means "bad camera quality/low light".
    // We check if encoder is unknown AND provenance score is high (indicative of mobile file logic in python)
    const isMobile = provenance.encoder === "unknown" && provenance.provenanceScore > 0.9; 
    
    if (perceptualScore < 0.20 && !isMobile) {
        authenticityScore = Math.min(authenticityScore, 0.40);
        console.log("Veto Triggered: Low perceptual score on non-mobile video.");
    }

    // B. "The Real Video Saver"
    // If provenance detects a mobile filename pattern or C2PA, trust it more.
    if (provenance.hasC2PA || provenance.provenanceScore > 0.9) {
        // Boost significantly, as filename patterns are hard to fake accidentally
        authenticityScore = Math.max(authenticityScore, 0.75);
    }

    // C. Model Attribution Override (RELAXED)
    // Only cap if confidence is VERY high (> 0.75) AND it's not a generic "Sora?" guess.
    if (attribution.confidence > 0.75 && !attribution.detectedModel.includes("?")) {
        authenticityScore = Math.min(authenticityScore, 0.3);
    }
    
    // --- OVERRIDE LOGIC END ---

    // 5. Classification
    let classification: FullAnalysisOutput['classification'];
    if (authenticityScore >= 0.70) classification = 'Likely Real';
    else if (authenticityScore < 0.5) classification = 'Likely AI-Generated';
    else classification = 'Uncertain / Mixed Evidence';
    
    // 6. Reasoning
    const reasoningResult = await generateReasoningExplanation({
      perceptualScore,
      perceptualExplanation,
      semanticScore,
      semanticExplanation,
      classification,
      finalScore: authenticityScore,
      additionalContext: {
        provenance,
        attribution,
        adversarial,
        consensus: { score: consensusScore, explanation: consensusExplanation }
      }
    });

    return {
      authenticityScore: Number(authenticityScore.toFixed(2)),
      classification,
      reasoningSummary: reasoningResult.reasoningSummary,
      perceptualScore,
      semanticScore,
      provenanceScore: provenance.provenanceScore,
      consensusScore,
      perceptualExplanation,
      semanticExplanation,
      forensicIndicators,
      anomalyFrames,
      attribution,
      provenance,
    };
  }
);