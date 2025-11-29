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
    // 1. Parse Python Results (Robust Fallbacks)
    const cvResult = input.perceptualAnalysis as ComputerVisionOutput;
    
    // Default Fallbacks
    const perceptualScore = cvResult.perceptualScore ?? 0.5;
    const provenance = cvResult.provenance ?? { provenanceScore: 0.5, encoder: 'unknown', hasC2PA: false, isMobileFilename: false, isSocialFilename: false };
    const attribution = cvResult.attribution ?? { detectedModel: 'Unknown', confidence: 0 };
    const adversarial = cvResult.adversarial ?? { robustnessScore: 0.5, explanation: 'N/A' };
    const anomalyFrames = cvResult.anomalyFrames ?? [];
    const forensicIndicators = cvResult.forensicIndicators ?? { avgNoiseResidual: 0, modelConfidence: 0, spectralSlope: 0 };
    const perceptualExplanation = cvResult.explanation ?? "Perceptual analysis completed.";

    // 2. Run Semantic Analysis (SAFE MODE)
    let semanticResult;
    try {
        semanticResult = await analyzeSemanticCoherence({
            videoDataUri: input.videoDataUri,
        });
    } catch (e) {
        console.error("Semantic Agent Failed:", e);
        semanticResult = { semanticScore: 0.5, explanation: "Semantic analysis unavailable (Service Error)." };
    }
    const { semanticScore, explanation: semanticExplanation } = semanticResult;

    // 3. Run Collective Consensus (SAFE MODE)
    let consensusResult;
    try {
        consensusResult = await runCollectiveConsensus(semanticExplanation);
    } catch (e) {
        console.error("Consensus Agent Failed:", e);
        consensusResult = { consensusScore: 0.5, explanation: "Consensus check skipped." };
    }
    const { consensusScore, explanation: consensusExplanation } = consensusResult;

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

    // 1. Identify Source Trust
    const isMobile = provenance.isMobileFilename; // VID_2025...
    const isSocial = provenance.isSocialFilename; // WhatsApp Video...
    const isTrustedSource = isMobile || isSocial;

    // A. "The Sora/Veo Breaker"
    // If NOT trusted source and visually looks fake -> Kill it.
    if (perceptualScore < 0.25 && !isTrustedSource) {
        authenticityScore = Math.min(authenticityScore, 0.40);
        console.log("Veto: Low perceptual score on non-trusted video.");
    }

    // B. "The Decoy Trap" (Strictly for SOCIAL media)
    // Real Social Media videos (WhatsApp/TikTok) are heavily compressed (< 0.08 score).
    // If a "WhatsApp" video has a "Moderate" score (0.1 - 0.3), it's suspicious (Renamed AI video).
    // FIX: We do NOT apply this to 'isMobile' because raw phone videos CAN be 0.2-0.3 depending on lighting.
    if (isSocial && perceptualScore > 0.08 && perceptualScore < 0.30) {
         console.log("Decoy Trap Triggered: Social filename but suspicious score.");
         authenticityScore = Math.min(authenticityScore, 0.45);
    } 
    
    // C. "The Real Video Saver" (Mobile & Social Boost)
    // If it survived the Decoy Trap (or is Mobile), AND Semantics are good, we trust the filename.
    else if (isTrustedSource && semanticScore > 0.70) {
        // We give a slightly higher boost to Mobile files since they are harder to fake than renaming a file
        const boostTarget = isMobile ? 0.80 : 0.75;
        authenticityScore = Math.max(authenticityScore, boostTarget);
    }

    // D. "Known AI Encoder Trap"
    if (provenance.provenanceScore < 0.35) {
        authenticityScore = Math.min(authenticityScore, 0.30);
    }

    // E. "C2PA Verified" (Absolute Trust)
    if (provenance.hasC2PA) {
         authenticityScore = 0.99;
    }

    // F. Model Attribution Override
    if (attribution.confidence > 0.75 && !attribution.detectedModel.includes("?")) {
        authenticityScore = Math.min(authenticityScore, 0.3);
    }
    
    // --- OVERRIDE LOGIC END ---

    // 5. Classification
    let classification: FullAnalysisOutput['classification'];
    if (authenticityScore >= 0.70) classification = 'Likely Real';
    else if (authenticityScore < 0.5) classification = 'Likely AI-Generated';
    else classification = 'Uncertain / Mixed Evidence';
    
    // 6. Reasoning (SAFE MODE)
    let reasoningSummary = `Analysis Complete. Classification: ${classification} (${(authenticityScore*100).toFixed(0)}%).`;
    
    try {
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
        
        if (reasoningResult && reasoningResult.reasoningSummary) {
             reasoningSummary = reasoningResult.reasoningSummary;
        }
    } catch (e) {
        console.error("Reasoning Agent Failed (LLM Error):", e);
        reasoningSummary += " (Detailed AI reasoning currently unavailable).";
    }

    return {
      authenticityScore: Number(authenticityScore.toFixed(2)),
      classification,
      reasoningSummary,
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