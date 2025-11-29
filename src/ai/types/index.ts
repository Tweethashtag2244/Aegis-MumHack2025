import { z } from 'zod';

// === 1. Computer Vision / Perceptual Agent Output ===
export const ComputerVisionOutputSchema = z.object({
  perceptualScore: z.number().describe('Score 0-1 (1 = Real).'),
  explanation: z.string(),
  forensicIndicators: z.object({
    avgNoiseResidual: z.number(),
    modelConfidence: z.number(),
    spectralSlope: z.number(),
  }),
  anomalyFrames: z.any(),
  
  // NEW: Agent Data Structures
  provenance: z.object({
    provenanceScore: z.number(),
    encoder: z.string(),
    hasC2PA: z.boolean(),
    isMobileFilename: z.boolean().optional(),
    creationTime: z.string().nullable().optional(),
  }),
  attribution: z.object({
    detectedModel: z.string(),
    confidence: z.number(),
  }),
  adversarial: z.object({
    robustnessScore: z.number(),
    explanation: z.string(),
  }),
});
export type ComputerVisionOutput = z.infer<typeof ComputerVisionOutputSchema>;


// === 2. Collective Consensus Agent ===
export const CollectiveConsensusInputSchema = z.object({
  videoDescription: z.string(),
});
// Added missing type export
export type CollectiveConsensusInput = z.infer<typeof CollectiveConsensusInputSchema>;

export const CollectiveConsensusOutputSchema = z.object({
  consensusScore: z.number(),
  explanation: z.string(),
});
export type CollectiveConsensusOutput = z.infer<typeof CollectiveConsensusOutputSchema>;


// === 3. Semantic Agent ===
export const AnalyzeSemanticCoherenceInputSchema = z.object({
  videoDataUri: z.string(),
});
// Added missing type export (This was the cause of your error)
export type AnalyzeSemanticCoherenceInput = z.infer<typeof AnalyzeSemanticCoherenceInputSchema>;

export const AnalyzeSemanticCoherenceOutputSchema = z.object({
  semanticScore: z.number(),
  explanation: z.string(),
});
export type AnalyzeSemanticCoherenceOutput = z.infer<typeof AnalyzeSemanticCoherenceOutputSchema>;


// === 4. Reasoning Agent ===
export const GenerateReasoningExplanationInputSchema = z.object({
    perceptualScore: z.number(),
    perceptualExplanation: z.string(),
    semanticScore: z.number(),
    semanticExplanation: z.string(),
    // Allow complex JSON objects for the new agents
    additionalContext: z.any().optional(), 
    classification: z.string(),
    finalScore: z.number(),
});
export type GenerateReasoningExplanationInput = z.infer<typeof GenerateReasoningExplanationInputSchema>;
  
export const GenerateReasoningExplanationOutputSchema = z.object({
    reasoningSummary: z.string(),
});
export type GenerateReasoningExplanationOutput = z.infer<typeof GenerateReasoningExplanationOutputSchema>;


// === 5. Full Analysis Orchestrator ===
export const FullAnalysisInputSchema = z.object({
    videoDataUri: z.string(),
    perceptualAnalysis: z.any(), 
});
export type FullAnalysisInput = z.infer<typeof FullAnalysisInputSchema>;
  
export const FullAnalysisOutputSchema = z.object({
      authenticityScore: z.number(),
      classification: z.enum(['Likely Real', 'Likely AI-Generated', 'Uncertain / Mixed Evidence', 'Verified Real']),
      reasoningSummary: z.string(),
      
      // Detailed scores for UI
      perceptualScore: z.number(),
      semanticScore: z.number(),
      provenanceScore: z.number(),
      consensusScore: z.number(),

      // Agent Details
      perceptualExplanation: z.string(),
      semanticExplanation: z.string(),
      forensicIndicators: z.any().optional(),
      anomalyFrames: z.any().optional(),
      attribution: z.any().optional(),
      provenance: z.any().optional(),
});
export type FullAnalysisOutput = z.infer<typeof FullAnalysisOutputSchema>;