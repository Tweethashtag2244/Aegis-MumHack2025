
import { z } from 'zod';

// === AnalyzePerceptualForensics / runComputerVisionAnalysis ===
export const ComputerVisionInputSchema = z.object({
  videoDataUri: z
    .string()
    .describe(
      "A video as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
});
export type ComputerVisionInput = z.infer<typeof ComputerVisionInputSchema>;

export const ComputerVisionOutputSchema = z.object({
  perceptualScore: z
    .number()
    .describe(
      'A score representing the likelihood the video is real based on perceptual artifacts (0-1, 1 being highly likely real).'
    ),
  explanation: z
    .string()
    .describe('Explanation of the perceptual forensic analysis.'),
  forensicIndicators: z.object({
    avgFFTDeviation: z.number(),
    avgNoiseResidual: z.number(),
    modelConfidence: z.number(),
    spectralSlope: z.number(),
  }).describe('A JSON object of deterministic forensic indicators.'),
  anomalyFrames: z.any().describe('An array of frame numbers or timestamps with detected anomalies.'),
});
export type ComputerVisionOutput = z.infer<typeof ComputerVisionOutputSchema>;


// === AnalyzeSemanticCoherence ===
export const AnalyzeSemanticCoherenceInputSchema = z.object({
  videoDataUri: z
    .string()
    .describe(
      "A video as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
});
export type AnalyzeSemanticCoherenceInput = z.infer<typeof AnalyzeSemanticCoherenceInputSchema>;

export const AnalyzeSemanticCoherenceOutputSchema = z.object({
  semanticScore: z
    .number()
    .describe(
      'A score representing the semantic coherence between the video and its audio (0-1, 1 being highly coherent).'
    ),
  explanation: z.string().describe('Explanation of the semantic coherence score.'),
});
export type AnalyzeSemanticCoherenceOutput = z.infer<typeof AnalyzeSemanticCoherenceOutputSchema>;


// === GenerateReasoningExplanation ===
export const GenerateReasoningExplanationInputSchema = z.object({
    perceptualScore: z.number().describe('Perceptual Forensics Agent score (0–1).'),
    perceptualExplanation: z.string().describe('Explanation from the Perceptual Forensics Agent.'),
    semanticScore: z.number().describe('Semantic Coherence Agent score (0–1).'),
    semanticExplanation: z.string().describe('Explanation from the Semantic Coherence Agent.'),
    classification: z.string().describe("The final classification based on weighted scores."),
    finalScore: z.number().describe("The final weighted authenticity score."),
});
export type GenerateReasoningExplanationInput = z.infer<typeof GenerateReasoningExplanationInputSchema>;
  
export const GenerateReasoningExplanationOutputSchema = z.object({
    reasoningSummary: z.string().describe("The final, unified explanation for the classification."),
});
export type GenerateReasoningExplanationOutput = z.infer<typeof GenerateReasoningExplanationOutputSchema>;


// === FullAnalysis ===
export const FullAnalysisInputSchema = z.object({
    videoDataUri: z
      .string()
      .describe(
        "A video as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
      ),
    perceptualAnalysis: ComputerVisionOutputSchema.describe("The result from the computer vision analysis step."),
});
export type FullAnalysisInput = z.infer<typeof FullAnalysisInputSchema>;
  
export const FullAnalysisOutputSchema = z.object({
      authenticityScore: z.number(),
      classification: z.enum(['Likely Real', 'Likely AI-Generated', 'Uncertain / Mixed Evidence']),
      reasoningSummary: z.string(),
      perceptualScore: z.number(),
      perceptualExplanation: z.string(),
      semanticScore: z.number(),
      semanticExplanation: z.string(),
      forensicIndicators: z.any().optional(),
      anomalyFrames: z.any().optional(),
});
export type FullAnalysisOutput = z.infer<typeof FullAnalysisOutputSchema>;
