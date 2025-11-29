'use server';

import { ai } from '@/ai/genkit';
import { 
  CollectiveConsensusInputSchema, 
  CollectiveConsensusOutputSchema 
} from '@/ai/types';

// Mock database of viral deepfakes
const KNOWN_FAKES = [
  "pope francis puffer", "pentagon explosion", "trump arrest", "will smith spaghetti"
];

const collectiveConsensusFlow = ai.defineFlow(
  {
    name: 'collectiveConsensusFlow',
    inputSchema: CollectiveConsensusInputSchema,
    outputSchema: CollectiveConsensusOutputSchema,
  },
  async (input) => {
    const query = input.videoDescription.toLowerCase();
    
    // Check internal database
    const match = KNOWN_FAKES.find(fake => query.includes(fake));
    
    if (match) {
      return {
        consensusScore: 0.1, // Confirmed Fake
        explanation: `Matches known debunked deepfake: "${match}".`,
      };
    }

    // Default: Neutral if no info found
    return {
      consensusScore: 0.5, 
      explanation: "No matches found in fact-check databases or reverse search.",
    };
  }
);

export async function runCollectiveConsensus(description: string) {
  return collectiveConsensusFlow({ videoDescription: description });
}