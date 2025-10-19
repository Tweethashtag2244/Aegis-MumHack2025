"use client";

import type { FC } from 'react';
import type { AnalysisResult } from './aegis-dashboard';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { ScoreGauge } from './score-gauge';
import { Bot, Scale, CheckCircle, BrainCircuit, Microscope, AlertTriangle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Button } from './ui/button';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';

interface AnalysisResultsDisplayProps {
  result: AnalysisResult;
  onFeedback: (label: 'Real' | 'AI-Generated') => void;
}

const getClassificationClasses = (classification: AnalysisResult['classification']) => {
  switch (classification) {
    case 'Likely Real':
      return 'bg-primary/10 text-primary border-primary/20';
    case 'Likely AI-Generated':
      return 'bg-destructive/10 text-destructive border-destructive/20';
    case 'Uncertain / Mixed Evidence':
      return 'bg-accent/10 text-accent-foreground border-accent/20';
    default:
      return 'bg-muted text-muted-foreground border-border';
  }
};

const AnalysisResultsDisplay: FC<AnalysisResultsDisplayProps> = ({ result, onFeedback }) => {
  const classificationClasses = getClassificationClasses(result.classification);
  const authenticityScore = Math.round(result.authenticityScore * 100);

  const hasForensicData = result.forensicIndicators && Object.keys(result.forensicIndicators).length > 0;

  return (
    <div className="grid gap-6">
      <Card className="overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 bg-muted/50">
          <div>
            <CardTitle>Analysis Complete</CardTitle>
            <CardDescription className="mt-1">
              Results for <span className="font-medium text-foreground">{result.fileName}</span>
            </CardDescription>
          </div>
          <Badge variant="outline" className={cn('text-sm px-3 py-1', classificationClasses)}>
            {result.classification}
          </Badge>
        </CardHeader>
        <CardContent className="p-6 grid md:grid-cols-3 gap-6 items-start">
          <div className="flex flex-col items-center justify-center gap-4 md:border-r md:border-border md:pr-6">
            <h3 className="text-center font-semibold text-lg">Authenticity Score</h3>
            <ScoreGauge value={authenticityScore} />
            <p className="text-sm text-center text-muted-foreground">
              Weighted score from all agents.
            </p>
          </div>
          <div className="md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Card>
              <CardHeader className="flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Perceptual</CardTitle>
                <Microscope className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{Math.round(result.perceptualScore * 100)}%</div>
                <p className="text-xs text-muted-foreground">Forensic artifacts.</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Semantic</CardTitle>
                <Scale className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{Math.round(result.semanticScore * 100)}%</div>
                <p className="text-xs text-muted-foreground">A/V coherence.</p>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-primary" />
            <CardTitle>AI Reasoning Summary</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-foreground/90 leading-relaxed">{result.reasoningSummary}</p>
          
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="details">
              <AccordionTrigger>View Detailed Analysis</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-6 pt-4">
                  <Separator />
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-2 flex items-center gap-2"><Microscope className="h-4 w-4" /> Perceptual Forensics:</h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">{result.perceptualExplanation}</p>
                    </div>
                    <div>
                        <h4 className="font-semibold mb-2 flex items-center gap-2"><Scale className="h-4 w-4" /> Semantic Coherence:</h4>
                        <p className="text-sm text-muted-foreground leading-relaxed">{result.semanticExplanation}</p>
                    </div>
                  </div>

                  {hasForensicData && (
                    <>
                      <Separator />
                      <div>
                        <h4 className="font-semibold mb-2 flex items-center gap-2"><AlertTriangle className="h-4 w-4 text-accent" /> Forensic Indicators</h4>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm mt-4">
                            <div className="bg-muted/50 p-3 rounded-lg">
                                <p className="text-xs text-muted-foreground">Avg. FFT Deviation</p>
                                <p className="font-mono font-bold text-base">{(result.forensicIndicators.avgFFTDeviation || 0).toPrecision(3)}</p>
                            </div>
                            <div className="bg-muted/50 p-3 rounded-lg">
                                <p className="text-xs text-muted-foreground">Avg. Noise Residual</p>
                                <p className="font-mono font-bold text-base">{(result.forensicIndicators.avgNoiseResidual || 0).toPrecision(3)}</p>
                            </div>
                            <div className="bg-muted/50 p-3 rounded-lg">
                                <p className="text-xs text-muted-foreground">Spectral Slope (α)</p>
                                <p className="font-mono font-bold text-base">{(result.forensicIndicators.spectralSlope || 0).toPrecision(3)}</p>
                            </div>
                             <div className="bg-muted/50 p-3 rounded-lg">
                                <p className="text-xs text-muted-foreground">Model Confidence</p>
                                <p className="font-mono font-bold text-base">{(result.forensicIndicators.modelConfidence || 0).toPrecision(3)}</p>
                            </div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

        </CardContent>
        <CardFooter className="flex-col items-start gap-4 pt-6">
          <Separator />
          <div className="flex items-center justify-between w-full">
            <p className="text-sm text-muted-foreground">Help improve the AI: What is this video?</p>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => onFeedback('Real')}>
                <CheckCircle className="mr-2 h-4 w-4" /> Real
              </Button>
              <Button variant="outline" size="sm" onClick={() => onFeedback('AI-Generated')}>
                <BrainCircuit className="mr-2 h-4 w-4" /> AI-Generated
              </Button>
            </div>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
};

export default AnalysisResultsDisplay;
