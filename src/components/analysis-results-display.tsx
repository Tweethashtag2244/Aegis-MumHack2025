"use client";

import type { FC } from 'react';
import type { AnalysisResult } from './aegis-dashboard';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { ScoreGauge } from './score-gauge';
import { 
  Bot, Scale, CheckCircle, BrainCircuit, Microscope, 
  Fingerprint, Globe, ShieldAlert, FileSearch 
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from './ui/button';

interface AnalysisResultsDisplayProps {
  result: AnalysisResult;
  onFeedback: (label: 'Real' | 'AI-Generated') => void;
}

const AnalysisResultsDisplay: FC<AnalysisResultsDisplayProps> = ({ result, onFeedback }) => {
  const {
    authenticityScore,
    classification,
    reasoningSummary,
    perceptualScore,
    semanticScore,
    consensusScore,
    perceptualExplanation,
    semanticExplanation,
    attribution,
    provenance,
    forensicIndicators
  } = result;

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const scorePercent = Math.round(authenticityScore * 100);

  return (
    <div className="grid gap-6 animate-in fade-in-50">
      
      {/* 1. Main Verdict Card */}
      <Card className="overflow-hidden border-2 border-primary/10 bg-card/50 shadow-md">
        <CardHeader className="flex flex-row items-center justify-between pb-2 bg-muted/30">
          <div>
            <CardTitle>Aegis Verdict</CardTitle>
            <CardDescription>Analysis for <span className="font-mono text-xs">{result.fileName}</span></CardDescription>
          </div>
          <Badge variant={classification === 'Likely Real' ? 'default' : 'destructive'} className="text-sm px-3 py-1">
            {classification}
          </Badge>
        </CardHeader>
        <CardContent className="p-6 grid md:grid-cols-3 gap-8 items-center">
          <div className="flex flex-col items-center justify-center gap-2">
            <ScoreGauge value={scorePercent} />
            <p className="text-sm font-medium text-muted-foreground mt-2">Authenticity Probability</p>
          </div>
          <div className="md:col-span-2 space-y-4">
             <div className="flex items-start gap-3">
                <Bot className="h-6 w-6 text-primary mt-1" />
                <div className="space-y-1">
                   <h4 className="font-semibold text-lg">AI Reasoning</h4>
                   <p className="text-muted-foreground leading-relaxed text-sm">{reasoningSummary}</p>
                </div>
             </div>
          </div>
        </CardContent>
      </Card>

      {/* 2. Agent Details Tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="perceptual">Perceptual</TabsTrigger>
          <TabsTrigger value="semantic">Semantic</TabsTrigger>
          <TabsTrigger value="provenance">Provenance</TabsTrigger>
        </TabsList>

        {/* Tab: Overview (Summary Cards) */}
        <TabsContent value="overview" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Attribution */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Attribution</CardTitle>
                <Fingerprint className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-xl font-bold truncate">{attribution?.detectedModel || 'Unknown'}</div>
                <p className="text-xs text-muted-foreground">Confidence: {((attribution?.confidence || 0)*100).toFixed(0)}%</p>
              </CardContent>
            </Card>
            {/* Consensus */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Consensus</CardTitle>
                <Globe className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className={cn("text-2xl font-bold", getScoreColor(consensusScore || 0))}>
                  {((consensusScore || 0)*100).toFixed(0)}%
                </div>
                <p className="text-xs text-muted-foreground">External Verification</p>
              </CardContent>
            </Card>
            {/* Perceptual */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Visuals</CardTitle>
                <Microscope className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className={cn("text-2xl font-bold", getScoreColor(perceptualScore))}>
                  {(perceptualScore * 100).toFixed(0)}%
                </div>
                <p className="text-xs text-muted-foreground">Artifact Analysis</p>
              </CardContent>
            </Card>
            {/* Semantic */}
            <Card>
               <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Semantic</CardTitle>
                <Scale className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className={cn("text-2xl font-bold", getScoreColor(semanticScore))}>
                  {(semanticScore * 100).toFixed(0)}%
                </div>
                <p className="text-xs text-muted-foreground">Logic & Physics</p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Tab: Perceptual Details */}
        <TabsContent value="perceptual" className="mt-6 space-y-4">
           <Card>
             <CardHeader>
               <CardTitle>Perceptual Forensics Agent</CardTitle>
               <CardDescription>Analysis of spectral noise, compression artifacts, and pixel irregularities.</CardDescription>
             </CardHeader>
             <CardContent className="space-y-6">
                <div className="space-y-2">
                   <div className="flex justify-between text-sm"><span>Score</span><span className="font-bold">{(perceptualScore*100).toFixed(1)}%</span></div>
                   <Progress value={perceptualScore*100} />
                </div>
                <div className="bg-muted p-4 rounded-md text-sm">
                   <p>{perceptualExplanation}</p>
                </div>
                {forensicIndicators && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
                     <div className="p-3 bg-secondary/50 rounded-lg">
                        <p className="text-xs text-muted-foreground">Noise Residual</p>
                        <p className="font-mono font-bold">{(forensicIndicators.avgNoiseResidual || 0).toFixed(1)}</p>
                     </div>
                     <div className="p-3 bg-secondary/50 rounded-lg">
                        <p className="text-xs text-muted-foreground">Spectral Slope</p>
                        <p className="font-mono font-bold">{(forensicIndicators.spectralSlope || 0).toFixed(2)}</p>
                     </div>
                  </div>
                )}
             </CardContent>
           </Card>
        </TabsContent>

        {/* Tab: Semantic Details */}
        <TabsContent value="semantic" className="mt-6 space-y-4">
           <Card>
             <CardHeader>
               <CardTitle>Semantic Coherence Agent</CardTitle>
               <CardDescription>Evaluates scene logic, lighting consistency, and audio-visual sync.</CardDescription>
             </CardHeader>
             <CardContent className="space-y-6">
                <div className="space-y-2">
                   <div className="flex justify-between text-sm"><span>Score</span><span className="font-bold">{(semanticScore*100).toFixed(1)}%</span></div>
                   <Progress value={semanticScore*100} />
                </div>
                <div className="bg-muted p-4 rounded-md text-sm">
                   <p>{semanticExplanation}</p>
                </div>
             </CardContent>
           </Card>
        </TabsContent>

        {/* Tab: Provenance & Attribution */}
        <TabsContent value="provenance" className="mt-6 space-y-4">
           <Card>
             <CardHeader>
               <CardTitle>Provenance & Attribution</CardTitle>
               <CardDescription>Metadata analysis and AI model fingerprinting.</CardDescription>
             </CardHeader>
             <CardContent className="space-y-4">
                <div className="flex items-center justify-between border-b pb-4">
                    <div>
                      <p className="font-medium">Detected Encoder</p>
                      <p className="text-sm text-muted-foreground">{provenance?.encoder || "Unknown"}</p>
                    </div>
                    {provenance?.encoder?.includes('lavf') ? 
                      <Badge variant="destructive">Suspicious</Badge> : <Badge variant="secondary">Neutral</Badge>
                    }
                </div>
                <div className="flex items-center justify-between pt-2">
                    <div>
                      <p className="font-medium">C2PA Signature</p>
                      <p className="text-sm text-muted-foreground">Digital Trust Manifest</p>
                    </div>
                    {provenance?.hasC2PA ? 
                       <Badge className="bg-green-600">Verified</Badge> : <Badge variant="outline">Missing</Badge>
                    }
                </div>
             </CardContent>
           </Card>
        </TabsContent>
      </Tabs>

      {/* Feedback Section */}
      <Card>
        <CardContent className="flex items-center justify-between p-4">
           <p className="text-sm text-muted-foreground">Help us improve Aegis. What is this video?</p>
           <div className="flex gap-2">
              <Button variant="ghost" size="sm" onClick={() => onFeedback('Real')} className="gap-2">
                <CheckCircle className="h-4 w-4 text-green-600" /> Real
              </Button>
              <Button variant="ghost" size="sm" onClick={() => onFeedback('AI-Generated')} className="gap-2">
                <BrainCircuit className="h-4 w-4 text-red-600" /> AI
              </Button>
           </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AnalysisResultsDisplay;