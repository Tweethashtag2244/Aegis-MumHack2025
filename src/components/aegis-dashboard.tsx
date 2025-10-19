
"use client";

import { useState, useMemo, useEffect } from 'react';
import type { FC } from 'react';
import { useToast } from '@/hooks/use-toast';
import { runFullAnalysis, type FullAnalysisOutput } from '@/ai/flows/run-full-analysis';
import type { ComputerVisionOutput } from '@/ai/types';
import VideoUploadForm from './video-upload-form';
import AnalysisResultsDisplay from './analysis-results-display';
import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAuth, useUser } from '@/firebase';
import { signInAnonymously } from 'firebase/auth';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { useFirestore } from '@/firebase';
import { errorEmitter } from '@/firebase/error-emitter';
import { FirestorePermissionError } from '@/firebase/errors';

export type AnalysisResult = FullAnalysisOutput & {
  fileName: string;
  fileSize: number;
  videoDataUri: string;
};

const readFileAsDataURL = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
};

export const AegisDashboard: FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();
  const auth = useAuth();
  const { user, loading: userLoading } = useUser();
  const firestore = useFirestore();

  useEffect(() => {
    if (auth && !user && !userLoading) {
      signInAnonymously(auth).catch((error) => {
        console.error("Anonymous sign-in failed:", error);
        toast({
          variant: "destructive",
          title: "Authentication Failed",
          description: "Could not sign in anonymously. Feedback will not be saved.",
        });
      });
    }
  }, [auth, user, userLoading, toast]);
  
  const videoPreviewUrl = useMemo(() => {
    if (videoFile) {
      return URL.createObjectURL(videoFile);
    }
    return null;
  }, [videoFile]);

  const handleFileChange = (file: File | null) => {
    setVideoFile(file);
    setAnalysisResult(null); 
  };

  const handleClear = () => {
    setVideoFile(null);
    setAnalysisResult(null);
    if (videoPreviewUrl) {
        URL.revokeObjectURL(videoPreviewUrl);
    }
  }
  
  const handleFeedback = (label: 'Real' | 'AI-Generated') => {
    if (!analysisResult || !user || !firestore) {
      toast({
        variant: 'destructive',
        title: 'Feedback Error',
        description: 'Could not submit feedback. No analysis result, user, or database connection found.',
      });
      return;
    }
  
    const feedbackData = {
      userId: user.uid,
      fileName: analysisResult.fileName,
      authenticityScore: analysisResult.authenticityScore,
      classification: analysisResult.classification,
      userLabel: label,
      createdAt: serverTimestamp(),
    };
  
    const feedbackCollection = collection(firestore, 'feedback');
  
    addDoc(feedbackCollection, feedbackData)
      .then(() => {
        toast({
          title: 'Feedback Submitted',
          description: 'Thank you for helping improve our AI!',
        });
      })
      .catch((serverError) => {
        const permissionError = new FirestorePermissionError({
          path: feedbackCollection.path,
          operation: 'create',
          requestResourceData: feedbackData,
        });
        errorEmitter.emit('permission-error', permissionError);
  
        toast({
          variant: 'destructive',
          title: 'Feedback Submission Error',
          description: 'Could not save your feedback. Please check your connection or try again later.',
        });
      });
  };

  const handleAnalyze = async () => {
    if (!videoFile) {
      toast({
        variant: 'destructive',
        title: 'No Video Selected',
        description: 'Please upload a video file to analyze.',
      });
      return;
    }

    setIsLoading(true);
    setAnalysisResult(null);

    try {
      // Step 1: Call the Computer Vision server directly from the client
      const CV_SERVER_URL = 'http://127.0.0.1:5001/analyze';
      const formData = new FormData();
      formData.append('video', videoFile);

      const cvResponse = await fetch(CV_SERVER_URL, {
        method: 'POST',
        body: formData,
      });

      if (!cvResponse.ok) {
        const errorBody = await cvResponse.text();
        console.error("CV server error response:", errorBody);
        throw new Error(`CV server responded with status: ${cvResponse.status} ${cvResponse.statusText}`);
      }
      
      const perceptualAnalysis: ComputerVisionOutput = await cvResponse.json();

      // Step 2: Get the video data URI for the Genkit flow
      const videoDataUri = await readFileAsDataURL(videoFile);
      
      // Step 3: Run the main analysis flow with the CV results
      const result = await runFullAnalysis({ 
        videoDataUri,
        perceptualAnalysis,
      });

      setAnalysisResult({
        ...result,
        fileName: videoFile.name,
        fileSize: videoFile.size,
        videoDataUri,
      });

    } catch (error) {
      console.error('Analysis failed:', error);
      toast({
        variant: 'destructive',
        title: 'Analysis Failed',
        description: error instanceof Error ? error.message : 'An unexpected error occurred during the video analysis. Please try again.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto max-w-5xl py-8 px-4">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-bold tracking-tight md:text-4xl">Video Authenticity Analysis</h1>
        <p className="mt-2 text-lg text-muted-foreground">
          Upload a video to detect signs of AI generation using a multi-layered analysis.
        </p>
      </div>

      <div className="grid gap-8">
        <VideoUploadForm onFileChange={handleFileChange} videoPreviewUrl={videoPreviewUrl} file={videoFile} />
        
        {videoFile && !analysisResult && !isLoading && (
            <div className="flex justify-center gap-4">
                <Button onClick={handleAnalyze} disabled={isLoading || userLoading || !user} size="lg">
                    {(isLoading || userLoading) && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    {isLoading ? 'Analyzing...' : userLoading ? 'Authenticating...' : 'Analyze Video'}
                </Button>
                <Button onClick={handleClear} variant="outline" size="lg" disabled={isLoading}>
                    Clear
                </Button>
            </div>
        )}
        
        {isLoading && (
            <div className="flex flex-col items-center justify-center gap-4 rounded-lg border border-dashed border-border p-8 text-center animate-in fade-in-50">
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
                <h3 className="text-xl font-semibold">Performing AI Analysis...</h3>
                <p className="text-muted-foreground">This may take a moment. We're running perceptual and semantic analysis agents.</p>
            </div>
        )}

        {analysisResult && (
          <div className="space-y-8 animate-in fade-in-50">
            <AnalysisResultsDisplay result={analysisResult} onFeedback={handleFeedback} />
            <div className="flex justify-center">
              <Button onClick={handleClear} variant="outline" size="lg">Analyze Another Video</Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
