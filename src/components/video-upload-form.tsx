"use client";

import type { FC, DragEvent } from 'react';
import { useState, useRef } from 'react';
import { cn } from '@/lib/utils';
import { UploadCloud, FileVideo, X } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface VideoUploadFormProps {
  onFileChange: (file: File | null) => void;
  videoPreviewUrl: string | null;
  file: File | null;
}

const VideoUploadForm: FC<VideoUploadFormProps> = ({ onFileChange, videoPreviewUrl, file }) => {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      if (files[0].type.startsWith('video/')) {
        onFileChange(files[0]);
      }
    }
  };

  const handleButtonClick = () => {
    inputRef.current?.click();
  };
  
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      if (files[0].type.startsWith('video/')) {
        onFileChange(files[0]);
      }
    }
  };
  
  const handleRemoveFile = () => {
    onFileChange(null);
    if(inputRef.current) {
        inputRef.current.value = "";
    }
  }

  return (
    <Card className="shadow-lg">
      <CardContent className="p-6">
        {videoPreviewUrl && file ? (
          <div className="flex flex-col items-center gap-4">
            <div className="w-full max-w-lg aspect-video rounded-lg overflow-hidden border bg-muted shadow-inner">
              <video src={videoPreviewUrl} controls className="w-full h-full object-contain" />
            </div>
            <div className="flex items-center gap-3 bg-muted/50 p-3 rounded-lg w-full max-w-lg">
                <FileVideo className="h-6 w-6 text-muted-foreground flex-shrink-0" />
                <div className="flex-grow min-w-0">
                    <p className="font-medium text-sm truncate">{file.name}</p>
                    <p className="text-xs text-muted-foreground">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
                <Button variant="ghost" size="icon" onClick={handleRemoveFile} className="flex-shrink-0">
                    <X className="h-4 w-4" />
                </Button>
            </div>
          </div>
        ) : (
          <div
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={handleButtonClick}
            className={cn(
              'flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer transition-colors',
              isDragging ? 'border-primary bg-primary/10' : 'border-border hover:border-primary/50'
            )}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={handleFileSelect}
            />
            <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center">
              <UploadCloud className="w-10 h-10 mb-4 text-muted-foreground" />
              <p className="mb-2 text-sm font-semibold text-foreground">
                Click to upload or drag and drop
              </p>
              <p className="text-xs text-muted-foreground">Any video format (MP4, MOV, etc.)</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default VideoUploadForm;
