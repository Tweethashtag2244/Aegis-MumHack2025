
'use client';
import { initializeFirebase, FirebaseProvider } from '@/firebase';
import type { PropsWithChildren } from 'react';
import { useMemo } from 'react';

export const FirebaseClientProvider = ({ children }: PropsWithChildren) => {
  const firebaseInstances = useMemo(() => initializeFirebase(), []);
  return (
    <FirebaseProvider value={firebaseInstances}>
      {children}
    </FirebaseProvider>
  );
};
