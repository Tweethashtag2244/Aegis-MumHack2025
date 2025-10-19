
"use client";

import { useEffect } from "react";
import { useToast } from "@/hooks/use-toast";
import { errorEmitter } from "@/firebase/error-emitter";
import {
  FirestorePermissionError,
  type SecurityRuleContext,
} from "@/firebase/errors";
import { useUser } from "@/firebase/auth/use-user";

const FirebaseErrorListener = () => {
  const { toast } = useToast();
  const { user } = useUser();

  useEffect(() => {
    const handlePermissionError = (error: FirestorePermissionError) => {
      const context: SecurityRuleContext = error.context;
      const errorMessage = `
Firestore Security Rules Denied Request:
- **Operation:** ${context.operation}
- **Path:** ${context.path}
- **User Auth State:** ${user ? "Authenticated (UID: " + user.uid + ")" : "Unauthenticated"}
      `;

      console.error(error);

      toast({
        variant: "destructive",
        title: "Firestore Permission Error",
        description: (
          <pre className="mt-2 w-[340px] rounded-md bg-slate-950 p-4">
            <code className="text-white">{errorMessage}</code>
          </pre>
        ),
      });
    };

    errorEmitter.on("permission-error", handlePermissionError);

    return () => {
      errorEmitter.off("permission-error", handlePermissionError);
    };
  }, [toast, user]);

  return null;
};

export default FirebaseErrorListener;
