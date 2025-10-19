
export type SecurityRuleOperation =
  | "get"
  | "list"
  | "create"
  | "update"
  | "delete";

export type SecurityRuleContext = {
  path: string;
  operation: SecurityRuleOperation;
  requestResourceData?: any;
};

export class FirestorePermissionError extends Error {
  public readonly context: SecurityRuleContext;

  constructor(context: SecurityRuleContext) {
    const message = `Firestore permission denied for ${context.operation} on ${context.path}`;
    super(message);
    this.name = "FirestorePermissionError";
    this.context = context;
    Object.setPrototypeOf(this, FirestorePermissionError.prototype);
  }
}
