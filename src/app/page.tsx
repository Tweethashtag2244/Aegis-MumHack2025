import { AegisDashboard } from '@/components/aegis-dashboard';
import { Shield } from 'lucide-react';

export default function Home() {
  return (
    <div className="flex flex-col min-h-dvh bg-background text-foreground">
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 max-w-screen-2xl items-center">
          <div className="mr-4 flex items-center">
            <Shield className="h-6 w-6 text-primary" />
            <span className="ml-2 font-bold text-lg">Aegis</span>
          </div>
        </div>
      </header>
      <main className="flex-1">
        <AegisDashboard />
      </main>
      <footer className="py-6 md:px-8 md:py-0">
        <div className="container flex flex-col items-center justify-center gap-4 md:h-24 md:flex-row">
          <p className="text-balance text-center text-sm leading-loose text-muted-foreground">
            Built with Next.js, Genkit, and ShadCN UI.
          </p>
        </div>
      </footer>
    </div>
  );
}
