'use client';

import { EvaluationView } from '@/component/evaluation/EvaluationView';
import { useRouter } from 'next/navigation';

export default function EvaluationPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-[#f8f9fc] flex flex-col">
      <EvaluationView onBackToChat={() => router.push('/')} />
    </div>
  );
}
