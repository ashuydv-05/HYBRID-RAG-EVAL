'use client';

import React, { useRef, useEffect } from 'react';
import { Message } from '@/types/chat';
import { MessageItem } from './MessageItem';
import { Search, Sparkles, FileText } from 'lucide-react';

interface MessageListProps {
  messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-12 bg-white overflow-y-auto">
        <div className="text-center max-w-2xl mx-auto w-full">
          {/* Header Title with Paper Airplane Doodle */}
          <div className="relative inline-block mb-3">
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
              <span className="text-[#5542f6]">arXiv</span>{' '}
              <span className="text-[#1e1b4b]">Research Assistant</span>
            </h1>

            {/* Paper Airplane Doodle Icon */}
            <div className="absolute -top-3 -right-10 hidden sm:block pointer-events-none">
              <svg
                width="36"
                height="36"
                viewBox="0 0 48 48"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="text-[#5542f6] opacity-80 rotate-12"
              >
                <path
                  d="M44 4L2 22L20 28L26 46L44 4Z"
                  stroke="#5542f6"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M44 4L20 28"
                  stroke="#5542f6"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {/* Dotted Trail */}
                <path
                  d="M12 36C8 38 4 39 2 42"
                  stroke="#818cf8"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeDasharray="3 3"
                />
              </svg>
            </div>
          </div>

          {/* Subtitle */}
          <p className="text-slate-500 text-sm md:text-base leading-relaxed max-w-lg mx-auto mb-10">
            Ask me anything about AI/ML research papers.
            <br />
            I can search, summarize, and help you understand papers from arXiv.
          </p>

          {/* 3 Prompt Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto text-left">
            <PromptCard
              icon={<Search className="w-5 h-5 text-[#5542f6]" />}
              prefix="Find papers about"
              highlight="transformer models"
              query="Find papers about transformer models"
            />
            <PromptCard
              icon={<Sparkles className="w-5 h-5 text-[#5542f6]" />}
              prefix="What are the latest"
              highlight="advances in LLMs?"
              query="What are the latest advances in LLMs?"
            />
            <PromptCard
              icon={<FileText className="w-5 h-5 text-[#5542f6]" />}
              prefix="Summarize the paper"
              highlight="about attention mechanisms"
              query="Summarize the paper about attention mechanisms"
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto bg-white">
      {messages.map((message) => (
        <MessageItem key={message.id} message={message} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

function PromptCard({
  icon,
  prefix,
  highlight,
  query,
}: {
  icon: React.ReactNode;
  prefix: string;
  highlight: string;
  query: string;
}) {
  return (
    <button
      onClick={() => {
        const event = new CustomEvent('setInput', { detail: query });
        window.dispatchEvent(event);
      }}
      className="bg-white border border-slate-200/90 rounded-2xl p-5 hover:border-[#5542f6]/40 hover:shadow-[0_8px_20px_rgba(85,66,246,0.08)] transition-all duration-200 text-left group flex flex-col justify-between min-h-[135px] cursor-pointer"
    >
      <div className="w-9 h-9 rounded-xl bg-[#f4f2ff] flex items-center justify-center mb-3 group-hover:scale-105 transition-transform">
        {icon}
      </div>
      <div>
        <p className="text-slate-600 text-xs leading-tight">{prefix}</p>
        <p className="text-slate-900 font-bold text-sm mt-0.5 group-hover:text-[#5542f6] transition-colors">
          {highlight}
        </p>
      </div>
    </button>
  );
}
