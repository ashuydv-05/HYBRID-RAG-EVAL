'use client';

import React, { useRef, useEffect } from 'react';
import { Message } from '@/types/chat';
import { MessageItem } from './MessageItem';

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
      <div className="flex-1 flex items-center justify-center p-8 bg-white">
        <div className="text-center max-w-2xl mx-auto">
          <h1 className="text-3xl font-semibold text-gray-800 mb-4">
            arXiv Research Assistant
          </h1>
          <p className="text-gray-500 mb-8">
            Ask me anything about AI/ML research papers. I can search, summarize,
            and help you understand papers from arXiv.
          </p>
          <div className="grid gap-3 max-w-lg mx-auto">
            <ExamplePrompt text="Find papers about transformer models" />
            <ExamplePrompt text="What are the latest advances in LLMs?" />
            <ExamplePrompt text="Summarize the paper about attention mechanisms" />
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

function ExamplePrompt({ text }: { text: string }) {
  return (
    <button
      className="w-full px-4 py-3 bg-white border border-gray-200 rounded-xl text-sm text-gray-600 hover:border-gray-300 hover:bg-gray-50 transition-all duration-200 text-left shadow-sm hover:shadow"
      onClick={() => {
        const event = new CustomEvent('setInput', { detail: text });
        window.dispatchEvent(event);
      }}
    >
      {text}
    </button>
  );
}
