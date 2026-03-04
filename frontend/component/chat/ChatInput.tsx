'use client';

import React, { useRef, useEffect, KeyboardEvent } from 'react';
import { ArrowUp, Square } from 'lucide-react';

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isLoading: boolean;
  isStreaming: boolean;
  onExternalInput?: (value: string) => void;
}

export function ChatInput({
  input,
  setInput,
  onSubmit,
  onStop,
  isLoading,
  isStreaming,
  onExternalInput,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [input]);

  useEffect(() => {
    if (!onExternalInput) return;

    const handleSetInput = (e: Event) => {
      const customEvent = e as CustomEvent<string>;
      setInput(customEvent.detail);
      textareaRef.current?.focus();
    };

    window.addEventListener('setInput', handleSetInput);
    return () => window.removeEventListener('setInput', handleSetInput);
  }, [setInput, onExternalInput]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !isLoading && !isStreaming) {
        onSubmit();
      }
    }
  };

  const handleSubmit = () => {
    if (input.trim() && !isLoading && !isStreaming) {
      onSubmit();
    }
  };

  const isActive = isLoading || isStreaming;

  return (
    <div className="bg-white border-t border-gray-200">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div
          className={`relative flex items-end bg-white border border-[#d9d9e3] rounded-[26px] transition-all duration-200 ${
            input.trim() || isActive ? 'shadow-sm' : ''
          }`}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message arXiv Research Assistant..."
            rows={1}
            className="flex-1 bg-transparent resize-none outline-none pl-5 pr-14 py-3 text-gray-800 placeholder-gray-400 max-h-[200px] text-sm leading-relaxed min-h-[44px]"
            disabled={isActive}
          />

          <div className="absolute right-2 bottom-2">
            <div className="relative inline-flex">
              {isActive && (
                <span
                  className="absolute rounded-full border-2 border-[#343541]/20 border-t-[#343541] animate-spin pointer-events-none"
                  style={{ inset: '-3px' }}
                />
              )}
              <button
                onClick={isActive ? onStop : handleSubmit}
                disabled={!isActive && !input.trim()}
                className={`relative w-9 h-9 flex items-center justify-center rounded-full transition-all duration-200 ${
                  isActive
                    ? 'bg-[#343541] text-white cursor-pointer'
                    : !input.trim()
                    ? 'bg-[#e5e5e5] text-[#a4a4a4] cursor-not-allowed'
                    : 'bg-[#343541] text-white hover:bg-[#2a2b32] cursor-pointer'
                }`}
                title={isActive ? 'Stop' : 'Send message'}
              >
                {isActive ? (
                  <Square size={14} fill="currentColor" />
                ) : (
                  <ArrowUp size={18} />
                )}
              </button>
            </div>
          </div>
        </div>

        <div className="text-center mt-2 text-xs text-gray-400">
          Press <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-500 font-mono text-[10px]">Enter</kbd> to send,
          <kbd className="mx-1 px-1.5 py-0.5 bg-gray-100 rounded text-gray-500 font-mono text-[10px]">Shift</kbd>+
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-500 font-mono text-[10px]">Enter</kbd> for new line
        </div>
      </div>
    </div>
  );
}
