'use client';

import { useChat } from '@/hooks/useChat';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { Menu, Plus, MessageSquare } from 'lucide-react';

export function ChatContainer() {
  const {
    messages,
    input,
    setInput,
    isLoading,
    isStreaming,
    sendMessage,
    stopStreaming,
    clearMessages,
  } = useChat({
    onError: (error) => {
      console.error('Chat error:', error);
    },
  });

  return (
    <div className="flex h-screen bg-white">
      <aside className="hidden md:flex w-64 bg-[#f9fafb] flex-col border-r border-gray-200">
        <div className="p-3">
          <button
            onClick={clearMessages}
            className="w-full flex items-center gap-3 px-3 py-3 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-100 hover:border-gray-400 transition-all duration-200"
          >
            <Plus size={16} />
            <span className="text-sm font-medium">New chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-2">
          {messages.length > 0 && (
            <div className="flex items-center gap-3 px-3 py-3 rounded-lg bg-white border border-gray-200 text-gray-700 shadow-sm">
              <MessageSquare size={16} className="text-gray-400" />
              <span className="text-sm truncate">
                {messages[0]?.content.slice(0, 30)}...
              </span>
            </div>
          )}
        </div>

        <div className="p-3">
          <button className="w-full flex items-center gap-3 px-3 py-3 rounded-lg text-gray-700 hover:bg-gray-100 transition-colors">
            <div className="w-8 h-8 rounded-full bg-[#10a37f] flex items-center justify-center text-sm font-medium text-white">
              U
            </div>
            <span className="text-sm font-medium">User</span>
          </button>
        </div>
      </aside>

      <main className="flex-1 flex flex-col relative">
        <header className="md:hidden flex items-center justify-between px-4 py-3 bg-white border-b border-gray-200">
          <button className="text-gray-600">
            <Menu size={24} />
          </button>
          <h1 className="text-gray-800 font-semibold">arXiv Research Assistant</h1>
          <button onClick={clearMessages} className="text-gray-600">
            <Plus size={24} />
          </button>
        </header>

        <MessageList messages={messages} />

        <ChatInput
          input={input}
          setInput={setInput}
          onSubmit={sendMessage}
          onStop={stopStreaming}
          isLoading={isLoading}
          isStreaming={isStreaming}
        />
      </main>
    </div>
  );
}
