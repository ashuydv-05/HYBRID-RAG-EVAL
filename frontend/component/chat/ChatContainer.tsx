'use client';

import { useState, useEffect } from 'react';
import { useChat } from '@/hooks/useChat';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { SettingsModal } from './SettingsModal';
import { EvaluationView } from '@/component/evaluation/EvaluationView';
import {
  Menu,
  Plus,
  MessageSquare,
  Sparkles,
  Star,
  Clock,
  BarChart3,
  KeyRound,
  Settings,
} from 'lucide-react';

export function ChatContainer() {
  const [activeTab, setActiveTab] = useState<'chat' | 'evaluation'>('chat');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [hasApiKey, setHasApiKey] = useState(true);

  useEffect(() => {
    const checkApiKey = () => {
      if (typeof window !== 'undefined') {
        const key = localStorage.getItem('groq_api_key');
        setHasApiKey(Boolean(key));
      }
    };
    checkApiKey();
    window.addEventListener('apiKeyUpdated', checkApiKey);
    return () => window.removeEventListener('apiKeyUpdated', checkApiKey);
  }, []);

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

  const handleNewChat = () => {
    setActiveTab('chat');
    clearMessages();
  };

  return (
    <div className="flex h-screen bg-white font-sans antialiased text-slate-900">
      {/* Left Sidebar */}
      <aside className="hidden md:flex w-64 bg-[#fcfcff] flex-col border-r border-slate-100 p-4 justify-between select-none">
        <div className="space-y-4">
          {/* Brand Header */}
          <div
            onClick={() => setActiveTab('chat')}
            className="flex items-center gap-2.5 px-2 py-1 cursor-pointer"
          >
            <div className="w-7 h-7 rounded-lg bg-[#eeebff] flex items-center justify-center text-[#5542f6]">
              <Sparkles size={16} />
            </div>
            <span className="font-semibold text-slate-800 text-sm tracking-tight">
              arXiv Assistant
            </span>
          </div>

          {/* New Chat Button */}
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-[#eeebff] hover:bg-[#e4dfff] text-[#5542f6] font-medium text-sm transition-all duration-200 shadow-xs cursor-pointer"
          >
            <Plus size={16} strokeWidth={2.5} />
            <span>New chat</span>
          </button>

          {/* Navigation Links */}
          <nav className="space-y-1 pt-2">
            <button
              onClick={() => setActiveTab('chat')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors cursor-pointer ${
                activeTab === 'chat'
                  ? 'bg-[#f4f2ff] text-[#5542f6]'
                  : 'text-slate-600 hover:text-[#5542f6] hover:bg-[#f4f2ff]'
              }`}
            >
              <MessageSquare size={17} className={activeTab === 'chat' ? 'text-[#5542f6]' : 'text-slate-400'} />
              <span>Chats</span>
            </button>

            <button
              onClick={() => setActiveTab('chat')}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-slate-600 hover:text-[#5542f6] hover:bg-[#f4f2ff] text-sm font-medium transition-colors cursor-pointer"
            >
              <Star size={17} className="text-slate-400" />
              <span>Starred</span>
            </button>

            <button
              onClick={() => setActiveTab('chat')}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-slate-600 hover:text-[#5542f6] hover:bg-[#f4f2ff] text-sm font-medium transition-colors cursor-pointer"
            >
              <Clock size={17} className="text-slate-400" />
              <span>History</span>
            </button>

            <button
              onClick={() => setActiveTab('evaluation')}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium transition-colors cursor-pointer ${
                activeTab === 'evaluation'
                  ? 'bg-[#f4f2ff] text-[#5542f6]'
                  : 'text-slate-600 hover:text-[#5542f6] hover:bg-[#f4f2ff]'
              }`}
            >
              <div className="flex items-center gap-3">
                <BarChart3 size={17} className={activeTab === 'evaluation' ? 'text-[#5542f6]' : 'text-slate-400'} />
                <span>Evaluation</span>
              </div>
              <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded-md bg-[#eeebff] text-[#5542f6]">
                2×2
              </span>
            </button>

            {/* API Keys / Settings */}
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-slate-600 hover:text-[#5542f6] hover:bg-[#f4f2ff] text-sm font-medium transition-colors cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <KeyRound size={17} className="text-[#5542f6]" />
                <span>API Keys</span>
              </div>
              <span className="w-2 h-2 rounded-full bg-emerald-500" title="API Key configured" />
            </button>
          </nav>
        </div>

        {/* User Profile Pill at Bottom */}
        <div className="pt-4 border-t border-slate-100">
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="w-full flex items-center gap-3 px-2 py-2 rounded-xl text-slate-700 hover:bg-[#f4f2ff] transition-colors group cursor-pointer"
          >
            <div className="w-8 h-8 rounded-full bg-[#2b4c7e] flex items-center justify-center text-xs font-bold text-white shadow-xs">
              U
            </div>
            <span className="text-sm font-semibold text-slate-800">User</span>
            <Settings size={15} className="text-slate-400 ml-auto group-hover:text-slate-600" />
          </button>
        </div>
      </aside>

      {/* Main Content View */}
      <main className="flex-1 flex flex-col relative bg-white overflow-hidden">
        {/* Mobile Header */}
        <header className="md:hidden flex items-center justify-between px-4 py-3 bg-white border-b border-slate-100">
          <button
            onClick={() => setActiveTab(activeTab === 'chat' ? 'evaluation' : 'chat')}
            className="text-slate-600"
          >
            <Menu size={22} />
          </button>
          <h1 className="text-slate-900 font-bold text-base">
            <span className="text-[#5542f6]">arXiv</span> {activeTab === 'chat' ? 'Assistant' : 'Evaluation Matrix'}
          </h1>
          <button onClick={handleNewChat} className="text-slate-600">
            <Plus size={22} />
          </button>
        </header>

        {activeTab === 'chat' ? (
          <>
            {/* Message Thread */}
            <MessageList messages={messages} />

            {/* Input Bar */}
            <ChatInput
              input={input}
              setInput={setInput}
              onSubmit={sendMessage}
              onStop={stopStreaming}
              isLoading={isLoading}
              isStreaming={isStreaming}
            />
          </>
        ) : (
          <EvaluationView onBackToChat={() => setActiveTab('chat')} />
        )}

        {/* API Key Configuration Modal */}
        <SettingsModal
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
        />
      </main>
    </div>
  );
}
