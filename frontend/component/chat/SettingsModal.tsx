'use client';

import React, { useState, useEffect } from 'react';
import { X, Key, Eye, EyeOff, Check, ExternalLink, ShieldCheck } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [groqKey, setGroqKey] = useState('');
  const [showGroq, setShowGroq] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setGroqKey(localStorage.getItem('groq_api_key') || '');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSave = () => {
    if (typeof window !== 'undefined') {
      if (groqKey.trim()) {
        localStorage.setItem('groq_api_key', groqKey.trim());
      } else {
        localStorage.removeItem('groq_api_key');
      }

      // Dispatch event to notify components
      window.dispatchEvent(new Event('apiKeyUpdated'));

      setSavedSuccess(true);
      setTimeout(() => {
        setSavedSuccess(false);
        onClose();
      }, 1000);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-fade-in">
      <div className="bg-white border border-slate-200 rounded-3xl shadow-2xl max-w-md w-full overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-slate-100 bg-[#fcfcff]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-[#eeebff] flex items-center justify-center text-[#5542f6]">
              <Key size={18} />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">API Key Configuration</h2>
              <p className="text-xs text-slate-500">Configure your Groq API key for inference</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-5">
          {/* Groq Key */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                <span>Groq API Key</span>
                <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                  Required
                </span>
              </label>
              <a
                href="https://console.groq.com/keys"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-[#5542f6] hover:underline flex items-center gap-1 font-medium"
              >
                <span>Get Free Key</span>
                <ExternalLink size={12} />
              </a>
            </div>
            <div className="relative">
              <input
                type={showGroq ? 'text' : 'password'}
                value={groqKey}
                onChange={(e) => setGroqKey(e.target.value)}
                placeholder="gsk_..."
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-2.5 pr-10 text-sm text-slate-800 placeholder-slate-400 focus:outline-hidden focus:border-[#5542f6] focus:bg-white transition-all font-mono"
              />
              <button
                type="button"
                onClick={() => setShowGroq(!showGroq)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
              >
                {showGroq ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            <p className="text-[11px] text-slate-500">
              Powers chat reasoning, hybrid search generation, and benchmark evaluation.
            </p>
          </div>

          <div className="bg-indigo-50/50 border border-indigo-100 rounded-2xl p-3.5 flex items-start gap-3">
            <ShieldCheck size={18} className="text-[#5542f6] mt-0.5 shrink-0" />
            <p className="text-xs text-slate-600 leading-relaxed">
              Your API key is stored locally in your browser and used only to authenticate requests to Groq.
            </p>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-slate-100 bg-[#fcfcff]">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-sm font-medium text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            className="flex items-center gap-2 px-5 py-2 rounded-xl bg-[#5542f6] hover:bg-[#4332e6] text-white text-sm font-semibold shadow-md shadow-[#5542f6]/20 transition-all cursor-pointer"
          >
            {savedSuccess ? (
              <>
                <Check size={16} />
                <span>Saved!</span>
              </>
            ) : (
              <span>Save & Apply</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
