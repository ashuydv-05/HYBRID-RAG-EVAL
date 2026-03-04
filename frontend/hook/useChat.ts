'use client';

import { useState, useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Message, Source } from '@/types/chat';
import { sendMessage, getSessionHistory, clearSession } from '@/lib/api';

const STREAMING_DELAY_MS = 3;

interface UseChatOptions {
  sessionId?: string;
  onError?: (error: Error) => void;
}

interface UseChatReturn {
  messages: Message[];
  input: string;
  setInput: (value: string) => void;
  isLoading: boolean;
  isStreaming: boolean;
  sessionId: string;
  sendMessage: () => Promise<void>;
  stopStreaming: () => void;
  clearMessages: () => void;
  loadSessionHistory: (sid: string) => Promise<void>;
}

export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string>(
    options.sessionId || uuidv4()
  );

  const abortControllerRef = useRef<AbortController | null>(null);
  const streamingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const cleanupStreaming = useCallback(() => {
    if (streamingIntervalRef.current) {
      clearTimeout(streamingIntervalRef.current);
      streamingIntervalRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setIsLoading(false);
  }, []);

  const addMessage = useCallback((message: Message) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const updateMessage = useCallback((id: string, updates: Partial<Message>) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, ...updates } : msg))
    );
  }, []);

  const simulateStreaming = (
    fullContent: string,
    assistantMessageId: string,
    sources: Source[]
  ) => {
    setIsStreaming(true);
    let currentIndex = 0;
    const chars = fullContent.split('');

    const streamNext = () => {
      if (currentIndex >= chars.length) {
        updateMessage(assistantMessageId, {
          isStreaming: false,
          content: fullContent,
          sources,
        });
        setIsStreaming(false);
        streamingIntervalRef.current = null;
        return;
      }

      const currentContent = chars.slice(0, currentIndex + 1).join('');
      updateMessage(assistantMessageId, {
        content: currentContent,
        sources,
      });
      currentIndex++;

      streamingIntervalRef.current = setTimeout(streamNext, STREAMING_DELAY_MS);
    };

    streamingIntervalRef.current = setTimeout(streamNext, STREAMING_DELAY_MS);
  };

  const handleSendMessage = useCallback(async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;

    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    };

    addMessage(userMessage);
    setInput('');
    setIsLoading(true);

    const assistantMessageId = uuidv4();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      timestamp: new Date(),
    };
    addMessage(assistantMessage);

    abortControllerRef.current = new AbortController();

    try {
      const response = await sendMessage({
        message: userMessage.content,
        session_id: sessionId,
      });

      simulateStreaming(response.answer, assistantMessageId, response.sources);

      if (response.session_id && response.session_id !== sessionId) {
        setSessionId(response.session_id);
      }
    } catch (error) {
      let errorMessage = 'An unexpected error occurred. Please try again.';

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request was cancelled.';
        } else if (error.message.includes('Failed to fetch')) {
          errorMessage = 'Cannot connect to server. Please check your connection.';
        } else {
          errorMessage = error.message;
        }
      }

      updateMessage(assistantMessageId, {
        content: `Error: ${errorMessage}`,
        isStreaming: false,
        isError: true,
      });

      options.onError?.(error instanceof Error ? error : new Error(errorMessage));
      setIsStreaming(false);
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [input, isLoading, sessionId, addMessage, updateMessage, options]);

  const stopStreaming = useCallback(() => {
    cleanupStreaming();

    setMessages((prev) => {
      const lastMsg = prev[prev.length - 1];
      if (lastMsg && lastMsg.role === 'assistant') {
        return prev.map((msg) =>
          msg.id === lastMsg.id ? { ...msg, isStreaming: false } : msg
        );
      }
      return prev;
    });
  }, [cleanupStreaming]);

  const clearMessages = useCallback(async () => {
    cleanupStreaming();

    try {
      await clearSession(sessionId);
    } catch {
    }

    setMessages([]);
    setSessionId(uuidv4());
    setInput('');
  }, [sessionId, cleanupStreaming]);

  const loadSessionHistory = useCallback(async (sid: string) => {
    setIsLoading(true);

    try {
      const history = await getSessionHistory(sid);

      const historyMessages: Message[] = history.messages.map((msg, index) => ({
        id: `history-${index}`,
        role: msg.role as 'user' | 'assistant',
        content: msg.content,
        timestamp: new Date(),
      }));

      setMessages(historyMessages);
      setSessionId(sid);
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Failed to load history');
      options.onError?.(err);
    } finally {
      setIsLoading(false);
    }
  }, [options]);

  return {
    messages,
    input,
    setInput,
    isLoading,
    isStreaming,
    sessionId,
    sendMessage: handleSendMessage,
    stopStreaming,
    clearMessages,
    loadSessionHistory,
  };
}
