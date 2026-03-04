'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check, ChevronDown, ChevronRight } from 'lucide-react';
import { useState, ComponentPropsWithoutRef } from 'react';
import { Message, Source } from '@/types/chat';

interface MessageItemProps {
  message: Message;
}

export function MessageItem({ message }: MessageItemProps) {
  const isUser = message.role === 'user';

  return (
    <div className="py-5 px-4 message-enter">
      <div className="max-w-3xl mx-auto">
        <div className={`${isUser ? 'flex justify-end' : ''}`}>
          {isUser ? (
            <div className="max-w-[80%] bg-[#f7f7f8] border border-gray-200 rounded-2xl rounded-tr-sm px-4 py-3 text-gray-800 text-base leading-relaxed break-words">
              {message.content}
            </div>
          ) : (
            <>
              <div className={`text-gray-800 text-sm leading-relaxed ${message.isStreaming && message.content ? 'streaming-cursor' : ''}`}>
                {message.isStreaming && !message.content ? (
                  <div className="not-prose flex items-center gap-1 py-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  </div>
                ) : (
                  <MarkdownRenderer content={message.content} />
                )}
              </div>
              {message.sources && message.sources.length > 0 && (
                <SourcesSection sources={message.sources} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function MarkdownRenderer({ content }: { content: string }) {
  const [copiedBlock, setCopiedBlock] = useState<string | null>(null);

  const handleCopy = async (text: string, blockId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedBlock(blockId);
      setTimeout(() => setCopiedBlock(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>,
        strong: ({ children }) => <strong className="font-semibold text-gray-900">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
        h1: ({ children }) => <h1 className="text-2xl font-bold mt-6 mb-3 text-gray-900">{children}</h1>,
        h2: ({ children }) => <h2 className="text-xl font-semibold mt-5 mb-2 text-gray-900">{children}</h2>,
        h3: ({ children }) => <h3 className="text-lg font-semibold mt-4 mb-2 text-gray-900">{children}</h3>,
        h4: ({ children }) => <h4 className="text-base font-semibold mt-3 mb-1 text-gray-900">{children}</h4>,
        ul: ({ children }) => <ul className="list-disc list-outside pl-5 mb-3 space-y-1">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-outside pl-5 mb-3 space-y-1">{children}</ol>,
        li: ({ children }) => <li className="leading-relaxed pl-1">{children}</li>,
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-gray-200 pl-4 text-gray-500 my-3">{children}</blockquote>
        ),
        hr: () => <hr className="border-gray-200 my-4" />,
        table: ({ children }) => (
          <div className="overflow-x-auto mb-3">
            <table className="w-full border-collapse text-sm">{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className="bg-gray-50">{children}</thead>,
        th: ({ children }) => <th className="border border-gray-200 px-3 py-2 text-left font-semibold text-gray-700">{children}</th>,
        td: ({ children }) => <td className="border border-gray-200 px-3 py-2 text-gray-700">{children}</td>,
        code({ node, inline, className, children, ...props }: ComponentPropsWithoutRef<'code'>) {
          const match = /language-(\w+)/.exec(className || '');
          const codeString = String(children).replace(/\n$/, '');
          const blockId = node?.position?.start?.line
            ? `code-${node.position.start.line}`
            : `code-${crypto.randomUUID()}`;

          if (!inline && match) {
            return (
              <div className="relative group rounded-lg overflow-hidden my-4">
                <div className="flex items-center justify-between bg-gray-100 px-4 py-2 border-b border-gray-200">
                  <span className="text-xs text-gray-500 font-medium uppercase">
                    {match[1]}
                  </span>
                  <button
                    onClick={() => handleCopy(codeString, blockId)}
                    className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 transition-colors"
                  >
                    {copiedBlock === blockId ? (
                      <>
                        <Check size={14} />
                        <span>Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy size={14} />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                </div>
                <SyntaxHighlighter
                  style={oneLight}
                  language={match[1]}
                  PreTag="div"
                  customStyle={{
                    margin: 0,
                    padding: '1rem',
                    fontSize: '0.875rem',
                    lineHeight: '1.7',
                  }}
                  wrapLines={true}
                  wrapLongLines={true}
                >
                  {codeString}
                </SyntaxHighlighter>
              </div>
            );
          }

          return (
            <code className="bg-gray-100 text-red-500 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
              {children}
            </code>
          );
        },
        pre({ children }) {
          return <>{children}</>;
        },
        a({ href, children, ...props }) {
          return (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#10a37f] hover:underline"
              {...props}
            >
              {children}
            </a>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function SourcesSection({ sources }: { sources: Source[] }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (sources.length === 0) return null;

  return (
    <div className="mt-5 pt-3 border-t border-gray-200">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown size={16} />
        ) : (
          <ChevronRight size={16} />
        )}
        <span>{isExpanded ? 'Hide sources' : `Sources (${sources.length})`}</span>
      </button>

      {isExpanded && (
        <div className="mt-3 grid gap-2">
          {sources.map((source, index) => (
            <SourceItem key={source.id || index} source={source} index={index} />
          ))}
        </div>
      )}
    </div>
  );
}

function SourceItem({ source, index }: { source: Source; index: number }) {
  return (
    <div className="flex items-start gap-3 p-3 bg-[#f7f7f8] rounded-xl border border-gray-200 hover:border-gray-300 transition-all duration-200">
      <div className="flex-shrink-0 w-6 h-6 rounded bg-white flex items-center justify-center shadow-sm">
        <span className="text-xs font-bold text-[#10a37f]">{index + 1}</span>
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-gray-800 truncate">
          {source.title}
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500 mt-1 flex-wrap">
          {source.arxiv_id && (
            <span className="font-mono bg-white px-1.5 py-0.5 rounded border border-gray-200 text-[#10a37f]">
              {source.arxiv_id}
            </span>
          )}
          {source.year && <span>{source.year}</span>}
          {source.score > 0 && (
            <span className="text-green-600 font-medium">
              {(source.score * 100).toFixed(0)}% match
            </span>
          )}
        </div>
        {Array.isArray(source.authors) && source.authors.length > 0 && (
          <div className="text-xs text-gray-400 mt-1 truncate">
            {source.authors.slice(0, 3).join(', ')}
            {source.authors.length > 3 && ' et al.'}
          </div>
        )}
      </div>
      {source.pdf_url && (
        <a
          href={source.pdf_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-shrink-0 text-gray-400 hover:text-[#10a37f] transition-colors p-1"
          title="Open PDF"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </a>
      )}
    </div>
  );
}
