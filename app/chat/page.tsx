'use client';

import { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          conversationHistory: messages,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        const aiMessage: Message = {
          role: 'assistant',
          content: data.message,
          timestamp: data.timestamp,
        };
        setMessages(prev => [...prev, aiMessage]);
      } else {
        const errorMessage: Message = {
          role: 'assistant',
          content: `Error: ${data.error || 'Failed to get response'}`,
          timestamp: new Date().toISOString(),
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error: any) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: ${error.message}`,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setInput('');
  };

  return (
    <div className="flex flex-col h-screen bg-[#0d0d0d]">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">AI</span>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white">Financial Analyst</h1>
            <p className="text-xs text-gray-500">Powered by Unusual Whales & GPT-4</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="px-4 py-2 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition"
            >
              Clear Chat
            </button>
          )}
          <a
            href="/"
            className="px-4 py-2 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition"
          >
            Dashboard
          </a>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-8">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full space-y-8">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto">
                  <span className="text-white font-bold text-2xl">AI</span>
                </div>
                <h2 className="text-2xl font-bold text-white">How can I help you today?</h2>
                <p className="text-gray-500">
                  Ask me about stocks, options flow, market conditions, or trading strategies
                </p>
              </div>

              {/* Example prompts */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
                <button
                  onClick={() => setInput("What's the current market sentiment?")}
                  className="p-4 text-left bg-gray-900 hover:bg-gray-800 border border-gray-800 hover:border-gray-700 rounded-xl transition group"
                >
                  <div className="text-sm font-medium text-white mb-1">Market Sentiment</div>
                  <div className="text-xs text-gray-500 group-hover:text-gray-400">
                    What's the current market sentiment?
                  </div>
                </button>
                <button
                  onClick={() => setInput("Analyze AAPL options flow")}
                  className="p-4 text-left bg-gray-900 hover:bg-gray-800 border border-gray-800 hover:border-gray-700 rounded-xl transition group"
                >
                  <div className="text-sm font-medium text-white mb-1">Options Flow</div>
                  <div className="text-xs text-gray-500 group-hover:text-gray-400">
                    Analyze AAPL options flow
                  </div>
                </button>
                <button
                  onClick={() => setInput("Is SPY showing unusual activity?")}
                  className="p-4 text-left bg-gray-900 hover:bg-gray-800 border border-gray-800 hover:border-gray-700 rounded-xl transition group"
                >
                  <div className="text-sm font-medium text-white mb-1">Unusual Activity</div>
                  <div className="text-xs text-gray-500 group-hover:text-gray-400">
                    Is SPY showing unusual activity?
                  </div>
                </button>
                <button
                  onClick={() => setInput("What are technical levels for NVDA?")}
                  className="p-4 text-left bg-gray-900 hover:bg-gray-800 border border-gray-800 hover:border-gray-700 rounded-xl transition group"
                >
                  <div className="text-sm font-medium text-white mb-1">Technical Analysis</div>
                  <div className="text-xs text-gray-500 group-hover:text-gray-400">
                    What are technical levels for NVDA?
                  </div>
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`flex space-x-3 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    {/* Avatar */}
                    <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                      msg.role === 'user'
                        ? 'bg-blue-600'
                        : 'bg-gradient-to-br from-blue-500 to-purple-600'
                    }`}>
                      <span className="text-white text-sm font-semibold">
                        {msg.role === 'user' ? 'You' : 'AI'}
                      </span>
                    </div>

                    {/* Message Content */}
                    <div className="flex-1">
                      <div className={`rounded-2xl px-4 py-3 ${
                        msg.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-900 text-gray-100 border border-gray-800'
                      }`}>
                        <div className="prose prose-invert max-w-none">
                          <div className="text-sm leading-relaxed whitespace-pre-wrap">
                            {msg.content}
                          </div>
                        </div>
                      </div>
                      <div className="mt-1 px-4 text-xs text-gray-600">
                        {new Date(msg.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Thinking indicator */}
              {loading && (
                <div className="flex justify-start">
                  <div className="flex space-x-3 max-w-[85%]">
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                      <span className="text-white text-sm font-semibold">AI</span>
                    </div>
                    <div className="flex-1">
                      <div className="rounded-2xl px-4 py-3 bg-gray-900 border border-gray-800">
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
                          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }} />
                          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }} />
                          <span className="text-sm text-gray-400 ml-2">Analyzing market data...</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input Area - Fixed at bottom */}
      <div className="border-t border-gray-800 bg-[#0d0d0d]">
        <div className="max-w-3xl mx-auto px-6 py-4">
          <div className="relative flex items-end space-x-3">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about stocks, options, technical analysis..."
                disabled={loading}
                rows={1}
                className="w-full bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 pr-12 text-white placeholder-gray-600 focus:outline-none focus:border-gray-700 resize-none max-h-32 disabled:opacity-50"
                style={{ minHeight: '48px' }}
              />
              <div className="absolute right-3 bottom-3 text-xs text-gray-600">
                {input.length > 0 && `${input.length} chars`}
              </div>
            </div>
            <button
              onClick={sendMessage}
              disabled={!input.trim() || loading}
              className="flex-shrink-0 w-12 h-12 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-800 disabled:cursor-not-allowed rounded-xl font-semibold transition flex items-center justify-center"
            >
              <svg
                className="w-5 h-5 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
          <div className="mt-2 text-xs text-gray-600 text-center">
            Press Enter to send • Shift + Enter for new line
          </div>
        </div>
      </div>
    </div>
  );
}
