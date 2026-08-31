"use client";

import { useState, useRef, useEffect } from "react";

type Message = {
  role: "user" | "ai";
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [tenantId, setTenantId] = useState("company_A");
  const [isLoading, setIsLoading] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userQuestion = input;
    setInput(""); 
    setIsLoading(true);

    setMessages((prev) => [...prev, { role: "user", content: userQuestion }]);
    setMessages((prev) => [...prev, { role: "ai", content: "" }]);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { 
            "Content-Type": "application/json",
            "Authorization": "Bearer supersecret123" 
        },
        body: JSON.stringify({
          question: userQuestion,
          tenant_id: tenantId,
        }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        setMessages((prev) => {
          const newMessages = [...prev];
          const lastIndex = newMessages.length - 1;
          newMessages[lastIndex] = {
            ...newMessages[lastIndex],
            content: newMessages[lastIndex].content + chunk.replace(/\*/g, "")
          };
          return newMessages;
        });
      }
    } catch (error) {
      console.error("Chat Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: "Sorry, there was an error connecting to the server." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-800 font-sans">
      {/* HEADER */}
      <header className="bg-white shadow-sm p-4 flex justify-between items-center z-10">
        <h1 className="text-xl font-bold text-blue-600">Enterprise RAG Assistant</h1>
        <div className="flex items-center space-x-2">
          <label className="text-sm font-semibold text-gray-500">Current Tenant:</label>
          <select
            value={tenantId}
            onChange={(e) => setTenantId(e.target.value)}
            disabled={isLoading}
            className="border border-gray-300 rounded-md p-1 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
          >
            <option value="company_A">Company A (Valve & AWS)</option>
            <option value="company_B">Company B (Apple)</option>
          </select>
        </div>
      </header>

      {/* CHAT MESSAGES AREA */}
      <main className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            Select a tenant and ask a question to your company data!
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-3xl rounded-lg p-4 shadow-sm ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-white border border-gray-200 whitespace-pre-wrap"
                }`}
              >
                <span className="font-bold text-xs uppercase opacity-50 block mb-1">
                  {msg.role === "user" ? "You" : "Assistant"}
                </span>

                {/* ANIMATION LOGIC */}
                {msg.role === "ai" && msg.content === "" && isLoading ? (
                  /* 1. BOUNCING DOTS (Waiting for first token) */
                  <div className="flex space-x-1.5 h-6 items-center px-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                ) : (
                  /* 2. TEXT & BLINKING CURSOR (Typing effect) */
                  <div>
                    {msg.content}
                    {msg.role === "ai" && isLoading && index === messages.length - 1 && (
                      <span className="inline-block w-2 h-4 ml-1 bg-gray-500 animate-pulse align-middle rounded-sm"></span>
                    )}
                  </div>
                )}

              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* INPUT AREA */}
      <footer className="bg-white border-t border-gray-200 p-4">
        <form
          onSubmit={sendMessage}
          className="max-w-4xl mx-auto flex items-center space-x-4"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
            placeholder={
              isLoading ? "AI is typing..." : "Ask a question about the company data..."
            }
            className="flex-1 border border-gray-300 rounded-full px-6 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 text-white rounded-full px-6 py-3 font-semibold hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            Send
          </button>
        </form>
      </footer>
    </div>
  );
}