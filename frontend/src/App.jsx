import { useState } from 'react'

// In production this should come from an environment variable
// (import.meta.env.VITE_API_URL) instead of being hardcoded — set that up
// when you deploy, so the frontend can point at your live backend URL.
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [question, setQuestion] = useState('')
  const [history, setHistory] = useState([]) // [{ question, answer }]
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!question.trim() || loading) return

    setLoading(true)
    setError(null)
    const currentQuestion = question
    setQuestion('')

    try {
      const historyPayload = history.flatMap((item) => [
        { role: 'user', content: item.question },
        { role: 'assistant', content: item.answer },
      ])

      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: currentQuestion, history: historyPayload }),
      })


      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`)
      }

      const data = await response.json()
      setHistory((prev) => [...prev, { question: currentQuestion, answer: data.answer }])
    } catch (err) {
      setError('Something went wrong talking to the agent. Is the backend running?')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <h1>AI Research Agent</h1>
      <p className="subtitle">
        Ask something that needs current information or a calculation — the
        agent will decide which tools to use.
      </p>

      <div className="history">
        {history.map((item, i) => (
          <div key={i} className="exchange">
            <div className="question">{item.question}</div>
            <div className="answer">{item.answer}</div>
          </div>
        ))}
        {loading && <div className="thinking">Agent is thinking…</div>}
      </div>

      {error && <div className="error">{error}</div>}

      <form onSubmit={handleSubmit} className="input-row">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a research question…"
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? '…' : 'Ask'}
        </button>
      </form>
    </div>
  )
}

export default App
