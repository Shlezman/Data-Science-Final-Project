import React, { useEffect, useRef, useState } from 'react';
import { getJson, postJson } from '../lib/api.js';

const POLL_MS = 2500;
const TIMEOUT_MS = 180_000;

/**
 * "Ask the analyst" — a live LLM panel over the day's headlines.
 *
 * The GPU box's LLM is not directly reachable from the UI host (firewall), so
 * requests go through the database queue: POST /api/llm/ask inserts a row, a
 * worker on the GPU box answers it with the local model, and this panel polls
 * GET /api/llm/answer until the answer lands.
 *
 * @param {object} props Component props.
 * @param {string} props.date The selected day (YYYY-MM-DD).
 * @returns {JSX.Element} The analyst card.
 */
export default function AnalystPanel({ date }) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [state, setState] = useState('idle');   // idle | waiting | done | error
  const [message, setMessage] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const submit = async (kind) => {
    if (!date || state === 'waiting') return;
    if (kind === 'ask' && !question.trim()) return;
    setState('waiting');
    setAnswer(null);
    setMessage(null);
    try {
      const req = await postJson('/api/llm/ask', {
        kind, date, question: kind === 'ask' ? question.trim() : undefined,
      });
      const startedAt = Date.now();
      clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const row = await getJson(`/api/llm/answer?id=${req.id}`);
          if (row.status === 'done') {
            clearInterval(pollRef.current);
            setAnswer(row.answer);
            setState('done');
          } else if (row.status === 'error') {
            clearInterval(pollRef.current);
            setMessage(row.answer || 'The analyst worker reported an error.');
            setState('error');
          } else if (Date.now() - startedAt > TIMEOUT_MS) {
            clearInterval(pollRef.current);
            setMessage('Timed out — the analyst worker may be offline on the GPU box.');
            setState('error');
          }
        } catch (err) {
          clearInterval(pollRef.current);
          setMessage(err.message);
          setState('error');
        }
      }, POLL_MS);
    } catch (err) {
      setMessage(err.message);
      setState('error');
    }
  };

  return (
    <div className="ss-card">
      <h3>Ask the analyst (LLM)</h3>
      <p className="ss-muted">
        The local language model reads the selected day&apos;s headlines and answers live.
        Narrate summarizes the day&apos;s narratives with an UP/DOWN lean; or ask your own question.
      </p>
      <div className="ss-controls">
        <button className="ss-btn" onClick={() => submit('narrate')}
                disabled={!date || state === 'waiting'}>
          {state === 'waiting' ? 'Thinking…' : `Narrate ${date || 'the day'}`}
        </button>
        <span className="ss-analyst__question">
          <input type="text" value={question} placeholder="e.g. What drove security-related news today?"
                 onChange={(e) => setQuestion(e.target.value)}
                 onKeyDown={(e) => { if (e.key === 'Enter') submit('ask'); }} />
        </span>
        <button className="ss-btn" onClick={() => submit('ask')}
                disabled={!date || !question.trim() || state === 'waiting'}>
          Ask
        </button>
      </div>
      {state === 'waiting' ? (
        <p className="ss-muted">Waiting for the model… (runs on the GPU box, usually under a minute)</p>
      ) : null}
      {state === 'error' ? <p className="ss-error-text">{message}</p> : null}
      {answer ? <div className="ss-analyst__answer">{answer}</div> : null}
    </div>
  );
}
