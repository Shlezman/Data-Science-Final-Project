import React, { useEffect, useState, useCallback } from 'react';
import { getJson } from '../lib/api.js';
import HeadlineList from './HeadlineList.jsx';

const PAGE_SIZE = 50;

/**
 * Archive view: pick a date from /api/dates, then page through that date's
 * headlines via /api/headlines using total/page_size for prev/next paging.
 * Includes a client-side substring filter over the currently loaded page.
 *
 * @returns {JSX.Element} The archive browser.
 */
export default function Archive() {
  const [dates, setDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState('');
  const [page, setPage] = useState(1);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    getJson('/api/dates')
      .then((res) => {
        const list = res?.dates || [];
        setDates(list);
        if (list.length > 0) {
          setSelectedDate(list[0]);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  const loadHeadlines = useCallback(async (date, pageNum) => {
    if (!date) {
      return;
    }
    setLoading(true);
    try {
      const params = new URLSearchParams({
        date,
        page: String(pageNum),
        page_size: String(PAGE_SIZE),
      });
      const res = await getJson(`/api/headlines?${params.toString()}`);
      setData(res);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHeadlines(selectedDate, page);
  }, [selectedDate, page, loadHeadlines]);

  const onDateChange = (e) => {
    setPage(1);
    setSelectedDate(e.target.value);
  };

  const total = data?.total ?? 0;
  const pageSize = data?.page_size ?? PAGE_SIZE;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  // Client-side substring filter over the currently loaded page only
  // (case-insensitive; plain includes works for Hebrew text too).
  const pageHeadlines = data?.headlines || [];
  const needle = filter.trim().toLowerCase();
  const visibleHeadlines = needle
    ? pageHeadlines.filter(
        (h) =>
          (h.headline || '').toLowerCase().includes(needle) ||
          (h.source || '').toLowerCase().includes(needle),
      )
    : pageHeadlines;

  return (
    <div className="ss-card">
      <h2>Archive</h2>
      <div className="ss-controls">
        <label className="ss-field">
          Date
          <select value={selectedDate} onChange={onDateChange}>
            {dates.length === 0 ? <option value="">No dates</option> : null}
            {dates.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label className="ss-field ss-archive-filter">
          Filter this page
          <input
            type="text"
            value={filter}
            placeholder="Headline or source…"
            onChange={(e) => setFilter(e.target.value)}
          />
        </label>
        {filter ? (
          <button className="ss-btn ss-btn--ghost" onClick={() => setFilter('')}>
            Clear
          </button>
        ) : null}
      </div>

      <p className="ss-muted ss-archive-legend">
        Sentiment badges: <span className="ss-badge pos">+3</span> positive ·{' '}
        <span className="ss-badge neg">−2</span> negative ·{' '}
        <span className="ss-badge neutral">n/a</span> gray = unscored
      </p>

      {error ? <p className="ss-error-text">Error: {error}</p> : null}
      {loading ? <p className="ss-muted">Loading…</p> : null}

      {data ? (
        <>
          <p className="ss-muted ss-archive-count">
            Showing {visibleHeadlines.length} of {pageHeadlines.length} on this
            page
          </p>
          {needle && visibleHeadlines.length === 0 ? (
            <p className="ss-muted">No headlines match the filter on this page.</p>
          ) : (
            <HeadlineList headlines={visibleHeadlines} />
          )}
          <div className="ss-pager">
            <button
              className="ss-btn secondary"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              Prev
            </button>
            <span>
              Page {data.page ?? page} of {totalPages} · {total} headlines
            </span>
            <button
              className="ss-btn secondary"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              Next
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}
