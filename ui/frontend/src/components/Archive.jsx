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
  // Zero-based, matching the API's `offset = page * page_size`. It used to
  // start at 1, so every day's first page silently began at offset 50 and the
  // 50 most recent headlines of each date were unreachable.
  const [page, setPage] = useState(0);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('');
  const [query, setQuery] = useState('');

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

  const loadHeadlines = useCallback(async (date, pageNum, search) => {
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
      if (search) {
        params.set('q', search);
      }
      const res = await getJson(`/api/headlines?${params.toString()}`);
      setData(res);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounce the box so a request goes out once typing settles, not per keystroke.
  useEffect(() => {
    const id = setTimeout(() => {
      setQuery((prev) => {
        const next = filter.trim();
        if (next !== prev) {
          setPage(0);
        }
        return next;
      });
    }, 350);
    return () => clearTimeout(id);
  }, [filter]);

  useEffect(() => {
    loadHeadlines(selectedDate, page, query);
  }, [selectedDate, page, query, loadHeadlines]);

  const onDateChange = (e) => {
    setPage(0);
    setSelectedDate(e.target.value);
  };

  const total = data?.total ?? 0;
  const pageSize = data?.page_size ?? PAGE_SIZE;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  // `total` now reflects the search, so the range and page count describe the
  // matches rather than the whole day.
  const visibleHeadlines = data?.headlines || [];
  const firstOnPage = total === 0 ? 0 : page * pageSize + 1;
  const lastOnPage = Math.min(total, page * pageSize + visibleHeadlines.length);
  const searching = Boolean(query);

  // Rendered above AND below the list: a page holds 50 rows, so after reading to
  // the bottom the controls are in reach, and after changing pages the controls
  // are still where you left them at the top.
  const pager = totalPages > 1 ? (
    <div className="ss-pager">
      <button
        className="ss-btn secondary"
        disabled={page <= 0}
        onClick={() => setPage((p) => Math.max(0, p - 1))}
      >
        Prev
      </button>
      <span>
        Page {page + 1} of {totalPages}
      </span>
      <button
        className="ss-btn secondary"
        disabled={page >= totalPages - 1}
        onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
      >
        Next
      </button>
    </div>
  ) : null;

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
          Search this date
          <input
            type="search"
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

      {data ? (
        // Hold the previous result at reduced opacity while refetching rather
        // than swapping in a "Loading…" line, which shifted the layout on every
        // keystroke and page change.
        <div className={loading ? 'is-refetching' : undefined}>
          <p className="ss-muted ss-archive-count">
            {total === 0
              ? (searching ? `No headlines match “${query}” on this date.` : 'No headlines for this date.')
              : (
                <>
                  {firstOnPage}–{lastOnPage} of {total}
                  {searching ? <> matching “{query}” on this date</> : ' headlines'}
                </>
              )}
          </p>
          {pager}
          <HeadlineList headlines={visibleHeadlines} />
          {pager}
        </div>
      ) : null}
    </div>
  );
}
