import React, { useEffect, useState, useCallback } from 'react';
import { getJson } from '../lib/api.js';
import { signedScore } from '../lib/format.js';
import HeadlineList from './HeadlineList.jsx';

const PAGE_SIZE = 50;

// Mirrors ui/queries.py SORT_KEYS. The direction labels differ per sort because
// "descending hour" and "descending sentiment" read as very different questions.
const SORTS = [
  { key: 'time', label: 'Time published', desc: 'Newest first', asc: 'Oldest first' },
  { key: 'sentiment', label: 'Sentiment', desc: 'Most positive first', asc: 'Most negative first' },
  { key: 'politics', label: 'Politics relevance', desc: 'Highest first', asc: 'Lowest first' },
  { key: 'economy', label: 'Economy relevance', desc: 'Highest first', asc: 'Lowest first' },
  { key: 'security', label: 'Security relevance', desc: 'Highest first', asc: 'Lowest first' },
  { key: 'health', label: 'Health relevance', desc: 'Highest first', asc: 'Lowest first' },
  { key: 'science', label: 'Science relevance', desc: 'Highest first', asc: 'Lowest first' },
  { key: 'technology', label: 'Technology relevance', desc: 'Highest first', asc: 'Lowest first' },
];

// Mirrors ui/queries.py CATEGORY_KEYS.
const CATEGORIES = [
  ['politics', 'Politics'],
  ['economy', 'Economy'],
  ['security', 'Security'],
  ['health', 'Health'],
  ['science', 'Science'],
  ['technology', 'Technology'],
];

const SENTIMENT_LEVELS = Array.from({ length: 21 }, (_, i) => 10 - i); // +10 → −10
const RELEVANCE_LEVELS = Array.from({ length: 10 }, (_, i) => i + 1); // 1 → 10

/**
 * Archive view: pick a date from /api/dates, then page through that date's
 * headlines via /api/headlines using total/page_size for prev/next paging.
 *
 * Search, score filters and sorting are all applied server-side across the whole
 * date — never over the rows already on screen — so the counts and the ordering
 * describe the day rather than the current page.
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
  const [sort, setSort] = useState('time');
  const [order, setOrder] = useState('desc');
  // '' means "no bound". Selects rather than number inputs: the scales are short
  // and closed, so this applies instantly and cannot produce an invalid value.
  const [sentimentMin, setSentimentMin] = useState('');
  const [sentimentMax, setSentimentMax] = useState('');
  const [category, setCategory] = useState('');
  const [categoryMin, setCategoryMin] = useState('1');

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

  const loadHeadlines = useCallback(async (date, pageNum, opts) => {
    if (!date) {
      return;
    }
    setLoading(true);
    try {
      const params = new URLSearchParams({
        date,
        page: String(pageNum),
        page_size: String(PAGE_SIZE),
        sort: opts.sort,
        order: opts.order,
      });
      if (opts.search) params.set('q', opts.search);
      if (opts.sentimentMin !== '') params.set('sentiment_min', opts.sentimentMin);
      if (opts.sentimentMax !== '') params.set('sentiment_max', opts.sentimentMax);
      if (opts.category) {
        params.set('category', opts.category);
        params.set('category_min', opts.categoryMin);
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
  // The selects below are applied immediately — there is nothing to settle.
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
    loadHeadlines(selectedDate, page, {
      search: query, sort, order, sentimentMin, sentimentMax, category, categoryMin,
    });
  }, [selectedDate, page, query, sort, order, sentimentMin, sentimentMax, category,
      categoryMin, loadHeadlines]);

  const onDateChange = (e) => {
    setPage(0);
    setSelectedDate(e.target.value);
  };

  // Every score control changes which rows match, so the current page number is
  // meaningless afterwards — page 4 of the old result set is rarely page 4 of the new one.
  const onSortChange = (e) => {
    setPage(0);
    setSort(e.target.value);
  };
  const onOrderChange = (e) => {
    setPage(0);
    setOrder(e.target.value);
  };
  const onSentimentMinChange = (e) => {
    setPage(0);
    setSentimentMin(e.target.value);
  };
  const onSentimentMaxChange = (e) => {
    setPage(0);
    setSentimentMax(e.target.value);
  };
  const onCategoryChange = (e) => {
    setPage(0);
    setCategory(e.target.value);
  };
  const onCategoryMinChange = (e) => {
    setPage(0);
    setCategoryMin(e.target.value);
  };

  const resetScores = () => {
    setPage(0);
    setSort('time');
    setOrder('desc');
    setSentimentMin('');
    setSentimentMax('');
    setCategory('');
    setCategoryMin('1');
  };

  const activeSort = SORTS.find((s) => s.key === sort) || SORTS[0];
  const scoresFiltered = sentimentMin !== '' || sentimentMax !== '' || Boolean(category);
  const scoresTouched = scoresFiltered || sort !== 'time' || order !== 'desc';
  // A sentiment window with the bounds crossed can never match; say so rather than
  // letting an empty result read as "this day has no negative news".
  const impossibleRange = sentimentMin !== '' && sentimentMax !== ''
    && Number(sentimentMin) > Number(sentimentMax);

  const total = data?.total ?? 0;
  const pageSize = data?.page_size ?? PAGE_SIZE;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  // `total` now reflects the search AND the score filters, so the range and page
  // count describe the matches rather than the whole day.
  const visibleHeadlines = data?.headlines || [];
  const firstOnPage = total === 0 ? 0 : page * pageSize + 1;
  const lastOnPage = Math.min(total, page * pageSize + visibleHeadlines.length);
  const searching = Boolean(query);

  // Spelled out in one line under the controls so an empty or surprising result is
  // always traceable to the filter that caused it.
  const criteria = [];
  if (searching) criteria.push(`matching “${query}”`);
  if (category) {
    const name = (CATEGORIES.find(([k]) => k === category) || [, category])[1];
    criteria.push(`${name.toLowerCase()} relevance ≥ ${categoryMin}`);
  }
  if (sentimentMin !== '' && sentimentMax !== '') {
    criteria.push(`sentiment ${signedScore(Number(sentimentMin))}…${signedScore(Number(sentimentMax))}`);
  } else if (sentimentMin !== '') {
    criteria.push(`sentiment ≥ ${signedScore(Number(sentimentMin))}`);
  } else if (sentimentMax !== '') {
    criteria.push(`sentiment ≤ ${signedScore(Number(sentimentMax))}`);
  }

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

      <div className="ss-controls ss-archive-scorebar">
        <label className="ss-field">
          Sort by
          <select value={sort} onChange={onSortChange}>
            {SORTS.map((s) => (
              <option key={s.key} value={s.key}>{s.label}</option>
            ))}
          </select>
        </label>
        <label className="ss-field">
          Direction
          <select value={order} onChange={onOrderChange}>
            <option value="desc">{activeSort.desc}</option>
            <option value="asc">{activeSort.asc}</option>
          </select>
        </label>

        <div className="ss-field ss-archive-range">
          Sentiment between
          <div className="ss-archive-range-row">
            <select value={sentimentMin} onChange={onSentimentMinChange} aria-label="Minimum sentiment">
              <option value="">Any</option>
              {SENTIMENT_LEVELS.map((n) => (
                <option key={n} value={n}>{signedScore(n)}</option>
              ))}
            </select>
            <span aria-hidden="true">–</span>
            <select value={sentimentMax} onChange={onSentimentMaxChange} aria-label="Maximum sentiment">
              <option value="">Any</option>
              {SENTIMENT_LEVELS.map((n) => (
                <option key={n} value={n}>{signedScore(n)}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="ss-field ss-archive-range">
          Category relevance
          <div className="ss-archive-range-row">
            <select value={category} onChange={onCategoryChange} aria-label="Category">
              <option value="">Any category</option>
              {CATEGORIES.map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
            <select
              value={categoryMin}
              onChange={onCategoryMinChange}
              disabled={!category}
              aria-label="Minimum relevance"
            >
              {RELEVANCE_LEVELS.map((n) => (
                <option key={n} value={n}>≥ {n}</option>
              ))}
            </select>
          </div>
        </div>

        {scoresTouched ? (
          <button className="ss-btn ss-btn--ghost" onClick={resetScores}>
            Reset scores
          </button>
        ) : null}
      </div>

      <p className="ss-muted ss-archive-legend">
        Sentiment badges: <span className="ss-badge pos">+3</span> positive ·{' '}
        <span className="ss-badge neg">−2</span> negative ·{' '}
        <span className="ss-badge neutral">n/a</span> gray = unscored
        {scoresFiltered ? ' · score filters exclude unscored headlines' : ''}
      </p>

      {error ? <p className="ss-error-text">Error: {error}</p> : null}
      {impossibleRange ? (
        <p className="ss-error-text">
          The sentiment range is inverted ({signedScore(Number(sentimentMin))} is above{' '}
          {signedScore(Number(sentimentMax))}), so nothing can match.
        </p>
      ) : null}

      {data ? (
        // Hold the previous result at reduced opacity while refetching rather
        // than swapping in a "Loading…" line, which shifted the layout on every
        // keystroke and page change.
        <div className={loading ? 'is-refetching' : undefined}>
          <p className="ss-muted ss-archive-count">
            {total === 0
              ? (criteria.length
                ? `No headlines on this date with ${criteria.join(' and ')}.`
                : 'No headlines for this date.')
              : (
                <>
                  {firstOnPage}–{lastOnPage} of {total}
                  {criteria.length
                    ? <> headlines {criteria.join(' and ')} on this date</>
                    : ' headlines'}
                  {sort !== 'time'
                    ? <> · sorted by {activeSort.label.toLowerCase()}, {(order === 'desc'
                      ? activeSort.desc : activeSort.asc).toLowerCase()}</>
                    : null}
                </>
              )}
          </p>
          {pager}
          <HeadlineList headlines={visibleHeadlines} highlight={sort === 'time' ? null : sort} />
          {pager}
        </div>
      ) : null}
    </div>
  );
}
