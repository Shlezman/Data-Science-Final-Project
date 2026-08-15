import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { getJson } from '../lib/api.js';
import DatePicker from './DatePicker.jsx';
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

const SENTIMENT_FILTERS = [
  { value: 'any', label: 'Any sentiment', min: '', max: '' },
  { value: 'positive', label: 'Positive (+1 to +10)', min: '1', max: '10' },
  { value: 'neutral', label: 'Neutral (0)', min: '0', max: '0' },
  { value: 'negative', label: 'Negative (−10 to −1)', min: '-10', max: '-1' },
];
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
  // Set when a picked day had no headlines and the nearest one was used instead.
  const [dateNote, setDateNote] = useState(null);
  // Mirrors `page` so the jump box can be typed into freely; it commits only on
  // Enter or blur, so a half-typed "1" on the way to "12" doesn't fire a request.
  const [pageInput, setPageInput] = useState('1');
  const [sentimentFilter, setSentimentFilter] = useState('any');
  const [category, setCategory] = useState('');
  const [categoryMin, setCategoryMin] = useState('1');
  const activeSentiment = SENTIMENT_FILTERS.find(({ value }) => value === sentimentFilter)
    || SENTIMENT_FILTERS[0];
  const sentimentMin = activeSentiment.min;
  const sentimentMax = activeSentiment.max;

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

  // Keep the jump box in step with pages changed by any other control (Prev/Next,
  // First/Last, or a filter resetting to page 1).
  useEffect(() => {
    setPageInput(String(page + 1));
  }, [page]);

  // `dates` arrives newest-first. Used only to resolve a date that isn't in the list;
  // stepping between dates lives in DatePicker.
  const dateIndex = useMemo(() => new Map(dates.map((d, i) => [d, i])), [dates]);

  // The calendar disables days with no headlines, so a pick normally lands on real
  // data. The fallback stays for safety: anything unknown resolves to the newest
  // date before it rather than rendering an empty page.
  const onDatePicked = (picked) => {
    if (!picked) {
      return;
    }
    setPage(0);
    if (dateIndex.has(picked)) {
      setDateNote(null);
      setSelectedDate(picked);
      return;
    }
    const snapped = dates.find((d) => d <= picked) || dates[dates.length - 1];
    if (!snapped) {
      return;
    }
    setSelectedDate(snapped);
    setDateNote(`No headlines on ${picked} — showing ${snapped}.`);
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
  const onSentimentChange = (e) => {
    setPage(0);
    setSentimentFilter(e.target.value);
  };
  const onCategoryChange = (e) => {
    setPage(0);
    setCategory(e.target.value);
  };
  const onCategoryMinChange = (e) => {
    setPage(0);
    setCategoryMin(e.target.value);
  };

  // Clears the search too, so one button empties the whole toolbar rather than
  // leaving a search the user has to hunt down separately.
  const resetAll = () => {
    setPage(0);
    setFilter('');
    setSort('time');
    setOrder('desc');
    setSentimentFilter('any');
    setCategory('');
    setCategoryMin('1');
  };

  const activeSort = SORTS.find((s) => s.key === sort) || SORTS[0];
  const scoresFiltered = sentimentFilter !== 'any' || Boolean(category);
  const scoresTouched = scoresFiltered || sort !== 'time' || order !== 'desc';
  const anyActive = scoresTouched || filter !== '';

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
  if (sentimentFilter !== 'any') criteria.push(`${activeSentiment.label.split(' (')[0].toLowerCase()} sentiment`);

  const commitPageInput = () => {
    const parsed = Number.parseInt(pageInput, 10);
    if (!Number.isFinite(parsed)) {
      setPageInput(String(page + 1));
      return;
    }
    const clamped = Math.min(Math.max(parsed, 1), totalPages);
    setPageInput(String(clamped));
    setPage(clamped - 1);
  };

  // Rendered above AND below the list: a page holds 50 rows, so after reading to
  // the bottom the controls are in reach, and after changing pages the controls
  // are still where you left them at the top.
  //
  // First/Last and the jump box matter because an unfiltered day runs to 16 pages;
  // reaching the oldest headline of a date used to cost 15 clicks on Next.
  const pager = totalPages > 1 ? (
    <div className="ss-pager">
      <button
        className="ss-btn secondary"
        disabled={page <= 0}
        onClick={() => setPage(0)}
        aria-label="First page"
        title="First page"
      >
        «
      </button>
      <button
        className="ss-btn secondary"
        disabled={page <= 0}
        onClick={() => setPage((p) => Math.max(0, p - 1))}
      >
        Prev
      </button>
      <span className="ss-pager__jump">
        Page
        <input
          type="number"
          min="1"
          max={totalPages}
          value={pageInput}
          onChange={(e) => setPageInput(e.target.value)}
          onBlur={commitPageInput}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              commitPageInput();
            }
          }}
          aria-label={`Page number, 1 to ${totalPages}`}
        />
        of {totalPages}
      </span>
      <button
        className="ss-btn secondary"
        disabled={page >= totalPages - 1}
        onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
      >
        Next
      </button>
      <button
        className="ss-btn secondary"
        disabled={page >= totalPages - 1}
        onClick={() => setPage(totalPages - 1)}
        aria-label="Last page"
        title="Last page"
      >
        »
      </button>
    </div>
  ) : null;

  return (
    <div className="ss-card">
      <div className="ss-dashboard-section-head">
        <div className="ss-dashboard-section-head__copy">
          <span className="ss-dashboard-section-head__eyebrow">Historical headlines</span>
          <h2>Archive</h2>
          <p>Search and filter scored news by date, sentiment and category relevance.</p>
        </div>
      </div>
      {/* One panel for every control. Date/search used to float bare on the card
          while the score controls sat in a box, so two halves of the same toolbar
          wore different skins. Widths are fixed per field so the labels line up
          into columns instead of landing wherever the content ended. */}
      <div className="ss-archive-toolbar">
        <div className="ss-archive-toolbar__row">
          {/* A date field rather than a <select>: the picker now covers every date on
              record, and 5,800+ options is a list you scroll, not one you use. Typing
              or opening the browser's calendar reaches any year directly. */}
          <div className="ss-field ss-field--date">
            Date
            <DatePicker dates={dates} value={selectedDate} onChange={onDatePicked} />
          </div>
          <label className="ss-field ss-archive-filter">
            Search this date
            <input
              type="search"
              value={filter}
              placeholder="Headline or source…"
              onChange={(e) => setFilter(e.target.value)}
            />
          </label>
        </div>

        <div className="ss-archive-toolbar__row ss-archive-toolbar__row--scores">
          <label className="ss-field ss-field--sort">
            Sort by
            <select value={sort} onChange={onSortChange}>
              {SORTS.map((s) => (
                <option key={s.key} value={s.key}>{s.label}</option>
              ))}
            </select>
          </label>
          <label className="ss-field ss-field--sort">
            Direction
            <select value={order} onChange={onOrderChange}>
              <option value="desc">{activeSort.desc}</option>
              <option value="asc">{activeSort.asc}</option>
            </select>
          </label>

          <label className="ss-field ss-field--sentiment">
            Sentiment
            <select
              value={sentimentFilter}
              onChange={onSentimentChange}
              aria-label="Sentiment"
            >
              {SENTIMENT_FILTERS.map(({ value, label }) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>

          <div className="ss-field">
            Category relevance
            <div className="ss-input-group">
              <select value={category} onChange={onCategoryChange}
                      className="ss-input-group__wide" aria-label="Category">
                <option value="">Any category</option>
                {CATEGORIES.map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>
              <select
                value={categoryMin}
                onChange={onCategoryMinChange}
                className="ss-input-group__narrow"
                disabled={!category}
                title={category ? undefined : 'Pick a category first'}
                aria-label="Minimum relevance"
              >
                {RELEVANCE_LEVELS.map((n) => (
                  <option key={n} value={n}>≥ {n}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Always rendered, disabled when there is nothing to clear. The two
              buttons this replaces appeared only once a control was touched, so
              the row jumped sideways mid-use. */}
          <button
            className="ss-btn ss-btn--ghost ss-archive-reset"
            onClick={resetAll}
            disabled={!anyActive}
          >
            Reset filters
          </button>
        </div>
      </div>

      {/* An actual key — swatch plus one word per term — rendered once. It replaced a
          sentence passed as a tooltip onto the "Sentiment" label of all 50 cards on
          the page. When score filters are on, "unscored" is struck through rather
          than explained in an extra clause. */}
      <dl className="ss-legend" aria-label="Sentiment badge key">
        <dt className="ss-legend__term">Sentiment</dt>
        <dd className="ss-legend__scale">−10…+10</dd>
        <dd className="ss-legend__item">
          <span className="ss-badge pos">+3</span> positive
        </dd>
        <dd className="ss-legend__item">
          <span className="ss-badge neutral">0</span> neutral
        </dd>
        <dd className="ss-legend__item">
          <span className="ss-badge neg">−2</span> negative
        </dd>
        <dd
          className={`ss-legend__item${scoresFiltered ? ' is-excluded' : ''}`}
          title={scoresFiltered ? 'Excluded by the active score filters' : undefined}
        >
          <span className="ss-badge neutral">n/a</span> unscored
        </dd>
      </dl>

      {dateNote ? <p className="ss-muted ss-archive-datenote">{dateNote}</p> : null}
      {error ? <p className="ss-error-text">Error: {error}</p> : null}
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
          <HeadlineList
            headlines={visibleHeadlines}
            highlight={sort === 'time' ? null : sort}
          />
          {pager}
        </div>
      ) : null}
    </div>
  );
}
