import React, { useEffect, useMemo, useRef, useState } from 'react';

const WEEKDAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                'August', 'September', 'October', 'November', 'December'];

const pad = (n) => String(n).padStart(2, '0');

/** Local date parts → "YYYY-MM-DD". Avoids Date's UTC string parsing entirely. */
const iso = (year, monthIndex, day) => `${year}-${pad(monthIndex + 1)}-${pad(day)}`;

const daysInMonth = (year, monthIndex) => new Date(year, monthIndex + 1, 0).getDate();
const firstWeekdayOf = (year, monthIndex) => new Date(year, monthIndex, 1).getDay();

/** "2026-08-13" → {year, monthIndex}, falling back to today when unparseable. */
function monthOf(isoDate) {
  const [y, m] = (isoDate || '').split('-').map(Number);
  const now = new Date();
  return Number.isFinite(y) && Number.isFinite(m)
    ? { year: y, monthIndex: m - 1 }
    : { year: now.getFullYear(), monthIndex: now.getMonth() };
}

/**
 * Calendar date picker for the archive.
 *
 * Chrome's native date-input calendar cannot be scrolled between months and offers
 * no way to jump a year, which over a 16-year archive means clicking its month arrow
 * a hundred times. This one navigates by mouse wheel, by month/year dropdown, and by
 * arrow buttons.
 *
 * Availability comes from the caller's already-loaded date list, so days with no
 * scraped headlines are greyed out without any extra request or API change.
 *
 * @param {string[]} dates Every date that has headlines, newest first.
 * @param {string} value Currently selected date, "YYYY-MM-DD".
 * @param {Function} onChange Called with the newly picked date.
 * @returns {JSX.Element} The trigger button plus its calendar popover.
 */
export default function DatePicker({ dates, value, onChange }) {
  const [open, setOpen] = useState(false);
  const [view, setView] = useState(() => monthOf(value));
  const rootRef = useRef(null);

  const available = useMemo(() => new Set(dates), [dates]);
  const newest = dates[0];
  const oldest = dates[dates.length - 1];
  const firstYear = oldest ? Number(oldest.slice(0, 4)) : new Date().getFullYear();
  const lastYear = newest ? Number(newest.slice(0, 4)) : firstYear;
  const years = useMemo(
    () => Array.from({ length: lastYear - firstYear + 1 }, (_, i) => lastYear - i),
    [firstYear, lastYear],
  );

  // Follow the selection when it changes from outside (the day steppers).
  useEffect(() => {
    if (value) {
      setView(monthOf(value));
    }
  }, [value]);

  // Close on outside click or Escape so the popover never traps the page. Position
  // needs no JS: the panel is absolutely positioned against this component, so it
  // tracks the trigger through ordinary layout — no scroll listener to miss.
  useEffect(() => {
    if (!open) {
      return undefined;
    }
    const onDown = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const shiftMonth = (delta) => {
    setView(({ year, monthIndex }) => {
      const n = monthIndex + delta;
      return { year: year + Math.floor(n / 12), monthIndex: ((n % 12) + 12) % 12 };
    });
  };

  const { year, monthIndex } = view;
  const lead = firstWeekdayOf(year, monthIndex);
  const total = daysInMonth(year, monthIndex);
  const cells = [
    ...Array.from({ length: lead }, () => null),
    ...Array.from({ length: total }, (_, i) => i + 1),
  ];

  const monthStart = iso(year, monthIndex, 1);
  const monthEnd = iso(year, monthIndex, total);
  const canGoBack = !oldest || monthStart > oldest;
  const canGoForward = !newest || monthEnd < newest;

  // Day stepping lives here too, so the whole control is one component and the
  // popover can be a SIBLING of the (overflow:hidden) input group rather than a
  // child of it.
  const index = dates.indexOf(value);
  const hasOlder = index >= 0 && index < dates.length - 1;
  const hasNewer = index > 0;
  const step = (delta) => {
    const next = dates[index + delta];
    if (next) {
      onChange(next);
    }
  };

  return (
    <div className="ss-datepicker" ref={rootRef}>
      <div className="ss-input-group">
        <button
          type="button"
          className="ss-step-btn"
          onClick={() => step(1)}
          disabled={!hasOlder}
          aria-label="Previous date with headlines"
          title="Previous date with headlines"
        >
          ‹
        </button>
        <button
          type="button"
          className="ss-datepicker__trigger"
          onClick={() => setOpen((o) => !o)}
          aria-haspopup="dialog"
          aria-expanded={open}
        >
          {value || 'Pick a date'}
        </button>
        <button
          type="button"
          className="ss-step-btn"
          onClick={() => step(-1)}
          disabled={!hasNewer}
          aria-label="Next date with headlines"
          title="Next date with headlines"
        >
          ›
        </button>
      </div>

      {open ? (
        <div
          className="ss-calendar"
          role="dialog"
          aria-label="Choose a date"
          // The whole point: a wheel over the calendar moves months. Chrome's native
          // picker ignores the wheel entirely, which is what sent us here.
          onWheel={(e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 1 : -1;
            if ((delta > 0 && canGoForward) || (delta < 0 && canGoBack)) {
              shiftMonth(delta);
            }
          }}
        >
          <div className="ss-calendar__head">
            <button
              type="button"
              className="ss-step-btn"
              onClick={() => shiftMonth(-1)}
              disabled={!canGoBack}
              aria-label="Previous month"
              title="Previous month"
            >
              ‹
            </button>

            {/* Dropdowns rather than only arrows: 2010 is 190 months from 2026. */}
            <span className="ss-calendar__pickers">
              <select
                value={monthIndex}
                onChange={(e) => setView((v) => ({ ...v, monthIndex: Number(e.target.value) }))}
                aria-label="Month"
              >
                {MONTHS.map((m, i) => <option key={m} value={i}>{m}</option>)}
              </select>
              <select
                value={year}
                onChange={(e) => setView((v) => ({ ...v, year: Number(e.target.value) }))}
                aria-label="Year"
              >
                {years.map((y) => <option key={y} value={y}>{y}</option>)}
              </select>
            </span>

            <button
              type="button"
              className="ss-step-btn"
              onClick={() => shiftMonth(1)}
              disabled={!canGoForward}
              aria-label="Next month"
              title="Next month"
            >
              ›
            </button>
          </div>

          <div className="ss-calendar__grid">
            {WEEKDAYS.map((d, i) => (
              <span key={`${d}-${i}`} className="ss-calendar__weekday" aria-hidden="true">{d}</span>
            ))}

            {cells.map((day, i) => {
              if (day === null) {
                return <span key={`pad-${i}`} className="ss-calendar__pad" />;
              }
              const dateStr = iso(year, monthIndex, day);
              const has = available.has(dateStr);
              const selected = dateStr === value;
              return (
                <button
                  key={dateStr}
                  type="button"
                  className={`ss-calendar__day${selected ? ' is-selected' : ''}`}
                  disabled={!has}
                  onClick={() => { onChange(dateStr); setOpen(false); }}
                  title={has ? dateStr : `${dateStr} · no headlines`}
                  aria-current={selected ? 'date' : undefined}
                >
                  {day}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}
