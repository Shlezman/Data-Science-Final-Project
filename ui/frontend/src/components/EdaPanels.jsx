import React, { useEffect, useMemo, useState } from 'react';
import { getJson } from '../lib/api.js';
import { Plot, darkLayout, PLOT_CONFIG, ACCENT } from '../lib/plotly.js';

const AXIS = {
  automargin: true,
};

const compactNumber = new Intl.NumberFormat('en', {
  notation: 'compact',
  maximumFractionDigits: 1,
});

function formatNumber(value) {
  return Number.isFinite(value) ? compactNumber.format(value) : '—';
}

function bucketDate(date, granularity) {
  if (granularity === 'day') return date;
  if (granularity === 'month') return `${date.slice(0, 7)}-01`;

  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return date;
  const day = parsed.getUTCDay() || 7;
  parsed.setUTCDate(parsed.getUTCDate() - day + 1);
  return parsed.toISOString().slice(0, 10);
}

function aggregateSeries(series, key, granularity, mode) {
  if (granularity === 'day') {
    return series.map((item) => ({ date: item.date, value: Number(item[key]) || 0 }));
  }

  const buckets = new Map();
  series.forEach((item) => {
    const date = bucketDate(item.date, granularity);
    const current = buckets.get(date) || { date, sum: 0, count: 0 };
    current.sum += Number(item[key]) || 0;
    current.count += 1;
    buckets.set(date, current);
  });

  return [...buckets.values()].map((item) => ({
    date: item.date,
    value: mode === 'mean' ? item.sum / item.count : item.sum,
  }));
}

function rollingAverage(series, windowSize) {
  return series.map((item, index) => {
    const start = Math.max(0, index - windowSize + 1);
    const window = series.slice(start, index + 1);
    const value = window.reduce((sum, point) => sum + point.value, 0) / window.length;
    return { date: item.date, value };
  });
}

function seriesAverage(series) {
  if (!series.length) return 0;
  return series.reduce((sum, point) => sum + point.value, 0) / series.length;
}

function isPartialPeriod(date, granularity) {
  if (!date || granularity === 'day') return false;
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return false;

  if (granularity === 'week') return parsed.getUTCDay() !== 0;

  const lastDay = new Date(Date.UTC(parsed.getUTCFullYear(), parsed.getUTCMonth() + 1, 0)).getUTCDate();
  return parsed.getUTCDate() !== lastDay;
}

function comparisonLabel(latest, average, formatter, mode = 'relative') {
  if (!Number.isFinite(latest) || !Number.isFinite(average)) return null;

  if (mode === 'difference') {
    const difference = latest - average;
    const direction = difference >= 0 ? 'above' : 'below';
    return `Latest ${formatter(latest)} · ${formatter(Math.abs(difference))} ${direction} avg`;
  }

  const delta = average ? ((latest - average) / Math.abs(average)) * 100 : 0;
  const direction = delta >= 0 ? 'above' : 'below';
  return `Latest ${formatter(latest)} · ${Math.abs(delta).toFixed(0)}% ${direction} avg`;
}

function percentHistogram(histogram) {
  const total = histogram.reduce((sum, item) => sum + (Number(item.count) || 0), 0);
  return histogram.map((item) => ({
    ...item,
    count: Number(item.count) || 0,
    percent: total ? ((Number(item.count) || 0) / total) * 100 : 0,
  }));
}

function histogramPeak(distribution) {
  return distribution.reduce((peak, item) => (
    !peak || item.percent > peak.percent ? item : peak
  ), null);
}

function percentScale(items) {
  const highest = Math.max(0, ...items.map((item) => Number(item.percent) || 0));
  const step = highest <= 12 ? 2 : highest <= 25 ? 5 : highest <= 50 ? 10 : 20;
  const max = Math.max(step, Math.ceil(highest / step) * step);
  return { max, step };
}

function Kpi({ label, value, detail, tone = '' }) {
  return (
    <div className={`ss-eda-kpi ${tone ? `is-${tone}` : ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function Panel({ title, subtitle, insight, data, layout = {}, className = '', height = 300 }) {
  return (
    <section className={`ss-eda-panel ${className}`}>
      <header className="ss-eda-panel__head">
        <div>
          <h3>{title}</h3>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {insight ? <span className="ss-eda-panel__insight">{insight}</span> : null}
      </header>
      <Plot
        data={data}
        layout={darkLayout(layout)}
        config={PLOT_CONFIG}
        style={{ width: '100%', height: `${height}px` }}
        useResizeHandler
      />
    </section>
  );
}

function DistributionPanel({ chartId, title, subtitle, items, variant }) {
  const peak = histogramPeak(items);
  const scale = percentScale(items);
  const box = { width: 640, height: 238, left: 42, right: 14, top: 18, bottom: 32 };
  const plotWidth = box.width - box.left - box.right;
  const plotHeight = box.height - box.top - box.bottom;
  const baseY = box.top + plotHeight;
  const slot = items.length ? plotWidth / items.length : plotWidth;
  const barWidth = Math.max(7, slot * 0.62);
  const point = (item, index) => ({
    x: box.left + (index * slot) + (slot / 2),
    y: baseY - ((Number(item.percent) || 0) / scale.max) * plotHeight,
  });
  const linePoints = items.map((item, index) => {
    const p = point(item, index);
    return `${p.x},${p.y}`;
  }).join(' ');
  const ticks = [];
  for (let value = 0; value <= scale.max; value += scale.step) ticks.push(value);
  const zeroIndex = items.findIndex((item) => Number(item.bin) === 0);
  const highIndex = items.findIndex((item) => Number(item.bin) >= 7);

  return (
    <section className="ss-eda-panel ss-distribution-panel">
      <header className="ss-distribution-panel__head">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        {peak ? (
          <div className="ss-distribution-panel__peak">
            <span>Peak score</span>
            <strong>{peak.bin}</strong>
            <small>{peak.percent.toFixed(1)}%</small>
          </div>
        ) : null}
      </header>

      <svg
        className="ss-distribution-chart"
        viewBox={`0 0 ${box.width} ${box.height}`}
        role="img"
        aria-label={`${title}. ${subtitle}`}
      >
        <defs>
          <linearGradient id={`${chartId}-negative`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#fb7185" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
          <linearGradient id={`${chartId}-positive`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6ee7b7" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
          <linearGradient id={`${chartId}-neutral`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#cbd5e1" />
            <stop offset="100%" stopColor="#64748b" />
          </linearGradient>
          <linearGradient id={`${chartId}-blue`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#93c5fd" />
            <stop offset="100%" stopColor="#3b82f6" />
          </linearGradient>
        </defs>

        {variant === 'sentiment' && zeroIndex >= 0 ? (
          <g className="ss-distribution-zones" aria-hidden="true">
            <rect x={box.left} y={box.top} width={zeroIndex * slot} height={plotHeight} className="is-negative" />
            <rect x={box.left + zeroIndex * slot} y={box.top} width={slot} height={plotHeight} className="is-neutral" />
            <rect x={box.left + (zeroIndex + 1) * slot} y={box.top}
              width={(items.length - zeroIndex - 1) * slot} height={plotHeight} className="is-positive" />
          </g>
        ) : null}
        {variant === 'relevance' && highIndex >= 0 ? (
          <rect x={box.left + highIndex * slot} y={box.top}
            width={(items.length - highIndex) * slot} height={plotHeight}
            className="ss-distribution-zone-high" aria-hidden="true" />
        ) : null}

        <g className="ss-distribution-grid" aria-hidden="true">
          {ticks.map((value) => {
            const y = baseY - (value / scale.max) * plotHeight;
            return (
              <g key={value}>
                <line x1={box.left} x2={box.width - box.right} y1={y} y2={y} />
                <text x={box.left - 8} y={y + 3} textAnchor="end">{value}%</text>
              </g>
            );
          })}
        </g>

        <polyline className="ss-distribution-line" points={linePoints} aria-hidden="true" />

        <g className="ss-distribution-bars">
          {items.map((item, index) => {
            const p = point(item, index);
            const height = Math.max(0, baseY - p.y);
            const bin = Number(item.bin);
            const isPeak = peak === item;
            const fill = variant === 'relevance'
              ? `url(#${chartId}-blue)`
              : bin < 0
                ? `url(#${chartId}-negative)`
                : bin > 0
                  ? `url(#${chartId}-positive)`
                  : `url(#${chartId}-neutral)`;
            const opacity = variant === 'relevance' ? 0.48 + (Math.max(0, bin) / 10) * 0.52 : 1;
            const showLabel = variant === 'relevance' || index % 2 === 0;
            return (
              <g key={item.bin} className={isPeak ? 'is-peak' : ''}>
                <rect
                  x={p.x - barWidth / 2}
                  y={p.y}
                  width={barWidth}
                  height={height}
                  rx={Math.min(6, barWidth / 3)}
                  fill={fill}
                  opacity={opacity}
                >
                  <title>Score {item.bin}: {item.percent.toFixed(1)}% ({item.count.toLocaleString()} headlines)</title>
                </rect>
                {isPeak ? <circle cx={p.x} cy={Math.max(box.top + 4, p.y - 7)} r="3" /> : null}
                {showLabel ? <text className="ss-distribution-x-label" x={p.x} y={baseY + 18} textAnchor="middle">{item.bin}</text> : null}
              </g>
            );
          })}
        </g>
      </svg>

      {variant === 'sentiment' ? (
        <div className="ss-distribution-legend" aria-label="Sentiment groups">
          <span><i className="is-negative" />Negative</span>
          <span><i className="is-neutral" />Neutral</span>
          <span><i className="is-positive" />Positive</span>
        </div>
      ) : (
        <div className="ss-distribution-legend ss-distribution-legend--scale" aria-label="Relevance score intensity">
          <span>Lower relevance</span><i /><span>Higher relevance</span>
        </div>
      )}
    </section>
  );
}

/**
 * Prepares a correlation matrix for display: blanks the diagonal and the upper
 * triangle, and derives a colour range from the data.
 *
 * A correlation matrix is symmetric with a constant 1.0 diagonal, so of 36
 * cells only 15 carry information — the rest are duplicates, and the diagonal
 * takes the strongest colour in the scale while saying nothing. Fixing the
 * scale at ±1 compounds it: real inter-category correlations here span roughly
 * ±0.25, so every meaningful cell lands in the washed-out middle of the ramp
 * and the whole panel reads as one flat colour.
 *
 * @param {{labels: string[], matrix: number[][]}} corr Raw correlation payload.
 * @returns {{z: (number|null)[][], labels: string[], bound: number,
 *   strongest: {a: string, b: string, r: number}|null}} Masked matrix, a
 *   symmetric colour bound fitted to the data, and the strongest pair.
 */
function prepareCorrelation(corr) {
  const labels = corr?.labels || [];
  const matrix = corr?.matrix || [];
  let peak = 0;
  let strongest = null;

  const z = matrix.map((row, i) => row.map((value, j) => {
    if (j >= i) return null;               // drop diagonal + mirrored half
    const r = Number(value);
    if (!Number.isFinite(r)) return null;
    if (Math.abs(r) > peak) {
      peak = Math.abs(r);
      strongest = { a: labels[i], b: labels[j], r };
    }
    return r;
  }));

  // Round the bound up to a clean step so the colourbar ticks read nicely,
  // with a floor so a near-zero matrix doesn't amplify noise into strong colour.
  const bound = Math.max(0.1, Math.ceil(peak * 20) / 20);
  return { z, labels, bound, strongest };
}

/**
 * Places the failure rate on a logarithmic quality scale.
 *
 * A linear bar is useless here: healthy is ~6 failures per 10,000 and broken is
 * hundreds, so on a 0–10,000 axis every realistic value sits within a pixel of
 * zero. Quality metrics that live near 100% only separate on a log axis, where
 * each decade gets equal room and a tenfold regression is a visible move rather
 * than an invisible one.
 *
 * @param {object} props Component props.
 * @param {number} props.value Failures per 10,000.
 * @returns {JSX.Element} The scale.
 */
function FailureScale({ value }) {
  const MIN = 0.1;
  const MAX = 1000;
  const span = Math.log10(MAX) - Math.log10(MIN);
  const pos = (v) => {
    const clamped = Math.min(MAX, Math.max(MIN, v || MIN));
    return ((Math.log10(clamped) - Math.log10(MIN)) / span) * 100;
  };
  const marker = pos(value);
  const goodEnd = pos(10);
  const warnEnd = pos(50);

  return (
    <div className="ss-failscale">
      <div className="ss-failscale__track" role="img"
           aria-label={`${value.toFixed(1)} failures per 10,000 — healthy is under 10`}>
        <span className="ss-failscale__zone is-good" style={{ left: 0, width: `${goodEnd}%` }} />
        <span className="ss-failscale__zone is-warn"
              style={{ left: `${goodEnd}%`, width: `${warnEnd - goodEnd}%` }} />
        <span className="ss-failscale__zone is-bad"
              style={{ left: `${warnEnd}%`, width: `${100 - warnEnd}%` }} />
        <span className="ss-failscale__marker" style={{ left: `${marker}%` }} />
      </div>
      <div className="ss-failscale__ticks" aria-hidden="true">
        {[0.1, 1, 10, 100, 1000].map((t) => (
          <span key={t} style={{ left: `${pos(t)}%` }}>{t}</span>
        ))}
      </div>
      <div className="ss-failscale__legend">
        <span><i className="is-good" />Healthy &lt;10</span>
        <span><i className="is-warn" />Watch 10–50</span>
        <span><i className="is-bad" />Degraded &gt;50</span>
      </div>
    </div>
  );
}

function ValidationSummary({ validation }) {
  const passed = Number(validation.passed) || 0;
  const failed = Number(validation.failed) || 0;
  const total = passed + failed;
  const rate = Number(validation.rate) || (total ? passed / total : 0);
  // Lead with the failure rate per 10k rather than a pass percentage. At this
  // scale the pass rate is 99.9% and a progress bar of it is visually
  // indistinguishable from 100% — it has no resolution where the data lives,
  // whereas "N per 10,000" moves visibly as quality changes.
  const failPer10k = total ? (failed / total) * 10000 : 0;
  const failedRate = total ? (failed / total) * 100 : 0;
  const tone = failPer10k <= 10 ? 'is-good' : failPer10k <= 50 ? 'is-warn' : 'is-bad';

  return (
    <section className="ss-eda-panel ss-validation-card">
      <header className="ss-eda-panel__head">
        <div>
          <h3>Validation quality</h3>
          <p>Headlines the scoring model failed to return a usable vector for</p>
        </div>
        <span className={`ss-validation-card__badge ${tone}`}>
          {rate >= 0.999 ? 'Healthy' : rate >= 0.99 ? 'Watch' : 'Degraded'}
        </span>
      </header>

      <div className="ss-validation-card__score">
        <strong>{failPer10k.toFixed(1)}</strong>
        <span>failures per 10,000 scored</span>
      </div>

      <FailureScale value={failPer10k} />

      <dl className="ss-validation-card__counts">
        <div>
          <dt>Failed</dt>
          <dd>{formatNumber(failed)} <small>({failedRate.toFixed(3)}%)</small></dd>
        </div>
        <div>
          <dt>Passed</dt>
          <dd>{formatNumber(passed)}</dd>
        </div>
        <div>
          <dt>Pass rate</dt>
          <dd>{(rate * 100).toFixed(2)}%</dd>
        </div>
      </dl>
    </section>
  );
}

/**
 * EDA section: readable summary metrics plus responsive trend, distribution,
 * correlation and validation panels. Time-series can be aggregated in the
 * browser without another API request.
 */
export default function EdaPanels() {
  const [eda, setEda] = useState(null);
  const [open, setOpen] = useState(true);
  const [ready, setReady] = useState(false);
  const [granularity, setGranularity] = useState('month');

  useEffect(() => {
    getJson('/api/eda')
      .then(setEda)
      .catch(() => setEda(null))
      .finally(() => setReady(true));
  }, []);

  const volume = eda?.volume || [];
  const sentTs = eda?.sentiment_ts || [];
  const sentHist = eda?.sentiment_hist || [];
  const relHist = eda?.relevance_hist || [];
  const corr = eda?.category_corr || { labels: [], matrix: [] };
  const val = eda?.validation || { passed: 0, failed: 0, rate: 0 };
  const hasData = ready && (volume.length || sentTs.length || sentHist.length);

  const volumeSeries = useMemo(
    () => aggregateSeries(volume, 'count', granularity, 'sum'),
    [volume, granularity],
  );
  const sentimentSeries = useMemo(
    () => aggregateSeries(sentTs, 'mean_sentiment', granularity, 'mean'),
    [sentTs, granularity],
  );
  const trendWindow = granularity === 'day' ? 30 : granularity === 'week' ? 8 : 6;
  const volumeTrend = useMemo(
    () => rollingAverage(volumeSeries, trendWindow),
    [volumeSeries, trendWindow],
  );
  const sentimentTrend = useMemo(
    () => rollingAverage(sentimentSeries, trendWindow),
    [sentimentSeries, trendWindow],
  );
  const sentimentDistribution = useMemo(() => percentHistogram(sentHist), [sentHist]);
  const relevanceDistribution = useMemo(() => percentHistogram(relHist), [relHist]);
  const correlation = useMemo(() => prepareCorrelation(corr), [corr]);

  const totalHeadlines = volume.reduce((sum, item) => sum + (Number(item.count) || 0), 0);
  const latestVolume = Number(volume.at(-1)?.count) || 0;
  const meanSentiment = sentTs.length
    ? sentTs.reduce((sum, item) => sum + (Number(item.mean_sentiment) || 0), 0) / sentTs.length
    : 0;
  const firstDate = volume[0]?.date || sentTs[0]?.date;
  const lastDate = volume.at(-1)?.date || sentTs.at(-1)?.date;
  const rangeLabel = firstDate && lastDate ? `${firstDate} – ${lastDate}` : 'No date range';
  const timeUnit = granularity === 'day' ? 'day' : granularity === 'week' ? 'week' : 'month';
  const trendLabel = `${trendWindow}-${timeUnit} trend`;
  const partialPeriod = isPartialPeriod(lastDate, granularity);
  const completeVolumeSeries = partialPeriod ? volumeSeries.slice(0, -1) : volumeSeries;
  const completeSentimentSeries = partialPeriod ? sentimentSeries.slice(0, -1) : sentimentSeries;
  const volumeAverage = seriesAverage(completeVolumeSeries);
  const latestVolumePoint = volumeSeries.at(-1);
  const latestSentiment = sentimentSeries.at(-1)?.value;
  const sentimentAverage = seriesAverage(completeSentimentSeries);
  const latestSentimentPoint = sentimentSeries.at(-1);
  const sentimentValues = sentimentSeries.map((item) => item.value);
  const observedSentimentMin = sentimentValues.length ? Math.min(0, ...sentimentValues) : -1;
  const observedSentimentMax = sentimentValues.length ? Math.max(0, ...sentimentValues) : 1;
  const sentimentPadding = Math.max(0.5, (observedSentimentMax - observedSentimentMin) * 0.14);
  const sentimentRange = [
    Math.max(-10, Math.floor(observedSentimentMin - sentimentPadding)),
    Math.min(10, Math.ceil(observedSentimentMax + sentimentPadding)),
  ];

  return (
    <div className="ss-card ss-eda-card">
      <button className="ss-collapse-head ss-collapse-button" type="button" onClick={() => setOpen((v) => !v)}>
        <span>Exploratory data analysis</span>
        <span className="ss-tag">{open ? 'hide' : 'show'}</span>
      </button>
      {!open ? null : !hasData ? (
        <>
          <p className="ss-muted">{ready ? 'No EDA data available.' : 'Loading EDA…'}</p>
          {ready && eda?.error ? (
            <p className="ss-error-text">EDA unavailable: {eda.error}</p>
          ) : null}
        </>
      ) : (
        <>
          <div className="ss-eda-kpis">
            <Kpi label="Total headlines" value={formatNumber(totalHeadlines)} detail={`${volume.length} observed days`} />
            <Kpi label="Latest daily volume" value={formatNumber(latestVolume)} detail={lastDate || 'Latest observation'} />
            <Kpi
              label="Mean sentiment"
              value={meanSentiment > 0 ? `+${meanSentiment.toFixed(2)}` : meanSentiment.toFixed(2)}
              detail="Daily average · −10 to +10"
              tone={meanSentiment > 0 ? 'positive' : meanSentiment < 0 ? 'negative' : ''}
            />
            <Kpi label="Validation pass" value={`${(Number(val.rate) * 100).toFixed(1)}%`} detail={`${formatNumber(val.failed)} failed`} tone="positive" />
            <Kpi label="Coverage" value={formatNumber(volume.length)} detail={rangeLabel} />
          </div>

          <section className="ss-trend-section">
            <div className="ss-eda-toolbar">
              <div>
                <span className="ss-eda-toolbar__eyebrow">Historical trends</span>
                <strong>Trend resolution</strong>
                <span>Controls only the two time-series charts below.</span>
              </div>
              <div className="ss-segmented" role="group" aria-label="Trend resolution">
                {['day', 'week', 'month'].map((unit) => (
                  <button
                    key={unit}
                    type="button"
                    className={granularity === unit ? 'is-active' : ''}
                    onClick={() => setGranularity(unit)}
                  >
                    {unit[0].toUpperCase() + unit.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="ss-eda-grid ss-eda-grid--trends">
              <Panel
                title={`Headline volume / ${timeUnit}`}
                subtitle="Observed volume with a smoothed rolling trend"
                insight={partialPeriod
                  ? `Latest ${formatNumber(latestVolumePoint?.value)} · Partial ${timeUnit}`
                  : comparisonLabel(latestVolumePoint?.value, volumeAverage, formatNumber)}
                height={285}
                data={[
                  {
                    name: 'Observed',
                    type: 'scatter',
                    mode: 'lines',
                    x: volumeSeries.map((d) => d.date),
                    y: volumeSeries.map((d) => d.value),
                    line: { color: 'rgba(96,165,250,0.44)', width: 1.25 },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(59,130,246,0.10)',
                    hovertemplate: '%{y:,.0f} headlines<extra>Observed</extra>',
                  },
                  {
                    name: trendLabel,
                    type: 'scatter',
                    mode: 'lines',
                    x: volumeTrend.map((d) => d.date),
                    y: volumeTrend.map((d) => d.value),
                    line: { color: ACCENT, width: 2.6 },
                    hovertemplate: '%{y:,.0f} headlines<extra>Trend</extra>',
                  },
                  {
                    name: 'Latest',
                    type: 'scatter',
                    mode: 'markers',
                    x: latestVolumePoint ? [latestVolumePoint.date] : [],
                    y: latestVolumePoint ? [latestVolumePoint.value] : [],
                    marker: { color: ACCENT, size: 9, line: { color: '#dbeafe', width: 2 } },
                    hovertemplate: '%{y:,.0f} headlines<extra>Latest</extra>',
                    showlegend: false,
                  },
                ]}
                layout={{
                  hovermode: 'x unified',
                  showlegend: true,
                  legend: { orientation: 'h', x: 1, xanchor: 'right', y: 1.13, yanchor: 'bottom', font: { size: 9 } },
                  margin: { l: 48, r: 12, t: 38, b: 38 },
                  shapes: [
                    { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: volumeAverage, y1: volumeAverage, line: { color: 'rgba(148,163,184,0.48)', width: 1, dash: 'dot' }, layer: 'below' },
                  ],
                  annotations: [
                    { xref: 'paper', x: 1, xanchor: 'right', y: volumeAverage, yanchor: 'bottom', text: `Period avg ${formatNumber(volumeAverage)}`, showarrow: false, font: { size: 9, color: '#94a3b8' }, bgcolor: 'rgba(15,23,42,0.72)', borderpad: 2 },
                  ],
                  xaxis: { ...AXIS, type: 'date', showgrid: false, zeroline: false },
                  yaxis: { ...AXIS, type: 'linear', rangemode: 'tozero', nticks: 5, tickformat: '~s', zeroline: false },
                }}
              />
              <Panel
                title={`Mean sentiment / ${timeUnit}`}
                subtitle="Observed sentiment with a smoothed rolling trend"
                insight={partialPeriod
                  ? `Latest ${latestSentiment?.toFixed(2)} · Partial ${timeUnit}`
                  : comparisonLabel(latestSentiment, sentimentAverage, (value) => value.toFixed(2), 'difference')}
                height={285}
                data={[
                  {
                    name: 'Observed',
                    type: 'scatter',
                    mode: 'lines',
                    x: sentimentSeries.map((d) => d.date),
                    y: sentimentSeries.map((d) => d.value),
                    line: { color: 'rgba(251,191,36,0.44)', width: 1.25 },
                    hovertemplate: '%{y:.2f}<extra>Observed</extra>',
                  },
                  {
                    name: trendLabel,
                    type: 'scatter',
                    mode: 'lines',
                    x: sentimentTrend.map((d) => d.date),
                    y: sentimentTrend.map((d) => d.value),
                    line: { color: '#f59e0b', width: 2.6 },
                    hovertemplate: '%{y:.2f}<extra>Trend</extra>',
                  },
                  {
                    name: 'Latest',
                    type: 'scatter',
                    mode: 'markers',
                    x: latestSentimentPoint ? [latestSentimentPoint.date] : [],
                    y: latestSentimentPoint ? [latestSentimentPoint.value] : [],
                    marker: { color: '#f59e0b', size: 9, line: { color: '#fef3c7', width: 2 } },
                    hovertemplate: '%{y:.2f}<extra>Latest</extra>',
                    showlegend: false,
                  },
                ]}
                layout={{
                  hovermode: 'x unified',
                  showlegend: true,
                  legend: { orientation: 'h', x: 1, xanchor: 'right', y: 1.13, yanchor: 'bottom', font: { size: 9 } },
                  margin: { l: 46, r: 12, t: 38, b: 38 },
                  shapes: [
                    { type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: sentimentRange[0], y1: 0, fillcolor: 'rgba(248,113,113,0.045)', line: { width: 0 }, layer: 'below' },
                    { type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: sentimentRange[1], fillcolor: 'rgba(52,211,153,0.04)', line: { width: 0 }, layer: 'below' },
                    { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: sentimentAverage, y1: sentimentAverage, line: { color: 'rgba(148,163,184,0.48)', width: 1, dash: 'dot' }, layer: 'below' },
                  ],
                  annotations: [
                    { xref: 'paper', x: 1, xanchor: 'right', y: sentimentAverage, yanchor: 'bottom', text: `Period avg ${sentimentAverage.toFixed(2)}`, showarrow: false, font: { size: 9, color: '#94a3b8' }, bgcolor: 'rgba(15,23,42,0.72)', borderpad: 2 },
                  ],
                  xaxis: { ...AXIS, type: 'date', showgrid: false, zeroline: false },
                  yaxis: { ...AXIS, type: 'linear', range: sentimentRange, nticks: 5, zeroline: true, zerolinewidth: 1.5 },
                }}
              />
            </div>
          </section>

          <div className="ss-eda-subsection-head">
            <div>
              <strong>Distribution and quality</strong>
              <span>Dataset composition, relationships and validation.</span>
            </div>
          </div>

          <div className="ss-eda-grid ss-eda-grid--analysis">
            <DistributionPanel
              chartId="sentiment-distribution"
              title="Sentiment distribution"
              subtitle="Percentage of headlines in each score bin"
              items={sentimentDistribution}
              variant="sentiment"
            />
            <DistributionPanel
              chartId="relevance-distribution"
              title="Highest category relevance"
              subtitle="Distribution of each headline's strongest category score"
              items={relevanceDistribution}
              variant="relevance"
            />
            <Panel
              title="Category correlation"
              subtitle="Pairwise relationship between category relevance scores"
              insight={correlation.strongest
                ? `Strongest: ${correlation.strongest.a} × ${correlation.strongest.b} (${correlation.strongest.r.toFixed(2)})`
                : null}
              height={320}
              data={[{
                type: 'heatmap',
                z: correlation.z,
                x: correlation.labels,
                y: correlation.labels,
                // Scale fitted to the data instead of a fixed ±1: the real
                // values span a narrow band, so ±1 would render every cell as
                // the same pale midtone.
                zmin: -correlation.bound,
                zmax: correlation.bound,
                zmid: 0,
                colorscale: 'RdBu',
                reversescale: true,
                xgap: 2,
                ygap: 2,
                hoverongaps: false,
                text: correlation.z.map((row) => row.map((v) => (v === null ? '' : v.toFixed(2)))),
                texttemplate: '%{text}',
                textfont: { size: 11 },
                hovertemplate: '%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar: {
                  title: { text: 'r', side: 'top' },
                  thickness: 10,
                  len: 0.85,
                  tickvals: [-correlation.bound, 0, correlation.bound],
                  ticks: 'outside',
                  ticklen: 3,
                },
              }]}
              layout={{
                margin: { l: 88, r: 18, t: 22, b: 72 },
                xaxis: { ...AXIS, type: 'category', tickangle: -35 },
                yaxis: { ...AXIS, type: 'category' },
              }}
            />
            <ValidationSummary validation={val} />
          </div>
        </>
      )}
    </div>
  );
}
