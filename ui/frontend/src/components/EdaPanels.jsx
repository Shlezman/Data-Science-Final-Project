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

function percentHistogram(histogram) {
  const total = histogram.reduce((sum, item) => sum + (Number(item.count) || 0), 0);
  return histogram.map((item) => ({
    ...item,
    count: Number(item.count) || 0,
    percent: total ? ((Number(item.count) || 0) / total) * 100 : 0,
  }));
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

function Panel({ title, subtitle, data, layout = {}, className = '', height = 300 }) {
  return (
    <section className={`ss-eda-panel ${className}`}>
      <header className="ss-eda-panel__head">
        <div>
          <h3>{title}</h3>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
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

function ValidationSummary({ validation }) {
  const passed = Number(validation.passed) || 0;
  const failed = Number(validation.failed) || 0;
  const total = passed + failed;
  const rate = Number(validation.rate) || (total ? passed / total : 0);
  const failedRate = total ? (failed / total) * 100 : 0;

  return (
    <section className="ss-eda-panel ss-validation-card">
      <header className="ss-eda-panel__head">
        <div>
          <h3>Validation quality</h3>
          <p>Share of processed headlines that passed validation</p>
        </div>
      </header>
      <div className="ss-validation-card__score">
        <strong>{(rate * 100).toFixed(1)}%</strong>
        <span>passed</span>
      </div>
      <div className="ss-validation-card__track" aria-label={`${(rate * 100).toFixed(1)}% passed`}>
        <span style={{ width: `${Math.min(100, Math.max(0, rate * 100))}%` }} />
      </div>
      <dl className="ss-validation-card__counts">
        <div>
          <dt>Passed</dt>
          <dd>{formatNumber(passed)}</dd>
        </div>
        <div>
          <dt>Failed</dt>
          <dd>{formatNumber(failed)} <small>({failedRate.toFixed(2)}%)</small></dd>
        </div>
        <div>
          <dt>Total checked</dt>
          <dd>{formatNumber(total)}</dd>
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
  const sentimentDistribution = useMemo(() => percentHistogram(sentHist), [sentHist]);
  const relevanceDistribution = useMemo(() => percentHistogram(relHist), [relHist]);

  const totalHeadlines = volume.reduce((sum, item) => sum + (Number(item.count) || 0), 0);
  const latestVolume = Number(volume.at(-1)?.count) || 0;
  const meanSentiment = sentTs.length
    ? sentTs.reduce((sum, item) => sum + (Number(item.mean_sentiment) || 0), 0) / sentTs.length
    : 0;
  const firstDate = volume[0]?.date || sentTs[0]?.date;
  const lastDate = volume.at(-1)?.date || sentTs.at(-1)?.date;
  const rangeLabel = firstDate && lastDate ? `${firstDate} – ${lastDate}` : 'No date range';
  const timeUnit = granularity === 'day' ? 'day' : granularity === 'week' ? 'week' : 'month';

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

          <div className="ss-eda-toolbar">
            <div>
              <strong>Trend resolution</strong>
              <span>Aggregate dense daily history for a clearer signal.</span>
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

          <div className="ss-eda-grid">
            <Panel
              title={`Headline volume / ${timeUnit}`}
              subtitle={`${formatNumber(totalHeadlines)} headlines across the selected history`}
              data={[{
                type: 'scatter',
                mode: 'lines',
                x: volumeSeries.map((d) => d.date),
                y: volumeSeries.map((d) => d.value),
                line: { color: ACCENT, width: 2 },
                fill: 'tozeroy',
                fillcolor: 'rgba(59,130,246,0.12)',
                hovertemplate: '%{x}<br>%{y:,.0f} headlines<extra></extra>',
              }]}
              layout={{
                hovermode: 'x unified',
                xaxis: { ...AXIS, type: 'date' },
                yaxis: { ...AXIS, type: 'linear', title: 'Headlines', rangemode: 'tozero' },
              }}
            />
            <Panel
              title={`Mean sentiment / ${timeUnit}`}
              subtitle="Values above zero are positive; values below zero are negative"
              data={[{
                type: 'scatter',
                mode: 'lines',
                x: sentimentSeries.map((d) => d.date),
                y: sentimentSeries.map((d) => d.value),
                line: { color: '#f59e0b', width: 2 },
                hovertemplate: '%{x}<br>Mean sentiment: %{y:.2f}<extra></extra>',
              }]}
              layout={{
                hovermode: 'x unified',
                xaxis: { ...AXIS, type: 'date' },
                yaxis: { ...AXIS, type: 'linear', range: [-10, 10], zeroline: true, zerolinewidth: 2 },
              }}
            />
            <Panel
              title="Sentiment distribution"
              subtitle="Percentage of headlines in each score bin"
              data={[{
                type: 'bar',
                x: sentimentDistribution.map((d) => d.bin),
                y: sentimentDistribution.map((d) => d.percent),
                customdata: sentimentDistribution.map((d) => d.count),
                marker: {
                  color: sentimentDistribution.map((d) => (Number(d.bin) < 0 ? '#ef4444' : Number(d.bin) > 0 ? '#22c55e' : '#8b93a1')),
                },
                hovertemplate: 'Score %{x}<br>%{y:.1f}% · %{customdata:,} headlines<extra></extra>',
              }]}
              layout={{
                bargap: 0.12,
                xaxis: { ...AXIS, type: 'linear', title: 'Sentiment score (−10 to +10)', dtick: 2 },
                yaxis: { ...AXIS, type: 'linear', title: 'Share of headlines', ticksuffix: '%' },
              }}
            />
            <Panel
              title="Highest category relevance"
              subtitle="Distribution of each headline's strongest category score"
              data={[{
                type: 'bar',
                x: relevanceDistribution.map((d) => d.bin),
                y: relevanceDistribution.map((d) => d.percent),
                customdata: relevanceDistribution.map((d) => d.count),
                marker: { color: ACCENT },
                hovertemplate: 'Score %{x}<br>%{y:.1f}% · %{customdata:,} headlines<extra></extra>',
              }]}
              layout={{
                bargap: 0.12,
                xaxis: { ...AXIS, type: 'linear', title: 'Highest relevance score (0 to 10)', dtick: 1 },
                yaxis: { ...AXIS, type: 'linear', title: 'Share of headlines', ticksuffix: '%' },
              }}
            />
            <Panel
              title="Category correlation"
              subtitle="Pairwise relationship between category relevance scores"
              height={320}
              data={[{
                type: 'heatmap',
                z: corr.matrix,
                x: corr.labels,
                y: corr.labels,
                zmin: -1,
                zmax: 1,
                zmid: 0,
                colorscale: 'RdBu',
                reversescale: true,
                text: corr.matrix?.map((row) => row.map((value) => Number(value).toFixed(2))),
                texttemplate: '%{text}',
                textfont: { size: 10 },
                hovertemplate: '%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>',
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
