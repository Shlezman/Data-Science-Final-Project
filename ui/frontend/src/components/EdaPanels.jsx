import React, { useEffect, useState } from 'react';
import { getJson } from '../lib/api.js';
import { Plot, darkLayout, PLOT_CONFIG, ACCENT } from '../lib/plotly.js';

const PLOT_STYLE = { width: '100%', height: '260px' };

/**
 * One titled chart panel.
 *
 * @param {object} props Component props.
 * @param {string} props.title Panel heading.
 * @param {Array} props.data Plotly traces.
 * @param {object} [props.layout] Extra Plotly layout keys.
 * @returns {JSX.Element} The panel.
 */
function Panel({ title, data, layout = {} }) {
  return (
    <div className="ss-eda-panel">
      <p className="ss-section-title">{title}</p>
      <Plot data={data} layout={darkLayout(layout)} config={PLOT_CONFIG}
            style={PLOT_STYLE} useResizeHandler />
    </div>
  );
}

/**
 * EDA section: headline volume, sentiment time-series + histogram, relevance
 * histogram, 6×6 category-relevance correlation heatmap, and validation pass
 * rate. Reads /api/eda once; collapsible to keep the dashboard tidy.
 *
 * @returns {JSX.Element} The EDA section.
 */
export default function EdaPanels() {
  const [eda, setEda] = useState(null);
  const [open, setOpen] = useState(true);
  const [ready, setReady] = useState(false);

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

  return (
    <div className="ss-card">
      <h2 className="ss-collapse-head" onClick={() => setOpen((v) => !v)}>
        Exploratory data analysis
        <span className="ss-tag">{open ? 'hide' : 'show'}</span>
      </h2>
      {!open ? null : !hasData ? (
        <p className="ss-muted">{ready ? 'No EDA data available.' : 'Loading EDA…'}</p>
      ) : (
        <div className="ss-eda-grid">
          <Panel title="Headline volume / day"
                 data={[{ type: 'scatter', mode: 'lines', x: volume.map((d) => d.date),
                          y: volume.map((d) => d.count), line: { color: ACCENT } }]} />
          <Panel title="Mean sentiment / day"
                 data={[{ type: 'scatter', mode: 'lines', x: sentTs.map((d) => d.date),
                          y: sentTs.map((d) => d.mean_sentiment), line: { color: '#f59e0b' } }]}
                 layout={{ yaxis: { range: [-10, 10] } }} />
          <Panel title="Sentiment distribution (−10..+10)"
                 data={[{ type: 'bar', x: sentHist.map((d) => d.bin),
                          y: sentHist.map((d) => d.count), marker: { color: '#f59e0b' } }]} />
          <Panel title="Top-category relevance (0..10)"
                 data={[{ type: 'bar', x: relHist.map((d) => d.bin),
                          y: relHist.map((d) => d.count), marker: { color: ACCENT } }]} />
          <Panel title="Category correlation"
                 data={[{ type: 'heatmap', z: corr.matrix, x: corr.labels, y: corr.labels,
                          zmin: -1, zmax: 1, colorscale: 'RdBu', reversescale: true }]}
                 layout={{ margin: { l: 80, r: 16, t: 30, b: 70 } }} />
          <Panel title={`Validation pass rate — ${(val.rate * 100).toFixed(1)}%`}
                 data={[{ type: 'bar', orientation: 'h', x: [val.passed, val.failed],
                          y: ['passed', 'failed'],
                          marker: { color: ['#22c55e', '#ef4444'] } }]} />
        </div>
      )}
    </div>
  );
}
