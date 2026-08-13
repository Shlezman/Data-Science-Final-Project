/*
 * Shared Plotly setup. Uses react-plotly.js's factory over the prebuilt
 * plotly.js-dist-min bundle (Vite-friendly — no source build of plotly.js),
 * plus a dark, transparent layout so charts sit cleanly on the app's cards.
 */
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';

export const Plot = createPlotlyComponent(Plotly);

export const UP = '#22c55e';
export const DOWN = '#ef4444';
export const NEUTRAL = '#8b93a1';
export const ACCENT = '#3b82f6';

/**
 * Builds a transparent, dark-friendly Plotly layout merged with overrides.
 *
 * @param {object} [overrides] Layout keys to merge over the defaults.
 * @returns {object} A Plotly layout object.
 */
export function darkLayout(overrides = {}) {
  const light = typeof document !== 'undefined'
    && document.documentElement.dataset.theme === 'light';
  const axis = {
    gridcolor: light ? 'rgba(15,23,42,0.10)' : 'rgba(255,255,255,0.08)',
    zerolinecolor: light ? 'rgba(15,23,42,0.18)' : 'rgba(255,255,255,0.12)',
  };
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: light ? '#334155' : '#c9d1d9', size: 12 },
    margin: { l: 48, r: 16, t: 30, b: 40 },
    ...overrides,
    xaxis: { ...axis, ...(overrides.xaxis || {}) },
    yaxis: { ...axis, ...(overrides.yaxis || {}) },
  };
}

export const PLOT_CONFIG = { displayModeBar: false, responsive: true };
