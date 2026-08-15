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
 * Chart colours for the active theme, read from the DOM at call time.
 *
 * Anything a chart draws that is NOT a data colour belongs here. Values hardcoded
 * for the dark card — white gridlines, near-white labels, a navy annotation pill —
 * survive the theme switch and turn invisible (or into a dark blob) on the light
 * card, which is exactly what went wrong.
 *
 * @returns {object} `{light, text, muted, grid, zeroline, annotationBg, halo}`.
 */
export function chartTheme() {
  const light = typeof document !== 'undefined'
    && document.documentElement.dataset.theme === 'light';
  return {
    light,
    text: light ? '#334155' : '#c9d1d9',
    muted: light ? '#475569' : '#94a3b8',
    grid: light ? 'rgba(15,23,42,0.10)' : 'rgba(255,255,255,0.08)',
    zeroline: light ? 'rgba(15,23,42,0.18)' : 'rgba(255,255,255,0.12)',
    // A pill drawn ON the card, so it has to match the card rather than invert it.
    annotationBg: light ? 'rgba(255,255,255,0.86)' : 'rgba(15,23,42,0.72)',
    // Ring separating a marker from the line beneath it: the card colour, so on
    // dark it reads as a soft glow and on light as a clean cut-out.
    halo: light ? '#ffffff' : null,
  };
}

/**
 * Builds a transparent, theme-aware Plotly layout merged with overrides.
 *
 * Named for the dark default it began as; it follows the active theme.
 *
 * @param {object} [overrides] Layout keys to merge over the defaults.
 * @returns {object} A Plotly layout object.
 */
export function darkLayout(overrides = {}) {
  const t = chartTheme();
  const axis = { gridcolor: t.grid, zerolinecolor: t.zeroline };
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: t.text, size: 12 },
    margin: { l: 48, r: 16, t: 30, b: 40 },
    ...overrides,
    xaxis: { ...axis, ...(overrides.xaxis || {}) },
    yaxis: { ...axis, ...(overrides.yaxis || {}) },
  };
}

export const PLOT_CONFIG = { displayModeBar: false, responsive: true };
