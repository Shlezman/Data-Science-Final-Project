import React from 'react';

/**
 * Infrastructure tab — the system-design and deployment diagrams (the same SVG
 * figures used in the project book), rendered on light plates so they stay
 * legible in both themes.
 */

const FIGURES = [
  {
    src: 'figures/fig1_pipeline.svg',
    title: 'End-to-end pipeline',
    caption: 'Scrape → LLM scoring → features → model zoo → registry → live dashboard.',
  },
  {
    src: 'figures/fig2_system_architecture.svg',
    title: 'System architecture',
    caption: 'Loosely-coupled modules communicating through the PostgreSQL database.',
  },
  {
    src: 'figures/fig3_deployment_topology.svg',
    title: 'Two-host deployment topology',
    caption: 'GPU compute node and database/UI host, decoupled through the shared database.',
  },
  {
    src: 'figures/fig4_chronological_split.svg',
    title: 'Leakage-safe chronological split',
    caption: '70 / 15 / 15 train / validation / test; all transforms fit on the train slice only.',
  },
  {
    src: 'figures/fig5_registry_lifecycle.svg',
    title: 'Model-registry lifecycle',
    caption: 'Train, register, auto-select with sticky manual override, and serve the champion.',
  },
];

export default function Infrastructure() {
  return (
    <div className="ss-card">
      <div className="ss-dashboard-section-head">
        <div className="ss-dashboard-section-head__copy">
          <span className="ss-dashboard-section-head__eyebrow">Infrastructure</span>
          <h2>System design &amp; deployment</h2>
          <p>Architecture, data flow, and the leakage-safe modeling pipeline.</p>
        </div>
      </div>

      <div className="ss-infra-grid">
        {FIGURES.map((f) => (
          <figure key={f.src} className="ss-infra-figure">
            <h3>{f.title}</h3>
            <div className="ss-infra-figure__plate">
              <img src={f.src} alt={f.title} loading="lazy" />
            </div>
            <figcaption>{f.caption}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
