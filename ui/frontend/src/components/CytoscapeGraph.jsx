import React, { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import {
  toCytoscape,
  cytoscapeStylesheet,
  coseLayout,
} from '../lib/cytoscapeConfig.js';

/**
 * Interactive cytoscape graph. Initializes on mount, re-renders elements
 * when `graph` changes, reports node taps via `onNodeTap`, and destroys the
 * instance on unmount. Guards against empty/missing graphs with a message.
 *
 * @param {object} props Component props.
 * @param {object|null} props.graph API graph payload, or null when absent.
 * @param {(node: object|null) => void} props.onNodeTap Called with the
 *   tapped node's data (id/label/type/attrs), or null when background tapped.
 * @param {(maps: object) => void} [props.onLegend] Receives node color map
 *   for legend rendering whenever the graph changes.
 * @returns {JSX.Element} The graph container.
 */
export default function CytoscapeGraph({ graph, onNodeTap, onLegend }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  const isEmpty =
    !graph || !Array.isArray(graph.nodes) || graph.nodes.length === 0;

  useEffect(() => {
    if (isEmpty || !containerRef.current) {
      return undefined;
    }

    const { elements, nodeColors } = toCytoscape(graph);
    if (onLegend) {
      onLegend(nodeColors);
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: cytoscapeStylesheet(),
      layout: coseLayout(),
    });

    cy.on('tap', 'node', (evt) => {
      const d = evt.target.data();
      onNodeTap({ id: d.id, label: d.label, type: d.type, attrs: d.attrs });
    });
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        onNodeTap(null);
      }
    });

    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph, isEmpty, onNodeTap, onLegend]);

  return (
    <div className="ss-graph" ref={containerRef}>
      {isEmpty ? (
        <div className="ss-graph-empty">
          No simulation for this date / mode.
        </div>
      ) : null}
    </div>
  );
}
