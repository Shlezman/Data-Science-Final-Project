/*
 * Cytoscape styling / data-mapping helpers. Colors are keyed off node and
 * edge `type` from the API graph payload, with a deterministic fallback
 * palette for any unforeseen types so the graph never renders colorless.
 */

const NODE_PALETTE = [
  '#3b82f6',
  '#22c55e',
  '#f59e0b',
  '#a855f7',
  '#ef4444',
  '#06b6d4',
  '#ec4899',
  '#84cc16',
];

const EDGE_PALETTE = [
  '#94a3b8',
  '#60a5fa',
  '#f472b6',
  '#fbbf24',
  '#34d399',
];

/**
 * Assigns a stable color to each distinct value via a small palette.
 *
 * @param {string[]} values Distinct type strings, order defines color.
 * @param {string[]} palette Color list to cycle through.
 * @returns {Record<string, string>} Map from value to hex color.
 */
function buildColorMap(values, palette) {
  const map = {};
  values.forEach((value, index) => {
    map[value] = palette[index % palette.length];
  });
  return map;
}

/**
 * Converts an API graph payload into cytoscape elements plus color maps.
 *
 * @param {object} graph Graph with `nodes` and `edges` arrays.
 * @returns {{elements: object[], nodeColors: object, edgeColors: object}}
 *   Cytoscape elements and the type->color maps for legend + styling.
 */
export function toCytoscape(graph) {
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph?.edges) ? graph.edges : [];

  const nodeTypes = [...new Set(nodes.map((n) => n.type || 'node'))];
  const edgeTypes = [...new Set(edges.map((e) => e.type || 'edge'))];
  const nodeColors = buildColorMap(nodeTypes, NODE_PALETTE);
  const edgeColors = buildColorMap(edgeTypes, EDGE_PALETTE);

  const elements = [
    ...nodes.map((n) => ({
      group: 'nodes',
      data: {
        id: String(n.id),
        label: n.label ?? String(n.id),
        type: n.type || 'node',
        color: nodeColors[n.type || 'node'],
        attrs: n.attrs || {},
      },
    })),
    ...edges.map((e, i) => ({
      group: 'edges',
      data: {
        id: `e${i}-${e.src}-${e.dst}`,
        source: String(e.src),
        target: String(e.dst),
        type: e.type || 'edge',
        weight: typeof e.weight === 'number' ? e.weight : 1,
        color: edgeColors[e.type || 'edge'],
      },
    })),
  ];

  return { elements, nodeColors, edgeColors };
}

/**
 * Returns the cytoscape stylesheet driving node/edge appearance.
 *
 * @returns {object[]} Cytoscape style array.
 */
export function cytoscapeStylesheet() {
  return [
    {
      selector: 'node',
      style: {
        'background-color': 'data(color)',
        label: 'data(label)',
        color: '#e8ebef',
        'font-size': 9,
        'text-outline-width': 2,
        'text-outline-color': '#1b1f24',
        'text-valign': 'center',
        'text-halign': 'center',
        width: 26,
        height: 26,
        'border-width': 1,
        'border-color': 'rgba(255,255,255,0.35)',
      },
    },
    {
      selector: 'node:selected',
      style: {
        'border-width': 3,
        'border-color': '#ffffff',
      },
    },
    {
      selector: 'edge',
      style: {
        width: 'mapData(weight, 0, 1, 1, 4)',
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        opacity: 0.7,
      },
    },
  ];
}

/**
 * The cose force-directed layout config used on every (re)render.
 *
 * @returns {object} Cytoscape layout options.
 */
export function coseLayout() {
  return {
    name: 'cose',
    animate: false,
    nodeRepulsion: 8000,
    idealEdgeLength: 80,
    padding: 24,
    fit: true,
  };
}
