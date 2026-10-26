import React, { useState, useRef } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Info, Eye } from 'lucide-react';

export default function MoleculeViewer({
  graph,
  showImportance = true,
  title = "",
  subtitle = "",
  className = ""
}) {
  const [scale, setScale] = useState(1);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    return (
      <div className={`flex flex-col items-center justify-center h-80 bg-slate-900/40 rounded-2xl border border-slate-800 text-slate-500 ${className}`}>
        <Info className="w-8 h-8 mb-2 opacity-50" />
        <p className="text-sm font-medium">No molecule graph loaded</p>
      </div>
    );
  }

  const nodes = graph.nodes;
  const edges = graph.edges || [];

  const handleZoomIn = () => setScale((s) => Math.min(s + 0.2, 2.5));
  const handleZoomOut = () => setScale((s) => Math.max(s - 0.2, 0.5));
  const handleReset = () => {
    setScale(1);
    setPan({ x: 0, y: 0 });
  };

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  // Compute importance color
  const getImportanceGlow = (imp) => {
    if (!imp || imp < 0.1) return null;
    if (imp > 0.7) return 'rgba(239, 68, 68, 0.7)'; // strong red
    if (imp > 0.4) return 'rgba(245, 158, 11, 0.6)'; // orange
    return 'rgba(59, 130, 246, 0.5)'; // blue
  };

  return (
    <div className={`relative bg-slate-900/60 backdrop-blur-md rounded-2xl border border-slate-800/80 overflow-hidden flex flex-col ${className}`}>
      {/* Header bar */}
      <div className="px-5 py-3 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/80">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-slate-100 text-sm">{title || graph.name || "Molecule Graph"}</h3>
            {graph.label_text && (
              <span
                className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${
                  graph.label === 1
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                }`}
              >
                {graph.label_text}
              </span>
            )}
          </div>
          {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-1 bg-slate-800/60 p-1 rounded-lg border border-slate-700/60">
          <button
            onClick={handleZoomIn}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-700/60 rounded transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-700/60 rounded transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleReset}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-700/60 rounded transition-colors"
            title="Reset View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* SVG Container */}
      <div
        className="relative flex-1 min-h-[360px] cursor-grab active:cursor-grabbing select-none overflow-hidden bg-radial from-slate-900 via-slate-950 to-slate-950"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg
          viewBox="0 0 480 480"
          className="w-full h-full"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.15s ease-out'
          }}
        >
          <defs>
            {/* Filter for glowing saliency */}
            <filter id="glow-high" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glow-med" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background grid */}
          <g opacity="0.06">
            {Array.from({ length: 13 }).map((_, i) => (
              <line key={`gx-${i}`} x1={i * 40} y1="0" x2={i * 40} y2="480" stroke="#94a3b8" strokeWidth="1" strokeDasharray="2,2" />
            ))}
            {Array.from({ length: 13 }).map((_, i) => (
              <line key={`gy-${i}`} x1="0" y1={i * 40} x2="480" y2={i * 40} stroke="#94a3b8" strokeWidth="1" strokeDasharray="2,2" />
            ))}
          </g>

          {/* Render Edges (Bonds) */}
          <g>
            {edges.map((edge, idx) => {
              const src = nodes.find((n) => n.id === edge.source);
              const tgt = nodes.find((n) => n.id === edge.target);
              if (!src || !tgt) return null;

              const isAromatic = edge.is_aromatic || edge.type === 'Aromatic';
              const isDouble = edge.is_double || edge.type === 'Double';

              if (isDouble) {
                // Parallel double bond
                const dx = tgt.x - src.x;
                const dy = tgt.y - src.y;
                const len = Math.hypot(dx, dy) || 1;
                const offsetX = (-dy / len) * 3;
                const offsetY = (dx / len) * 3;

                return (
                  <g key={`edge-double-${idx}`}>
                    <line
                      x1={src.x + offsetX}
                      y1={src.y + offsetY}
                      x2={tgt.x + offsetX}
                      y2={tgt.y + offsetY}
                      stroke="#64748b"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                    />
                    <line
                      x1={src.x - offsetX}
                      y1={src.y - offsetY}
                      x2={tgt.x - offsetX}
                      y2={tgt.y - offsetY}
                      stroke="#64748b"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                    />
                  </g>
                );
              }

              return (
                <line
                  key={`edge-${idx}`}
                  x1={src.x}
                  y1={src.y}
                  x2={tgt.x}
                  y2={tgt.y}
                  stroke={isAromatic ? '#f59e0b' : '#475569'}
                  strokeWidth={isAromatic ? '3' : '2.5'}
                  strokeDasharray={isAromatic ? '5,3' : undefined}
                  strokeLinecap="round"
                  opacity={isAromatic ? '0.85' : '0.7'}
                />
              );
            })}
          </g>

          {/* Render Nodes (Atoms) */}
          <g>
            {nodes.map((node) => {
              const imp = showImportance ? (node.importance ?? 0) : 0;
              const glowColor = getImportanceGlow(imp);
              const isHovered = hoveredNode && hoveredNode.id === node.id;

              return (
                <g
                  key={`node-${node.id}`}
                  className="cursor-pointer transition-transform"
                  onMouseEnter={() => setHoveredNode(node)}
                  onMouseLeave={() => setHoveredNode(null)}
                >
                  {/* Saliency Halo */}
                  {glowColor && (
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={18 + imp * 14}
                      fill={glowColor}
                      filter={imp > 0.6 ? 'url(#glow-high)' : 'url(#glow-med)'}
                      className={imp > 0.6 ? 'animate-pulse-slow' : ''}
                    />
                  )}

                  {/* Node Outer Circle */}
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={isHovered ? 17 : 14}
                    fill={node.color || '#3b82f6'}
                    stroke={isHovered ? '#ffffff' : '#0f172a'}
                    strokeWidth={isHovered ? '2.5' : '2'}
                    className="transition-all duration-150"
                  />

                  {/* Atom Symbol Text */}
                  <text
                    x={node.x}
                    y={node.y + 4}
                    textAnchor="middle"
                    fill="#ffffff"
                    fontSize="11"
                    fontWeight="bold"
                    pointerEvents="none"
                    style={{ textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}
                  >
                    {node.symbol}
                  </text>

                  {/* Atom Index Subscript */}
                  <text
                    x={node.x + 9}
                    y={node.y - 7}
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize="8"
                    fontWeight="semibold"
                    pointerEvents="none"
                  >
                    {node.id + 1}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Hover Tooltip */}
        {hoveredNode && (
          <div
            className="absolute z-20 pointer-events-none bg-slate-900/95 backdrop-blur-md px-3 py-2 rounded-xl border border-slate-700 shadow-2xl text-xs space-y-1"
            style={{
              left: Math.min(Math.max(hoveredNode.x + pan.x * scale, 20), 320),
              top: Math.max(hoveredNode.y + pan.y * scale - 70, 10)
            }}
          >
            <div className="flex items-center gap-1.5 font-bold text-slate-100">
              <span
                className="w-2.5 h-2.5 rounded-full inline-block"
                style={{ backgroundColor: hoveredNode.color }}
              />
              <span>
                {hoveredNode.name} ({hoveredNode.symbol}) #{hoveredNode.id + 1}
              </span>
            </div>
            <div className="text-slate-400 flex items-center justify-between gap-4">
              <span>Bond Degree:</span>
              <span className="font-mono text-slate-200">{hoveredNode.degree}</span>
            </div>
            {showImportance && (
              <div className="text-slate-400 flex items-center justify-between gap-4">
                <span>GNN Saliency:</span>
                <span
                  className={`font-mono font-bold ${
                    (hoveredNode.importance || 0) > 0.6
                      ? 'text-rose-400'
                      : (hoveredNode.importance || 0) > 0.3
                      ? 'text-amber-400'
                      : 'text-slate-300'
                  }`}
                >
                  {((hoveredNode.importance || 0) * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer Legend */}
      <div className="px-4 py-2.5 bg-slate-950/80 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-2">
        <div className="flex items-center gap-3">
          <span className="font-medium text-slate-300">Atoms:</span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#64748b]"></span> C
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#3b82f6]"></span> N
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#ef4444]"></span> O
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#06b6d4]"></span> Cl
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#f59e0b]"></span> Br
          </span>
        </div>

        {showImportance && (
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Attribution:</span>
            <div className="flex items-center gap-1">
              <span className="w-2.5 h-1.5 rounded-sm bg-slate-700"></span> Low
              <span className="w-2.5 h-1.5 rounded-sm bg-amber-500"></span> Med
              <span className="w-2.5 h-1.5 rounded-sm bg-rose-500"></span> High Saliency
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
