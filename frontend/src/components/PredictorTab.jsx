import React, { useState, useEffect } from 'react';
import { Play, Sparkles, AlertTriangle, ShieldCheck, HelpCircle, Flame, Dna } from 'lucide-react';
import MoleculeViewer from './MoleculeViewer';

export default function PredictorTab() {
  const [smiles, setSmiles] = useState('c1ccccc1[N+](=O)[O-]'); // Nitrobenzene default
  const [selectedModel, setSelectedModel] = useState('GIN');
  const [presets, setPresets] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load presets
  useEffect(() => {
    async function fetchPresets() {
      try {
        const res = await fetch('/api/presets');
        if (res.ok) {
          const data = await res.json();
          setPresets(data);
        }
      } catch (err) {
        console.error('Failed to load presets', err);
      }
    }
    fetchPresets();
  }, []);

  // Run initial prediction on mount
  useEffect(() => {
    handlePredict('c1ccccc1[N+](=O)[O-]', 'GIN');
  }, []);

  const handlePredict = async (inputSmiles = smiles, modelName = selectedModel) => {
    if (!inputSmiles || !inputSmiles.trim()) {
      setError('Please provide a valid SMILES string.');
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          smiles: inputSmiles.trim(),
          model_name: modelName
        })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Prediction failed');
      }

      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectPreset = (p) => {
    setSmiles(p.smiles);
    handlePredict(p.smiles, selectedModel);
  };

  return (
    <div className="space-y-6">
      {/* Molecule Input & Preset Selector */}
      <div className="rounded-3xl bg-slate-900/60 border border-slate-800/80 p-6 backdrop-blur-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-indigo-400" /> Interactive Molecule Inference & Toxicophore Lab
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Input any SMILES chemical notation or select benchmark presets to test GNN mutagenicity prediction.
            </p>
          </div>

          {/* Model Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-medium">Model:</span>
            <select
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                if (smiles) handlePredict(smiles, e.target.value);
              }}
              className="bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-500 font-semibold cursor-pointer"
            >
              <option value="GIN">GIN (Graph Isomorphism - Best 79.3%)</option>
              <option value="GCN">GCN (Graph Convolutional Net)</option>
              <option value="GRAPHSAGE">GraphSAGE (Mean Aggregation)</option>
              <option value="GAT">GAT (Graph Attention Net)</option>
            </select>
          </div>
        </div>

        {/* SMILES Input Bar */}
        <div className="flex flex-col sm:flex-row gap-2">
          <div className="relative flex-1">
            <input
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="e.g. c1ccccc1[N+](=O)[O-] (Nitrobenzene)"
              className="w-full px-4 py-3 rounded-xl bg-slate-950/80 border border-slate-700/80 text-white font-mono text-xs sm:text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
            />
          </div>
          <button
            onClick={() => handlePredict()}
            disabled={loading}
            className="px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-xs sm:text-sm transition-all shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 cursor-pointer"
          >
            {loading ? (
              <span>Inferring...</span>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" /> Run Prediction
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Preset Chips */}
        <div className="space-y-1.5 pt-2">
          <span className="text-xs font-semibold text-slate-400">Standard Chemical Presets:</span>
          <div className="flex flex-wrap gap-2">
            {presets.map((p) => (
              <button
                key={p.id}
                onClick={() => handleSelectPreset(p)}
                className={`px-3 py-1.5 rounded-xl text-xs font-medium border transition-all cursor-pointer flex items-center gap-1.5 ${
                  smiles === p.smiles
                    ? 'bg-indigo-950/60 border-indigo-500 text-indigo-300 shadow-sm'
                    : 'bg-slate-950/40 hover:bg-slate-950 border-slate-800 text-slate-300'
                }`}
              >
                <span>{p.name}</span>
                <span className="text-[10px] text-slate-500">({p.category})</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Prediction Output & Visualization */}
      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Decision Cards & Probabilities */}
          <div className="lg:col-span-5 space-y-4">
            {/* Main Decision Banner */}
            <div
              className={`p-6 rounded-3xl border backdrop-blur-sm flex flex-col justify-between ${
                prediction.is_mutagenic
                  ? 'bg-rose-950/20 border-rose-500/30 shadow-lg shadow-rose-950/30'
                  : 'bg-emerald-950/20 border-emerald-500/30 shadow-lg shadow-emerald-950/30'
              }`}
            >
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Predicted Classification
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-900/60 border border-slate-800 text-slate-300">
                  Model: {prediction.model_used}
                </span>
              </div>

              <div className="flex items-center gap-3">
                {prediction.is_mutagenic ? (
                  <div className="p-3 rounded-2xl bg-rose-500/20 border border-rose-500/40 text-rose-400">
                    <Flame className="w-8 h-8" />
                  </div>
                ) : (
                  <div className="p-3 rounded-2xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-400">
                    <ShieldCheck className="w-8 h-8" />
                  </div>
                )}
                <div>
                  <h3
                    className={`text-2xl font-black ${
                      prediction.is_mutagenic ? 'text-rose-400' : 'text-emerald-400'
                    }`}
                  >
                    {prediction.predicted_label}
                  </h3>
                  <p className="text-xs text-slate-300">
                    Confidence: <span className="font-bold">{prediction.confidence_pct}%</span>
                  </p>
                </div>
              </div>

              {/* Confidence Progress Bars */}
              <div className="mt-6 space-y-3 pt-4 border-t border-slate-800/60">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300 font-medium">P(Mutagenic | G):</span>
                    <span className="font-mono text-rose-400 font-bold">
                      {(prediction.p_mutagenic * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="bg-rose-500 h-full rounded-full transition-all duration-500"
                      style={{ width: `${prediction.p_mutagenic * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300 font-medium">P(Non-Mutagenic | G):</span>
                    <span className="font-mono text-emerald-400 font-bold">
                      {(prediction.p_non_mutagenic * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                      style={{ width: `${prediction.p_non_mutagenic * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Toxicophore Explanation & Atom Saliency */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-sm space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Dna className="w-4 h-4 text-indigo-400" /> Explainability & Key Atom Attribution
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Gradient saliency highlights atoms with highest contribution to the model's decision:
              </p>

              {/* Atom Ranking list */}
              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                {prediction.graph.nodes
                  .slice()
                  .sort((a, b) => (b.importance || 0) - (a.importance || 0))
                  .map((node, rank) => (
                    <div
                      key={node.id}
                      className="p-2 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-center justify-between text-xs"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-slate-500 font-mono w-4">#{rank + 1}</span>
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: node.color }}
                        />
                        <span className="font-medium text-slate-200">
                          {node.name} ({node.symbol}{node.id + 1})
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${
                              (node.importance || 0) > 0.6
                                ? 'bg-rose-500'
                                : (node.importance || 0) > 0.3
                                ? 'bg-amber-500'
                                : 'bg-slate-600'
                            }`}
                            style={{ width: `${(node.importance || 0) * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-slate-300 w-10 text-right">
                          {((node.importance || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          {/* Right: Molecule 2D Topology with Heatmap Glow */}
          <div className="lg:col-span-7">
            <MoleculeViewer
              graph={prediction.graph}
              title={`2D Graph Structure: ${prediction.graph.name}`}
              subtitle={`${prediction.graph.num_nodes} Atoms, ${prediction.graph.num_edges} Bonds. Glowing halos indicate toxicophore saliency.`}
              showImportance={true}
            />
          </div>
        </div>
      )}
    </div>
  );
}
