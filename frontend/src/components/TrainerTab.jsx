import React, { useState } from 'react';
import { Sliders, Play, CheckCircle2, AlertCircle, RefreshCw, Cpu } from 'lucide-react';

export default function TrainerTab() {
  const [modelName, setModelName] = useState('GIN');
  const [hiddenDim, setHiddenDim] = useState(64);
  const [numLayers, setNumLayers] = useState(2);
  const [pooling, setPooling] = useState('sum');
  const [lr, setLr] = useState(0.005);
  const [epochs, setEpochs] = useState(30);
  const [dropout, setDropout] = useState(0.2);

  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleStartTraining = async () => {
    setTraining(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          hidden_dim: Number(hiddenDim),
          num_layers: Number(numLayers),
          pooling: pooling,
          learning_rate: Number(lr),
          epochs: Number(epochs),
          dropout: Number(dropout)
        })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Training failed');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-3xl bg-slate-900/60 border border-slate-800/80 p-6 backdrop-blur-sm">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-indigo-400" /> Interactive GNN Hyperparameter Workbench
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Configure architecture depth, aggregation pooling, and optimization parameters to train and evaluate
              a live model on the MUTAG dataset in real-time.
            </p>
          </div>
          <span className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700 text-xs font-mono">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" /> PyTorch Geometric Engine
          </span>
        </div>
      </div>

      {/* Grid: Hyperparameter Controls on Left, Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Configuration Form */}
        <div className="lg:col-span-5 rounded-3xl bg-slate-900/60 border border-slate-800/80 p-6 backdrop-blur-sm space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">Hyperparameter Setup</h3>

          <div className="space-y-4 text-xs">
            {/* Model Architecture */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Architecture</label>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-xl px-3 py-2 focus:ring-1 focus:ring-indigo-500 font-medium cursor-pointer"
              >
                <option value="GIN">GIN (Graph Isomorphism Network)</option>
                <option value="GCN">GCN (Graph Convolutional Network)</option>
                <option value="GRAPHSAGE">GraphSAGE (Sample & Aggregate)</option>
                <option value="GAT">GAT (Graph Attention Network)</option>
              </select>
            </div>

            {/* Hidden Channels */}
            <div>
              <div className="flex justify-between text-slate-300 font-semibold mb-1">
                <span>Hidden Dimension</span>
                <span className="font-mono text-indigo-400">{hiddenDim} neurons</span>
              </div>
              <input
                type="range"
                min="16"
                max="128"
                step="16"
                value={hiddenDim}
                onChange={(e) => setHiddenDim(e.target.value)}
                className="w-full accent-indigo-500 cursor-pointer"
              />
            </div>

            {/* Number of Layers */}
            <div>
              <div className="flex justify-between text-slate-300 font-semibold mb-1">
                <span>GNN Message Passing Layers</span>
                <span className="font-mono text-indigo-400">{numLayers} layers</span>
              </div>
              <input
                type="range"
                min="1"
                max="4"
                step="1"
                value={numLayers}
                onChange={(e) => setNumLayers(e.target.value)}
                className="w-full accent-indigo-500 cursor-pointer"
              />
            </div>

            {/* Pooling Strategy */}
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Global Readout Pooling</label>
              <div className="grid grid-cols-3 gap-2">
                {['sum', 'mean', 'max'].map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => setPooling(p)}
                    className={`py-2 rounded-xl text-center font-bold capitalize transition-all cursor-pointer ${
                      pooling === p
                        ? 'bg-indigo-600 text-white shadow-sm'
                        : 'bg-slate-950/80 border border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            {/* Learning Rate & Epochs */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-300 font-semibold mb-1">Learning Rate</label>
                <select
                  value={lr}
                  onChange={(e) => setLr(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-xl px-3 py-2 font-mono cursor-pointer"
                >
                  <option value={0.001}>0.001</option>
                  <option value={0.005}>0.005</option>
                  <option value={0.01}>0.01</option>
                  <option value={0.05}>0.05</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-300 font-semibold mb-1">Epochs</label>
                <select
                  value={epochs}
                  onChange={(e) => setEpochs(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-xl px-3 py-2 font-mono cursor-pointer"
                >
                  <option value={15}>15 Epochs</option>
                  <option value={30}>30 Epochs</option>
                  <option value={50}>50 Epochs</option>
                </select>
              </div>
            </div>
          </div>

          <button
            onClick={handleStartTraining}
            disabled={training}
            className="w-full mt-4 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold text-xs transition-all shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 cursor-pointer"
          >
            {training ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" /> Training Model on MUTAG...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" /> Start Live Training
              </>
            )}
          </button>

          {error && (
            <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right: Results Card */}
        <div className="lg:col-span-7 space-y-4">
          {result ? (
            <div className="rounded-3xl bg-slate-900/60 border border-slate-800/80 p-6 backdrop-blur-sm space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">
                    TRAINING COMPLETE
                  </span>
                  <h3 className="text-lg font-bold text-white mt-1">
                    {result.model_name} Evaluation on Test Set
                  </h3>
                </div>
                <CheckCircle2 className="w-6 h-6 text-emerald-400" />
              </div>

              {/* Metrics Summary Row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3.5 rounded-2xl bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] uppercase font-bold text-slate-400">Test Accuracy</div>
                  <div className="text-xl font-black text-emerald-400 mt-1">
                    {(result.test_metrics.accuracy * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="p-3.5 rounded-2xl bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] uppercase font-bold text-slate-400">F1-Score</div>
                  <div className="text-xl font-black text-indigo-400 mt-1">
                    {(result.test_metrics.f1 * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="p-3.5 rounded-2xl bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] uppercase font-bold text-slate-400">Precision</div>
                  <div className="text-xl font-black text-slate-200 mt-1">
                    {(result.test_metrics.precision * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="p-3.5 rounded-2xl bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] uppercase font-bold text-slate-400">ROC-AUC</div>
                  <div className="text-xl font-black text-amber-400 mt-1">
                    {(result.test_metrics.auc * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Loss descent logs preview */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-slate-300">Training History (Last 5 Epochs):</h4>
                <div className="p-3 rounded-2xl bg-slate-950/90 border border-slate-800/80 font-mono text-[11px] text-slate-400 space-y-1">
                  {result.history.train_loss.slice(-5).map((loss, idx) => {
                    const epochNum = result.history.train_loss.length - 5 + idx + 1;
                    const valLoss = result.history.val_loss.slice(-5)[idx];
                    const valAcc = result.history.val_acc.slice(-5)[idx];
                    return (
                      <div key={idx} className="flex justify-between">
                        <span>Epoch {epochNum}:</span>
                        <span>Train Loss = {loss}</span>
                        <span>Val Loss = {valLoss}</span>
                        <span className="text-emerald-400">Val Acc = {(valAcc * 100).toFixed(1)}%</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full min-h-[300px] flex flex-col items-center justify-center p-8 rounded-3xl bg-slate-900/40 border border-dashed border-slate-800 text-center text-slate-500">
              <Sliders className="w-10 h-10 mb-3 opacity-40" />
              <h4 className="font-bold text-slate-300 text-sm">Workbench Ready</h4>
              <p className="text-xs max-w-sm mt-1">
                Select your parameters on the left and click "Start Live Training" to train and evaluate a fresh GNN model.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
