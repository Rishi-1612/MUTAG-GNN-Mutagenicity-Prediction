import React, { useEffect, useState } from 'react';
import { Network, Dna, Brain, Zap, CheckCircle2, ArrowRight, Activity, Layers, BarChart3 } from 'lucide-react';
import MoleculeViewer from './MoleculeViewer';

export default function OverviewTab({ onNavigateTab }) {
  const [overview, setOverview] = useState(null);
  const [previewMol, setPreviewMol] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [ovRes, molRes] = await Promise.all([
          fetch('/api/dataset/overview'),
          fetch('/api/dataset/molecule/0')
        ]);
        if (ovRes.ok) setOverview(await ovRes.json());
        if (molRes.ok) setPreviewMol(await molRes.json());
      } catch (err) {
        console.error('Failed to load overview data', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div className="space-y-8">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-indigo-950/70 via-slate-900/90 to-slate-950 p-8 border border-indigo-500/20 shadow-2xl backdrop-blur-xl">
        <div className="absolute -right-16 -top-16 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute right-40 -bottom-20 w-80 h-80 bg-rose-500/10 rounded-full blur-3xl pointer-events-none"></div>

        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-xs font-semibold tracking-wide">
            <SparkleIcon className="w-3.5 h-3.5" /> GNN Molecular Mutagenicity Benchmark
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
            Predicting Chemical Mutagenicity with Graph Neural Networks
          </h1>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            An end-to-end deep learning framework that models organic compounds as attributed molecular graphs,
            learns atomic message-passing representations, and classifies mutagenic toxicity using the standard
            benchmark MUTAG dataset.
          </p>

          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigateTab('predictor')}
              className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/30 flex items-center gap-2 cursor-pointer"
            >
              Test Custom Molecule <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigateTab('benchmarks')}
              className="px-5 py-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700 font-semibold text-sm transition-all flex items-center gap-2 cursor-pointer"
            >
              View Model Comparison
            </button>
          </div>
        </div>
      </div>

      {/* Metric Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-sm">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Dataset Size</span>
            <Dna className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-black text-white">188</div>
          <p className="text-xs text-slate-400 mt-1">Molecular graphs (nitroaromatic & heteroaromatic)</p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-sm">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Node Features</span>
            <Network className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-black text-white">7 Dim</div>
          <p className="text-xs text-slate-400 mt-1">Atoms: C, N, O, F, I, Cl, Br</p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-sm">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Best Accuracy</span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-black text-emerald-400">79.3%</div>
          <p className="text-xs text-slate-400 mt-1">GIN (Graph Isomorphism Net) test set</p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-sm">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Architectures</span>
            <Brain className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-black text-white">4 Models</div>
          <p className="text-xs text-slate-400 mt-1">GCN, GraphSAGE, GIN, GAT</p>
        </div>
      </div>

      {/* Methodology & GNN Pipeline */}
      <div className="rounded-3xl bg-slate-900/60 border border-slate-800/80 p-6 md:p-8 backdrop-blur-sm space-y-6">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Layers className="w-5 h-5 text-indigo-400" /> End-to-End Methodology & GNN Pipeline
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            How Graph Neural Networks learn relational biochemical patterns for graph-level mutagenicity classification.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2 relative">
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-bold">STAGE 1</span>
            <h4 className="text-sm font-semibold text-slate-200">Molecular Graph Input</h4>
            <p className="text-xs text-slate-400">
              Molecules parsed into attributed graphs <span className="font-mono text-slate-300">G = (V, E)</span>. Atoms are vertices, chemical bonds are edges.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 font-bold">STAGE 2</span>
            <h4 className="text-sm font-semibold text-slate-200">Atom & Bond Features</h4>
            <p className="text-xs text-slate-400">
              One-hot atomic vectors <span className="font-mono text-slate-300">X ∈ R^(N×7)</span> and 4-way bond attributes (Aromatic, Single, Double, Triple).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">STAGE 3</span>
            <h4 className="text-sm font-semibold text-slate-200">Message Passing</h4>
            <p className="text-xs text-slate-400">
              Atoms aggregate features from topological neighbors across multiple message passing hops: <span className="font-mono text-slate-300">h_v^(l+1)</span>.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-bold">STAGE 4</span>
            <h4 className="text-sm font-semibold text-slate-200">Graph Readout (Pooling)</h4>
            <p className="text-xs text-slate-400">
              Permutation-invariant pooling (Sum / Mean) condenses all atomic embeddings into a single molecular vector <span className="font-mono text-slate-300">z_G</span>.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-bold">STAGE 5</span>
            <h4 className="text-sm font-semibold text-slate-200">Classification Head</h4>
            <p className="text-xs text-slate-400">
              MLP classifier with Softmax yields predicted probability distribution: Mutagenic vs Non-Mutagenic.
            </p>
          </div>
        </div>
      </div>

      {/* Dataset Breakdown & Sample Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Class & Feature Statistics */}
        <div className="lg:col-span-5 rounded-3xl bg-slate-900/60 border border-slate-800/80 p-6 backdrop-blur-sm space-y-5">
          <h3 className="font-bold text-white text-base flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-indigo-400" /> MUTAG Benchmark Composition
          </h3>

          {/* Class Ratio Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-semibold">
              <span className="text-rose-400">Mutagenic (Class 1): 125 graphs (66.5%)</span>
              <span className="text-emerald-400">Non-Mutagenic (Class 0): 63 graphs (33.5%)</span>
            </div>
            <div className="h-3 w-full bg-slate-800 rounded-full overflow-hidden flex">
              <div className="bg-rose-500 h-full" style={{ width: '66.5%' }} title="Mutagenic: 125"></div>
              <div className="bg-emerald-500 h-full" style={{ width: '33.5%' }} title="Non-Mutagenic: 63"></div>
            </div>
          </div>

          {/* Atom distribution tags */}
          <div className="space-y-2 pt-2">
            <span className="text-xs font-semibold text-slate-400">Atom Distribution Across Dataset:</span>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-300 font-medium">Carbon (C)</span>
                <span className="font-mono text-slate-400">2,395 atoms</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 flex justify-between items-center">
                <span className="text-blue-300 font-medium">Nitrogen (N)</span>
                <span className="font-mono text-slate-400">345 atoms</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 flex justify-between items-center">
                <span className="text-rose-300 font-medium">Oxygen (O)</span>
                <span className="font-mono text-slate-400">593 atoms</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 flex justify-between items-center">
                <span className="text-cyan-300 font-medium">Chlorine (Cl)</span>
                <span className="font-mono text-slate-400">23 atoms</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 flex justify-between items-center">
                <span className="text-emerald-300 font-medium">Fluorine (F)</span>
                <span className="font-mono text-slate-400">12 atoms</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 flex justify-between items-center">
                <span className="text-amber-300 font-medium">Bromine/Iodine</span>
                <span className="font-mono text-slate-400">3 atoms</span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-900/50 text-xs text-indigo-200/90 leading-relaxed">
            <span className="font-bold text-indigo-100">Key Scientific Insight:</span> Mutagenicity is strongly triggered by the presence of electron-withdrawing nitro groups (<span className="font-mono text-rose-300">-NO2</span>) attached to conjugated aromatic rings, which undergo biological reduction into reactive hydroxylamines that bind to DNA.
          </div>
        </div>

        {/* Right: Live Sample Molecule Viewer */}
        <div className="lg:col-span-7">
          <MoleculeViewer
            graph={previewMol}
            title="MUTAG Molecule #1 Preview (1-Nitroaromatic Compound)"
            subtitle="17 atoms (14 C, 1 N, 2 O), 19 bonds. Highlighting nitro group toxicophore saliency."
            showImportance={true}
          />
        </div>
      </div>
    </div>
  );
}

function SparkleIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
    </svg>
  );
}
