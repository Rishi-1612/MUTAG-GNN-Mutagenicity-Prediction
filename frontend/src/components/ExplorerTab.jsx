import React, { useState, useEffect } from 'react';
import { Search, Filter, ChevronLeft, ChevronRight, Dna, Activity, CheckCircle2, XCircle } from 'lucide-react';
import MoleculeViewer from './MoleculeViewer';

export default function ExplorerTab() {
  const [molecules, setMolecules] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [filterLabel, setFilterLabel] = useState('all');
  const [selectedId, setSelectedId] = useState(0);
  const [selectedMol, setSelectedMol] = useState(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingMol, setLoadingMol] = useState(false);

  // Load molecule list on page or filter change
  useEffect(() => {
    async function fetchList() {
      setLoadingList(true);
      try {
        let url = `/api/dataset/molecules?page=${page}&limit=12`;
        if (filterLabel !== 'all') {
          url += `&label=${filterLabel}`;
        }
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          setMolecules(data.molecules);
          setTotal(data.total);
          setTotalPages(data.total_pages);
        }
      } catch (err) {
        console.error('Failed to load molecule list', err);
      } finally {
        setLoadingList(false);
      }
    }
    fetchList();
  }, [page, filterLabel]);

  // Load selected molecule details
  useEffect(() => {
    async function fetchMolecule() {
      setLoadingMol(true);
      try {
        const res = await fetch(`/api/dataset/molecule/${selectedId}`);
        if (res.ok) {
          const data = await res.json();
          setSelectedMol(data);
        }
      } catch (err) {
        console.error('Failed to load molecule details', err);
      } finally {
        setLoadingMol(false);
      }
    }
    fetchMolecule();
  }, [selectedId]);

  return (
    <div className="space-y-6">
      {/* Top Filter and Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <Filter className="w-4 h-4 text-indigo-400" />
          <span className="text-xs font-semibold text-slate-300">Filter by Ground Truth:</span>
          <div className="flex items-center gap-1.5 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => { setFilterLabel('all'); setPage(1); }}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                filterLabel === 'all' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All (188)
            </button>
            <button
              onClick={() => { setFilterLabel('1'); setPage(1); }}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                filterLabel === '1' ? 'bg-rose-500 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Mutagenic (125)
            </button>
            <button
              onClick={() => { setFilterLabel('0'); setPage(1); }}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                filterLabel === '0' ? 'bg-emerald-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Non-Mutagenic (63)
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span>Showing {molecules.length} of {total} compounds</span>
        </div>
      </div>

      {/* Main Split Grid: Left is Gallery list, Right is Detail Viewer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Molecule Grid */}
        <div className="lg:col-span-5 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[640px] overflow-y-auto pr-1">
            {loadingList ? (
              <div className="col-span-2 py-12 text-center text-slate-500 text-xs">
                Loading molecules...
              </div>
            ) : (
              molecules.map((m) => {
                const isSelected = m.id === selectedId;
                const isMutagenic = m.label === 1;

                return (
                  <div
                    key={m.id}
                    onClick={() => setSelectedId(m.id)}
                    className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between ${
                      isSelected
                        ? 'bg-indigo-950/40 border-indigo-500/60 shadow-lg shadow-indigo-500/10 ring-1 ring-indigo-500/30'
                        : 'bg-slate-900/40 hover:bg-slate-900/80 border-slate-800/80 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-bold text-sm text-slate-100">{m.name}</span>
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                          isMutagenic
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        }`}
                      >
                        {m.label_text}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/60">
                      <span>{m.num_nodes} Atoms</span>
                      <span>{m.num_edges} Bonds</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Pagination Controls */}
          <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400">
            <span>Page {page} of {totalPages}</span>
            <div className="flex items-center gap-1">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(p - 1, 1))}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-200 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(p + 1, totalPages))}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-200 transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Detailed Molecule Viewer + Predictions */}
        <div className="lg:col-span-7 space-y-4">
          <MoleculeViewer
            graph={selectedMol}
            title={selectedMol ? `${selectedMol.name} Interactive Topology` : 'Selected Molecule'}
            subtitle={selectedMol ? `${selectedMol.num_nodes} Atoms, ${selectedMol.num_edges} Bonds | Ground Truth: ${selectedMol.label_text}` : ''}
            showImportance={true}
          />

          {/* Multi-Model Consensus Cards */}
          {selectedMol && selectedMol.predictions && (
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-sm space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Activity className="w-4 h-4 text-indigo-400" /> Multi-Model Architectural Consensus
              </h4>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                {Object.entries(selectedMol.predictions).map(([modelName, pred]) => {
                  const isMut = pred.predicted_class === 1;
                  const matchesGroundTruth = pred.predicted_class === selectedMol.label;

                  return (
                    <div
                      key={modelName}
                      className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex flex-col justify-between space-y-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-xs text-slate-200">{modelName}</span>
                        {matchesGroundTruth ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" title="Matches Ground Truth" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5 text-amber-400" title="Disagrees with Ground Truth" />
                        )}
                      </div>

                      <div className="text-xs font-semibold">
                        <span className={isMut ? 'text-rose-400' : 'text-emerald-400'}>
                          {pred.predicted_label}
                        </span>
                      </div>

                      <div className="text-[11px] text-slate-400 font-mono">
                        {pred.confidence}% Conf
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
