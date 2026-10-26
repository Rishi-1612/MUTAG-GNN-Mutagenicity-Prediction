import React, { useState, useEffect } from 'react';
import {
  Layers,
  Network,
  Cpu,
  Award,
  Sparkles,
  Sliders,
  FileText,
  Dna,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

import OverviewTab from './components/OverviewTab';
import ExplorerTab from './components/ExplorerTab';
import PredictorTab from './components/PredictorTab';
import BenchmarkTab from './components/BenchmarkTab';
import TrainerTab from './components/TrainerTab';
import ReportTab from './components/ReportTab';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [backendHealth, setBackendHealth] = useState(null);

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        if (res.ok) {
          const data = await res.json();
          setBackendHealth(data);
        } else {
          setBackendHealth({ status: 'error' });
        }
      } catch (err) {
        setBackendHealth({ status: 'offline' });
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'overview', label: 'Overview & Pipeline', icon: Layers },
    { id: 'explorer', label: 'MUTAG Explorer', icon: Network },
    { id: 'predictor', label: 'Interactive Predictor', icon: Sparkles },
    { id: 'benchmarks', label: 'Model Benchmarks', icon: Award },
    { id: 'trainer', label: 'GNN Workbench', icon: Sliders },
    { id: 'report', label: 'Academic Defense Brief', icon: FileText },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 gap-4">
            {/* Logo & Title */}
            <div
              className="flex items-center gap-3 cursor-pointer"
              onClick={() => setActiveTab('overview')}
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-rose-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-white">
                <Dna className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-black text-base tracking-tight text-white">MUTAG-GNN</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-semibold">
                    v1.0 Prototype
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 font-medium">Molecular Mutagenicity Platform</p>
              </div>
            </div>

            {/* Desktop Navigation Tabs */}
            <nav className="hidden md:flex items-center gap-1 bg-slate-900/60 p-1 rounded-2xl border border-slate-800/80">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                      isActive
                        ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>

            {/* Status Pills */}
            <div className="flex items-center gap-2">
              {backendHealth?.status === 'healthy' ? (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span className="hidden sm:inline font-mono">Backend Online</span>
                </div>
              ) : (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs font-medium">
                  <span className="w-2 h-2 rounded-full bg-rose-400"></span>
                  <span className="hidden sm:inline font-mono">Connecting...</span>
                </div>
              )}

              <div className="hidden lg:flex items-center gap-1 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-slate-400 text-xs font-mono">
                <span>Top Model:</span>
                <span className="text-emerald-400 font-bold">GIN 79.3%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Row */}
        <div className="md:hidden flex overflow-x-auto px-4 py-2 border-t border-slate-800/60 gap-1 scrollbar-none bg-slate-950">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium shrink-0 ${
                  isActive
                    ? 'bg-indigo-600 text-white'
                    : 'text-slate-400 hover:text-slate-200 bg-slate-900/40'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && <OverviewTab onNavigateTab={setActiveTab} />}
        {activeTab === 'explorer' && <ExplorerTab />}
        {activeTab === 'predictor' && <PredictorTab />}
        {activeTab === 'benchmarks' && <BenchmarkTab />}
        {activeTab === 'trainer' && <TrainerTab />}
        {activeTab === 'report' && <ReportTab />}
      </main>

      {/* Academic Defense Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/90 py-6 mt-12 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-400">Deep Learning Project:</span>
            <span>Graph Neural Networks for Molecular Mutagenicity (MUTAG Benchmark)</span>
          </div>

          <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px]">
            <span>PyTorch Geometric 2.8</span>
            <span>•</span>
            <span>FastAPI Backend</span>
            <span>•</span>
            <span>React + Tailwind UI</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
