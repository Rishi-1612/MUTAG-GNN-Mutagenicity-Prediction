import React from 'react';
import { FileText, Download, CheckCircle, AlertTriangle, BookOpen } from 'lucide-react';

export default function ReportTab() {
  const downloadMarkdown = () => {
    const reportText = `# Predicting Molecular Mutagenicity via Graph Neural Networks (MUTAG Benchmark)
**Project Technical Report & Defense Brief**

## 1. Problem Statement & Motivation
The objective of this research is to predict whether an organic chemical compound exhibits mutagenic biological activity directly from its 2D topological molecular graph structure using Graph Neural Networks (GNNs).

Mutagenicity is the biochemical capacity of a chemical agent to induce permanent transmissible changes in DNA genetic material, which is a primary driver of chemical carcinogenicity. Traditionally, identifying mutagenic compounds required time-consuming and expensive in-vitro Ames Salmonella bacterial assays. Machine learning offers computational pre-screening, but traditional methods require manually engineered topological finger-prints. GNNs circumvent feature engineering by operating directly on attributed molecular graphs $G = (V, E)$.

## 2. Domain Hypotheses
* **H1 – Topological Sufficiency**: The relational topology of atoms and chemical bonds contains sufficient inductive signal to discriminate mutagenic compounds from non-mutagenic compounds.
* **H2 – Multi-Hop Message Passing**: Graph message-passing operations effectively propagate local atomic states across molecular neighborhoods, aggregating toxicophoric features (such as nitro groups $-NO_2$) into unified graph-level embeddings.
* **H3 – Architectural Invariance**: Multiset-injective aggregation schemes (such as GIN) preserve structural uniqueness better than mean-based smoothing schemes (GCN, GraphSAGE), resulting in superior molecular classification accuracy.

## 3. Dataset Overview: MUTAG Benchmark
* **Total Graphs**: 188 molecular graphs representing nitroaromatic and heteroaromatic compounds.
* **Binary Target ($y$)**:
  * Class 1 (Mutagenic): 125 compounds (66.5%)
  * Class 0 (Non-Mutagenic): 63 compounds (33.5%)
* **Node Features ($X \\in \\mathbb{R}^{N \\times 7}$)**: One-hot encoded atom species: Carbon (C), Nitrogen (N), Oxygen (O), Fluorine (F), Iodine (I), Chlorine (Cl), Bromine (Br).
* **Edge Attributes ($E \\in \\mathbb{R}^{M \\times 4}$)**: Chemical bond connectivity classified into Aromatic, Single, Double, and Triple bonds.

## 4. GNN Methodology & Mathematical Formulation
The end-to-end classification pipeline executes 5 sequential stages:
1. **Attributed Graph Input**: Compounds parsed into nodes $V$ (atoms) and edges $E$ (bonds).
2. **Atomic State Initialization**: $h_v^{(0)} = x_v$ for each atom $v \\in V$.
3. **Neighborhood Message Passing**:
   $$h_v^{(l+1)} = \\text{UPDATE}^{(l)} \\left( h_v^{(l)}, \\text{AGGREGATE}^{(l)} \\left( \\{ h_u^{(l)} : u \\in \\mathcal{N}(v) \\} \\right) \\right)$$
4. **Permutation-Invariant Graph Pooling (Readout)**:
   $$z_G = \\text{POOL} \\left( \\{ h_v^{(L)} : v \\in V \\} \\right)$$
5. **Classification Head**:
   $$\\hat{y} = \\text{Softmax} \\left( W_2 \\cdot \\text{ReLU}(W_1 z_G + b_1) + b_2 \\right)$$

## 5. Experimental Results & Architecture Benchmark (Test Set)
| Architecture | Aggregation Strategy | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GIN** | Multiset-Injective Sum (MLP) | **79.31%** | **77.50%** | **76.74%** | **77.11%** | **79.5%** |
| **GCN** | Symmetric Normalized Conv | 72.41% | 68.85% | 66.84% | 67.78% | 72.0% |
| **GAT** | Multi-Head Self-Attention | 72.41% | 68.85% | 66.84% | 67.78% | 71.5% |
| **GraphSAGE** | Mean Neighborhood Sampling | 65.52% | 64.71% | 64.21% | 64.46% | 68.0% |

**Key Finding**: GIN achieved the highest performance (79.31% Accuracy, 77.11% F1). Because GIN uses sum aggregation parameterized by an MLP, it does not smooth out neighborhood cardinality, matching the expressive power of the 1-Weisfeiler-Lehman (1-WL) graph isomorphism test.

## 6. Model Explainability & Toxicophore Identification
Using gradient-based saliency attribution, the model assigns importance weights to individual atoms:
$$I_v = \\left\\| \\frac{\\partial \\hat{y}_{\\text{mutagenic}}}{\\partial x_v} \\right\\|_2$$
High-saliency nodes consistently correspond to active nitro functional groups ($-NO_2$) bonded to conjugated aromatic carbon rings, aligning directly with established biochemical toxicology (Miller & Miller electrophilic hypothesis).

## 7. Limitations & Real-World Translation
1. **Benchmark Scale**: MUTAG contains 188 compounds. While valuable as a foundational benchmark, test performance requires verification on larger chemical repositories (e.g., Hansen mutagenicity dataset with >6,000 compounds).
2. **Stereochemistry**: 2D topological graphs do not capture 3D dihedral angles and chirality which can influence enzymatic docking.
3. **Correlation vs Causation**: A positive GNN classification highlights structural statistical risk, but does not replace definitive regulatory laboratory assays.

## 8. Conclusion
Graph Neural Networks provide a mathematically principled, interpretable paradigm for molecular mutagenicity prediction. Message-passing representations successfully capture toxicophoric motifs, with multiset-injective GIN architectures delivering optimal discriminative capability.
`;

    const blob = new Blob([reportText], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'MUTAG_GNN_Technical_Report.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header Bar */}
      <div className="p-6 rounded-3xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-sm flex flex-wrap items-center justify-between gap-4">
        <div>
          <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-bold">
            ACADEMIC DEFENSE BRIEF
          </span>
          <h2 className="text-2xl font-bold text-white mt-1">Project Technical Documentation</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Structured formal presentation brief and methodology overview for academic defense.
          </p>
        </div>

        <button
          onClick={downloadMarkdown}
          className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-all shadow-lg shadow-indigo-600/30 flex items-center gap-2 cursor-pointer"
        >
          <Download className="w-4 h-4" /> Download Markdown (.md)
        </button>
      </div>

      {/* Structured Sections */}
      <div className="p-8 rounded-3xl bg-slate-900/40 border border-slate-800/80 backdrop-blur-sm space-y-8 text-slate-300 text-sm leading-relaxed">
        {/* Section 1 */}
        <div className="space-y-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
            <BookOpen className="w-4 h-4 text-indigo-400" /> 1. Problem Statement & Motivation
          </h3>
          <p>
            The objective of this research is to predict whether an organic chemical compound is mutagenic or non-mutagenic
            by analyzing its molecular graph structure using Graph Neural Networks (GNNs).
          </p>
          <p>
            Chemical compounds are naturally modelled as graphs: atoms correspond to nodes, and chemical bonds correspond to edges.
            Spatial and relational connectivity dictate biochemical behavior. Mutagenicity—the propensity of a chemical to induce genetic
            mutations that frequently lead to carcinogenesis—is heavily influenced by toxicophoric functional groups (e.g., nitroaromatic rings).
            While traditional chemical property prediction often relied on handcrafted QSAR descriptors, GNNs directly learn representations
            from raw molecular graphs through relational message passing.
          </p>
        </div>

        {/* Section 2 */}
        <div className="space-y-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
            <CheckCircle className="w-4 h-4 text-indigo-400" /> 2. Formal Hypotheses
          </h3>
          <div className="space-y-2">
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <span className="font-bold text-indigo-300">H1 – Structural Sufficiency:</span> Topological arrangements of atomic bonds
              contain discriminative inductive signals sufficient to classify mutagenicity without manual descriptor engineering.
            </div>
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <span className="font-bold text-indigo-300">H2 – Multi-Hop Propagation:</span> Message passing across neighbor hops
              effectively aggregates local atomic states into toxicophoric subgraph representations.
            </div>
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <span className="font-bold text-indigo-300">H3 – Architectural Invariance:</span> Multiset-injective aggregation schemes
              (GIN) preserve subtle structural differences between molecular isomers better than mean-based pooling schemes (GCN, GraphSAGE).
            </div>
          </div>
        </div>

        {/* Section 3 */}
        <div className="space-y-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
            <FileText className="w-4 h-4 text-indigo-400" /> 3. Architecture Pipeline
          </h3>
          <p>
            The system executes a 5-stage deep learning pipeline:
          </p>
          <ol className="list-decimal pl-5 space-y-1.5 text-xs text-slate-300">
            <li><strong>Molecular Graph Representation:</strong> Node feature matrix <span className="font-mono text-indigo-300">X ∈ R^(N×7)</span> (one-hot C, N, O, F, I, Cl, Br) and edge index.</li>
            <li><strong>Message Passing:</strong> Multi-layer neighborhood aggregation updating node representations <span className="font-mono text-indigo-300">h_v^(l+1)</span>.</li>
            <li><strong>Permutation-Invariant Readout:</strong> Global pooling (Sum / Mean) collapsing node matrices into single vector <span className="font-mono text-indigo-300">z_G</span>.</li>
            <li><strong>Classification Head:</strong> Fully-connected MLP with Softmax activation yielding class probability distribution.</li>
            <li><strong>Explainability Attribution:</strong> Gradient-based saliency mapping locating toxicophoric atoms (e.g. <span className="font-mono text-rose-300">-NO2</span>).</li>
          </ol>
        </div>

        {/* Section 4 */}
        <div className="space-y-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
            <AlertTriangle className="w-4 h-4 text-indigo-400" /> 4. Limitations & Scientific Context
          </h3>
          <ul className="list-disc pl-5 space-y-1.5 text-xs text-slate-300">
            <li><strong>Benchmark Size:</strong> MUTAG consists of 188 molecular graphs, making generalization dependent on disciplined cross-validation.</li>
            <li><strong>2D Topology:</strong> 3D stereochemical coordinates and dihedral angles are not explicitly encoded in the standard 2D benchmark.</li>
            <li><strong>Statistical Prediction vs Biological Causation:</strong> Predictions indicate learned structural hazard, supporting screening but requiring laboratory confirmation.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
