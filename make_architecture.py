"""
generate_architecture_figure.py
---------------------------------
Generates figure2_architecture.png — a large, publication-quality
architecture diagram of KG-CTCN for the IEEE paper.

Sized for 0.75\textwidth in a two-column IEEE layout.

Requirements:
    pip install graphviz
    # Also needs Graphviz system binaries:
    # Windows: https://graphviz.org/download/  (add to PATH during install)
    # Linux:   sudo apt install graphviz
    # Mac:     brew install graphviz
"""

from graphviz import Digraph
import os

OUT_NAME = "figure2_architecture"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

g = Digraph(format='png')

# ── Global graph attributes ───────────────────────────────────────────────────
g.attr(
    rankdir='LR',
    fontname='Helvetica',
    fontsize='22',
    bgcolor='white',
    pad='0.6',
    nodesep='0.7',       # vertical spacing between nodes in same rank
    ranksep='1.1',       # horizontal spacing between ranks (columns)
    dpi='300',           # high-res output
    size='14,8!',        # force canvas to 14×8 inches (! = exact)
    ratio='fill',
)

# ── Default node style ────────────────────────────────────────────────────────
g.attr('node',
    shape='box',
    style='rounded,filled',
    fontname='Helvetica',
    fontsize='18',
    width='2.4',
    height='0.85',
    penwidth='1.8',
)

# ── Default edge style ────────────────────────────────────────────────────────
g.attr('edge',
    fontname='Helvetica',
    fontsize='14',
    penwidth='2.0',
    arrowsize='1.1',
)

# ── Input nodes ───────────────────────────────────────────────────────────────
with g.subgraph(name='cluster_inputs') as inp:
    inp.attr(
        label='Inputs',
        fontsize='20', fontname='Helvetica-Bold',
        style='rounded,dashed',
        color='#7f8c8d', bgcolor='#f9f9f9',
        penwidth='1.5',
    )
    inp.node('W',
        'Weather Sequence\n(28 days)\nRH2M, T2M, Rain',
        fillcolor='#d6eaf8', color='#1a5276', fontcolor='#1a5276')
    inp.node('A',
        'Agronomic Inputs\nVariety, Age\nRatoon, NDVI',
        fillcolor='#d5f5e3', color='#1e8449', fontcolor='#1e8449')

# ── Weather branch (Causal TCN) ───────────────────────────────────────────────
with g.subgraph(name='cluster_tcn') as tcn:
    tcn.attr(
        label='Causal TCN Encoder',
        fontsize='20', fontname='Helvetica-Bold',
        style='rounded,filled',
        color='#1a5276', bgcolor='#eaf4fc',
        penwidth='2.0',
    )
    tcn.node('T1',
        'TCN Layer 1\ndilation = 1',
        fillcolor='#aed6f1', color='#1a5276', fontcolor='#1a5276')
    tcn.node('T2',
        'TCN Layer 2\ndilation = 2',
        fillcolor='#7fb3d3', color='#1a5276', fontcolor='#1a5276')
    tcn.node('T3',
        'TCN Layer 3\ndilation = 4',
        fillcolor='#5499c7', color='#1a5276', fontcolor='white')
    tcn.node('Ew',
        'Temporal\nEmbedding  Eᵥᵥ',
        fillcolor='#1a5276', color='#1a5276', fontcolor='white',
        style='rounded,filled,bold')

# ── Agronomic branch (MLP) ────────────────────────────────────────────────────
with g.subgraph(name='cluster_mlp') as mlp:
    mlp.attr(
        label='Agronomic MLP Encoder',
        fontsize='20', fontname='Helvetica-Bold',
        style='rounded,filled',
        color='#1e8449', bgcolor='#eafaf1',
        penwidth='2.0',
    )
    mlp.node('M1',
        'MLP Layer 1\nReLU',
        fillcolor='#a9dfbf', color='#1e8449', fontcolor='#1e8449')
    mlp.node('M2',
        'MLP Layer 2\nReLU',
        fillcolor='#52be80', color='#1e8449', fontcolor='white')
    mlp.node('Ea',
        'Agronomic\nEmbedding  Eₐ',
        fillcolor='#1e8449', color='#1e8449', fontcolor='white',
        style='rounded,filled,bold')

# ── Knowledge Graph node ──────────────────────────────────────────────────────
g.node('KG',
    'Knowledge Graph\nModulation Weights',
    shape='diamond',
    fillcolor='#fdebd0', color='#784212', fontcolor='#784212',
    fontsize='17',
    width='2.6', height='1.1',
    penwidth='2.0',
)

# ── Fusion & output ───────────────────────────────────────────────────────────
with g.subgraph(name='cluster_fusion') as fus:
    fus.attr(
        label='Knowledge-Guided Fusion',
        fontsize='20', fontname='Helvetica-Bold',
        style='rounded,filled',
        color='#6c3483', bgcolor='#f5eef8',
        penwidth='2.0',
    )
    fus.node('F',
        'Cross-Modal\nAttention Fusion',
        fillcolor='#c39bd3', color='#6c3483', fontcolor='#6c3483',
        width='2.6')
    fus.node('R',
        'Risk Head\nσ(MLP(·))',
        fillcolor='#9b59b6', color='#6c3483', fontcolor='white',
        width='2.6')

g.node('O',
    'Risk Score\n0 → 1',
    shape='oval',
    fillcolor='#fdedec', color='#c0392b', fontcolor='#c0392b',
    fontsize='20', fontname='Helvetica-Bold',
    width='2.2', height='0.9',
    penwidth='2.5',
)

# ── Edges ─────────────────────────────────────────────────────────────────────
# Weather branch
g.edge('W',  'T1', color='#1a5276')
g.edge('T1', 'T2', color='#1a5276')
g.edge('T2', 'T3', color='#1a5276')
g.edge('T3', 'Ew', color='#1a5276')

# Agronomic branch
g.edge('A',  'M1', color='#1e8449')
g.edge('M1', 'M2', color='#1e8449')
g.edge('M2', 'Ea', color='#1e8449')

# Into fusion
g.edge('Ew', 'F',  color='#1a5276', penwidth='2.5')
g.edge('Ea', 'F',  color='#1e8449', penwidth='2.5')
g.edge('KG', 'F',  color='#784212', style='dashed', penwidth='2.0',
       label='  biological\n  priors')

# Fusion to output
g.edge('F',  'R',  color='#6c3483', penwidth='2.5')
g.edge('R',  'O',  color='#c0392b', penwidth='2.5')

# ── Render ────────────────────────────────────────────────────────────────────
out_path = g.render(
    filename=os.path.join(OUT_DIR, OUT_NAME),
    cleanup=True,
)
print(f"Saved: {out_path}")