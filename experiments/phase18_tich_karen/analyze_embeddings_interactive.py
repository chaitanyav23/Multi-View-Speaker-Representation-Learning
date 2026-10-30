"""
Interactive Multiscale Embedding Analysis (Script Version)
==========================================================

This script is structured to be used in an IDE that supports "Code Cells" 
(like VS Code or Spyder using # %%) or run as a standard Python script.

Logic:
- Uses a Global Coordinate System (PCA/UMAP fit on entire file).
- Uses pydash for efficient metadata grouping.
- Uses Plotly for interactive figures.
"""

# %% [Cell 1: Imports]
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import pydash as py_  
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import umap
import os

# %% [Cell 2: Configuration]
# Update these paths if necessary
BASE_DIR = Path("experiments/phase13_karen_generalization")
INPUT_DIR = BASE_DIR / "diarization/msdd_outputs/speaker_outputs"
SCALE = 1  # 1.2s windows

# Ensure directories exist for saving figures if running as a script
(BASE_DIR / "embedding_analysis/interactive_plots").mkdir(parents=True, exist_ok=True)

# %% [Cell 3: Data Loading Function]
def load_data_interactive(scale, input_dir):
    pkl_path = input_dir / f"embeddings/subsegments_scale{scale}_embeddings.pkl"
    json_path = input_dir / f"subsegments_scale{scale}.json"

    if not pkl_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Data for scale {scale} not found at {input_dir}. "
                                f"Ensure MSDD has been run and generated speaker_outputs.")

    print(f"Loading data for Scale {scale}...")
    
    with open(pkl_path, 'rb') as f:
        emb_dict = pickle.load(f)
    
    metadata = []
    with open(json_path, 'r') as f:
        for line in f:
            metadata.append(json.loads(line.strip()))
            
    # Group metadata by audio_id using pydash
    meta_by_audio = py_.group_by(metadata, lambda x: Path(x['audio_filepath']).stem)
        
    all_embeddings = []
    all_metadata = []
    
    for audio_id, emb_tensor in emb_dict.items():
        if audio_id in meta_by_audio:
            metas = sorted(meta_by_audio[audio_id], key=lambda x: x['offset'])
            num_emb = emb_tensor.shape[0]
            
            # Ensure alignment
            min_len = min(len(metas), num_emb)
            # Handle torch tensors if they aren't already numpy
            if hasattr(emb_tensor, 'cpu'):
                data_np = emb_tensor[:min_len].cpu().numpy()
            else:
                data_np = emb_tensor[:min_len]
                
            all_embeddings.append(data_np)
            all_metadata.extend(metas[:min_len])
                
    embeddings_mat = np.vstack(all_embeddings)
    return embeddings_mat, all_metadata

# Execute Load
embeddings, metadata = load_data_interactive(SCALE, INPUT_DIR)
print(f"Total Embeddings: {len(embeddings)}")

# %% [Cell 4: Global Coordinate Fitting]
print("Computing Global PCA & UMAP (Consistency Map)...")
pca = PCA(n_components=2)
pca_proj = pca.fit_transform(embeddings)

reducer = umap.UMAP(n_components=2, random_state=42)
umap_proj = reducer.fit_transform(embeddings)

# Create Master DataFrame
df = pd.DataFrame({
    'time': [m['offset'] for m in metadata],
    'duration': [m['duration'] for m in metadata],
    'pca_1': pca_proj[:, 0],
    'pca_2': pca_proj[:, 1],
    'umap_1': umap_proj[:, 0],
    'umap_2': umap_proj[:, 1]
})

# %% [Cell 5: Global Interactive Plot]
# This shows the entire file. Hover over points to see timestamps.
fig_global = px.scatter(
    df, x='pca_1', y='pca_2', color='time',
    title=f"Full File Global PCA (Scale {SCALE})",
    labels={'pca_1': 'PC1', 'pca_2': 'PC2', 'time': 'Time (s)'},
    color_continuous_scale='Viridis',
    hover_data=['time']
)
fig_global.update_traces(marker=dict(size=5, opacity=0.6))

# If in an interactive environment, this opens a window. 
# If running as a script, we'll also save it.
fig_global.show()
fig_global.write_html(BASE_DIR / "embedding_analysis/interactive_plots/global_pca_master.html")

# %% [Cell 6: Trajectory Visualization Function]
def plot_trajectory(start_time=None, end_time=None, method='pca'):
    """
    Plots the path of the conversation over time.
    If start/end are None, it plots the ENTIRE file.
    """
    plot_df = df.copy()
    if start_time is not None:
        plot_df = plot_df[plot_df['time'] >= start_time]
    if end_time is not None:
        plot_df = plot_df[plot_df['time'] <= end_time]
    
    if plot_df.empty:
        print("No data found for the selected time range.")
        return

    x_col, y_col = f"{method}_1", f"{method}_2"
    
    fig = go.Figure()

    # The Line (Trajectory)
    fig.add_trace(go.Scatter(
        x=plot_df[x_col], y=plot_df[y_col],
        mode='lines+markers',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
        marker=dict(
            size=8, color=plot_df['time'], colorscale='Viridis', 
            showscale=True, colorbar=dict(title="Time (s)")
        ),
        text=[f"Time: {t:.2f}s" for t in plot_df['time']],
        hoverinfo='text'
    ))

    # Markers for start and end of the selection
    fig.add_annotation(x=plot_df[x_col].iloc[0], y=plot_df[y_col].iloc[0], text="START", showarrow=True)
    fig.add_annotation(x=plot_df[x_col].iloc[-1], y=plot_df[y_col].iloc[-1], text="END", showarrow=True)

    fig.update_layout(
        title=f"Conversational Path: {start_time if start_time else 0}s to {end_time if end_time else 'End'}",
        xaxis_title=f"{method.upper()} 1",
        yaxis_title=f"{method.upper()} 2",
        template='plotly_white'
    )
    
    fig.show()

# Example: Run for the whole file
plot_trajectory(method='pca')

# %% [Cell 7: UMAP View (Better Cluster Separation)]
fig_umap = px.scatter(
    df, x='umap_1', y='umap_2', color='time',
    title="Global UMAP (Entire Audio File)",
    color_continuous_scale='Plasma'
)
fig_umap.show()
fig_umap.write_html(BASE_DIR / "embedding_analysis/interactive_plots/global_umap_master.html")
