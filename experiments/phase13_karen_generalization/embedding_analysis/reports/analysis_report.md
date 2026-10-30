 # Multiscale TitaNet Embedding Analysis Report

This report summarizes the geometric, temporal, and clustering properties of NeMo MSDD TitaNet speaker embeddings across multiple scales.

## Scale_0 (Window Size: 1.5s)
- **Number of Embeddings**: 489
- **Dimensionality**: 192
- **PCA Explained Variance (2D)**: 0.1583
- **Mean Pairwise Cosine Distance**: 0.8443 ± 0.1572
- **Mean Intra-Neighbor Distance (k=5)**: 0.4661
- **Distribution Shape**: Evaluated as **diffuse**
- **Silhouette Score**: N/A (No clustering labels available)

## Scale_1 (Window Size: 1.2s)
- **Number of Embeddings**: 617
- **Dimensionality**: 192
- **PCA Explained Variance (2D)**: 0.1380
- **Mean Pairwise Cosine Distance**: 0.8716 ± 0.1445
- **Mean Intra-Neighbor Distance (k=5)**: 0.4959
- **Distribution Shape**: Evaluated as **diffuse**
- **Silhouette Score**: N/A (No clustering labels available)

## Scale_2 (Window Size: 1.0s)
- **Number of Embeddings**: 742
- **Dimensionality**: 192
- **PCA Explained Variance (2D)**: 0.1248
- **Mean Pairwise Cosine Distance**: 0.8876 ± 0.1339
- **Mean Intra-Neighbor Distance (k=5)**: 0.5139
- **Distribution Shape**: Evaluated as **diffuse**
- **Silhouette Score**: N/A (No clustering labels available)

## Scale_3 (Window Size: 0.8s)
- **Number of Embeddings**: 929
- **Dimensionality**: 192
- **PCA Explained Variance (2D)**: 0.1056
- **Mean Pairwise Cosine Distance**: 0.9059 ± 0.1226
- **Mean Intra-Neighbor Distance (k=5)**: 0.5405
- **Distribution Shape**: Evaluated as **diffuse**
- **Silhouette Score**: N/A (No clustering labels available)

## Scale_4 (Window Size: 0.5s)
- **Number of Embeddings**: 1500
- **Dimensionality**: 192
- **PCA Explained Variance (2D)**: 0.0742
- **Mean Pairwise Cosine Distance**: 0.9361 ± 0.1032
- **Mean Intra-Neighbor Distance (k=5)**: 0.5904
- **Distribution Shape**: Evaluated as **diffuse**
- **Silhouette Score (from clustering labels)**: 0.0428

## General Observations
- **Multiscale Behavior**: As window size decreases, embeddings typically become noisier and more diffuse, affecting speaker separability.
- **Temporal Trajectories**: Trajectory plots show how embeddings transition smoothly or abruptly during speaker turns.
- **BWC Robustness**: Body-worn camera audio presents significant noise challenges, potentially reflected in low explained variance or highly diffuse distributions in shorter window scales.