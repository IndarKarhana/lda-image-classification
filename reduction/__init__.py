# Dimensionality reduction module
# 
# Components:
#   - lda.py: Standard LDA wrapper (sklearn)
#   - pca.py: PCA wrapper (sklearn)
#   - feature_profiler.py: Feature Space Profiler (DG-LDA Component 1)
#   - adaptive_components.py: Adaptive Component Selection (DG-LDA Component 2)
#   - regularized_lda.py: Backbone-Adaptive Regularized LDA (DG-LDA Component 3)
#   - cw_lda.py: Confusion-Weighted LDA (DG-LDA Component 4, NOVEL)
#   - dg_lda.py: DG-LDA orchestrator (combines all components)
