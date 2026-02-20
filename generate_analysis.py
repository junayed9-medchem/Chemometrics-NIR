"""
Milk Powder NIR Spectra - Analysis Script
Dataset: https://www.kaggle.com/datasets/aikemi/datasets-for-chemometrics-courses-snut
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend; change to 'TkAgg' or remove for interactive plots
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = 'C:/Users/junay/OneDrive/Desktop/kaggle/dataset/milk-powder_NIR_spectra.csv/milk-powder_NIR_spectra.csv'   # <-- update path if needed

df = pd.read_csv(DATA_PATH, index_col=0)
labels = df['labels'].astype(int).values
spec   = df.drop('labels', axis=1).astype(float)
X      = spec.values
wavelengths = np.arange(X.shape[1])         # channel indices (2–602 in original)
n_samples, n_channels = X.shape
n_classes = len(np.unique(labels))

print("=" * 60)
print("  MILK POWDER NIR SPECTRA – ANALYSIS SUMMARY")
print("=" * 60)
print(f"  Samples   : {n_samples}")
print(f"  Channels  : {n_channels}")
print(f"  Classes   : {n_classes}  ({np.unique(labels).tolist()})")
print(f"  Samples/class : {pd.Series(labels).value_counts().sort_index().to_dict()}")
print(f"  Reflectance range : [{X.min():.4f}, {X.max():.4f}]")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. COLOUR MAP (one colour per class)
# ─────────────────────────────────────────────────────────────────────────────
colors = cm.tab20(np.linspace(0, 1, n_classes))

# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURE 1 – RAW SPECTRA + MEAN SPECTRA + VARIABILITY
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(20, 5))
fig1.suptitle('Milk Powder NIR – Spectral Overview', fontsize=14, fontweight='bold')

# 3a. All raw spectra
ax = axes[0]
for cls_i, cls in enumerate(np.unique(labels)):
    rows = X[labels == cls]
    for row in rows:
        ax.plot(wavelengths, row, color=colors[cls_i], alpha=0.25, linewidth=0.6)
for cls_i, cls in enumerate(np.unique(labels)):
    ax.plot([], [], color=colors[cls_i], label=f'Class {cls}', linewidth=1.5)
ax.set_xlabel('Channel Index')
ax.set_ylabel('Reflectance')
ax.set_title('All Raw Spectra')
ax.legend(fontsize=6, ncol=2, loc='lower right')

# 3b. Mean spectrum per class
ax = axes[1]
for cls_i, cls in enumerate(np.unique(labels)):
    mean_s = X[labels == cls].mean(axis=0)
    ax.plot(wavelengths, mean_s, color=colors[cls_i], label=f'Class {cls}', linewidth=1.5)
ax.set_xlabel('Channel Index')
ax.set_ylabel('Mean Reflectance')
ax.set_title('Mean Spectrum per Class')
ax.legend(fontsize=6, ncol=2, loc='lower right')

# 3c. Overall mean ± 1 SD
ax = axes[2]
mean_all = X.mean(axis=0)
std_all  = X.std(axis=0)
ax.plot(wavelengths, mean_all, color='steelblue', linewidth=2, label='Mean')
ax.fill_between(wavelengths, mean_all - std_all, mean_all + std_all,
                alpha=0.35, color='steelblue', label='±1 SD')
ax.set_xlabel('Channel Index')
ax.set_ylabel('Reflectance')
ax.set_title('Overall Mean ± 1 SD')
ax.legend()

plt.tight_layout()
plt.savefig('fig1_spectral_overview.png', dpi=150, bbox_inches='tight')
print("[✓] Saved: fig1_spectral_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURE 2 – PCA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=10)
pcs = pca.fit_transform(X_scaled)
ev  = pca.explained_variance_ratio_

print("PCA Explained Variance (first 5 PCs):")
for i, v in enumerate(ev[:5]):
    print(f"  PC{i+1}: {v*100:.2f}%  (cumulative: {ev[:i+1].sum()*100:.2f}%)")
print()

fig2, axes = plt.subplots(1, 3, figsize=(20, 5))
fig2.suptitle('PCA Score Plots', fontsize=14, fontweight='bold')

# Scree plot
ax = axes[0]
ax.bar(range(1, 11), ev * 100, color='steelblue', edgecolor='white')
ax.plot(range(1, 11), np.cumsum(ev) * 100, 'r-o', markersize=5, label='Cumulative')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('Scree Plot')
ax.legend()
ax.set_xticks(range(1, 11))

# PC1 vs PC2
ax = axes[1]
for cls_i, cls in enumerate(np.unique(labels)):
    mask = labels == cls
    ax.scatter(pcs[mask, 0], pcs[mask, 1], color=colors[cls_i],
               label=f'Class {cls}', s=45, alpha=0.85, edgecolors='k', linewidths=0.3)
ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
ax.set_title('PC1 vs PC2')
ax.legend(fontsize=7, ncol=2)

# PC1 vs PC3
ax = axes[2]
for cls_i, cls in enumerate(np.unique(labels)):
    mask = labels == cls
    ax.scatter(pcs[mask, 0], pcs[mask, 2], color=colors[cls_i],
               label=f'Class {cls}', s=45, alpha=0.85, edgecolors='k', linewidths=0.3)
ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)')
ax.set_ylabel(f'PC3 ({ev[2]*100:.1f}%)')
ax.set_title('PC1 vs PC3')
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig('fig2_pca.png', dpi=150, bbox_inches='tight')
print("[✓] Saved: fig2_pca.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 3 – LDA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
# LDA on PCA-reduced data.
# LDA can produce at most min(n_classes-1, n_features) discriminant components.
# We need at least 3 PCs as input to potentially get LD1, LD2, LD3.
# Use whichever is larger: PCs for ≥99% variance OR enough PCs for 3 LDs.
n_pcs_99   = int(np.searchsorted(np.cumsum(ev), 0.99)) + 1
n_pcs_lda  = min(n_classes - 1, X_scaled.shape[1])   # max useful LDs = n_classes-1
n_pcs_input = max(n_pcs_99, n_pcs_lda, 3)            # ensure at least 3 PCs go in
X_pca = pcs[:, :n_pcs_input]

lda = LinearDiscriminantAnalysis()
lda_scores = lda.fit_transform(X_pca, labels)
lda_ev = lda.explained_variance_ratio_
n_lds = lda_scores.shape[1]   # actual number of LDs produced

print(f"LDA: using {n_pcs_input} PCs as input  →  {n_lds} discriminant components")
print(f"LDA LD1 explains {lda_ev[0]*100:.1f}%" +
      (f"  LD2: {lda_ev[1]*100:.1f}%" if n_lds > 1 else "") +
      (f"  LD3: {lda_ev[2]*100:.1f}%" if n_lds > 2 else ""))
print()

# Build plot pairs using only LDs that actually exist
ld_pairs = [(0, 1)] if n_lds < 3 else [(0, 1), (0, 2)]
n_plots  = len(ld_pairs)

fig3, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
if n_plots == 1:
    axes = [axes]          # make iterable
fig3.suptitle('LDA Score Plots', fontsize=14, fontweight='bold')

for ax_idx, (ld_x, ld_y) in enumerate(ld_pairs):
    ax = axes[ax_idx]
    for cls_i, cls in enumerate(np.unique(labels)):
        mask = labels == cls
        ax.scatter(lda_scores[mask, ld_x], lda_scores[mask, ld_y],
                   color=colors[cls_i], label=f'Class {cls}',
                   s=50, alpha=0.85, edgecolors='k', linewidths=0.3)
    ax.set_xlabel(f'LD{ld_x+1} ({lda_ev[ld_x]*100:.1f}%)')
    ax.set_ylabel(f'LD{ld_y+1} ({lda_ev[ld_y]*100:.1f}%)')
    ax.set_title(f'LD{ld_x+1} vs LD{ld_y+1}')
    ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig('fig3_lda.png', dpi=150, bbox_inches='tight')
print("[✓] Saved: fig3_lda.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASSIFICATION – SVM with cross-validation
# ─────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SVM on PCA-reduced data (uses same X_pca built for LDA above)
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
cv_scores = cross_val_score(svm, X_pca, labels, cv=cv, scoring='accuracy')

print("SVM Classification (RBF kernel, 5-fold CV on PCA features):")
print(f"  Accuracy per fold : {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean ± SD         : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE 4 – CONFUSION MATRIX (train-on-all for illustration)
# ─────────────────────────────────────────────────────────────────────────────
svm.fit(X_pca, labels)
y_pred = svm.predict(X_pca)
cm_mat = confusion_matrix(labels, y_pred, labels=np.unique(labels))

fig4, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_mat,
                              display_labels=[f'C{c}' for c in np.unique(labels)])
disp.plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_title('SVM Confusion Matrix (trained on all data)', fontsize=13)
plt.tight_layout()
plt.savefig('fig4_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("[✓] Saved: fig4_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 5 – SPECTRAL VARIABILITY HEATMAP (std per class per channel)
# ─────────────────────────────────────────────────────────────────────────────
std_matrix = np.array([X[labels == cls].std(axis=0) for cls in np.unique(labels)])

fig5, ax = plt.subplots(figsize=(16, 5))
im = ax.imshow(std_matrix, aspect='auto', cmap='hot_r', interpolation='nearest')
ax.set_yticks(range(n_classes))
ax.set_yticklabels([f'Class {c}' for c in np.unique(labels)])
ax.set_xlabel('Channel Index')
ax.set_title('Within-Class Spectral Standard Deviation Heatmap')
fig5.colorbar(im, ax=ax, label='Std Dev (Reflectance)')
plt.tight_layout()
plt.savefig('fig5_variability_heatmap.png', dpi=150, bbox_inches='tight')
print("[✓] Saved: fig5_variability_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  ANALYSIS COMPLETE – Files generated:")
print("  • fig1_spectral_overview.png")
print("  • fig2_pca.png")
print("  • fig3_lda.png")
print("  • fig4_confusion_matrix.png")
print("  • fig5_variability_heatmap.png")
print("=" * 60)
