"""
Improved Hybrid PINN with:
  1. Focal Loss (handles class imbalance, focuses on hard boundary cases)
  2. Dropout regularisation in the probability layer
  3. 4 engineered domain features (Ionic Strength, Ca/HCO3 ratio, Ca/Mg ratio, log_TDS)
  4. Optimal threshold selection from ROC curve
  5. Ensemble of 5 PINNs (different seeds, averaged probabilities)
  6. 5-Fold Grouped Cross-Validation with all improvements
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, confusion_matrix)
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import copy, os

OUT = '/Users/Nelo1.0/Desktop/Project2'

# ════════════════════════════════════════════════════
# 1.  DATA LOADING & FEATURE ENGINEERING
# ════════════════════════════════════════════════════
print("=" * 60)
print("  IMPROVED HYBRID PINN — FULL PIPELINE")
print("=" * 60)

df = pd.read_excel(f'{OUT}/data/raw/al_hajri_2020_field_data.xlsx')
df['Unique_Well'] = df['Field'].astype(str) + "_" + df['Well_Number'].astype(str)

# Impute Cl via mass balance
df['Cl_ppm'] = df['TDS_ppm'] - (df['Na_ppm'] + df['Ca_ppm'] + df['Mg_ppm'] + df['SO4_ppm'] + df['HCO3_ppm'])
df['Cl_ppm'] = df['Cl_ppm'].apply(lambda x: max(x, 100))
df['Temp_C'] = 60.0

df = df.rename(columns={'Na_ppm':'Na','Ca_ppm':'Ca','Mg_ppm':'Mg',
                         'SO4_ppm':'SO4','HCO3_ppm':'HCO3','pH':'PH',
                         'Cl_ppm':'Cl','Temp_C':'TEMP'})
df = df.dropna(subset=['Ca','Mg','Na','Cl','SO4','HCO3','PH','TEMP','Inspection_Result','Unique_Well'])

# --- log-transform ions (existing 8 features for Physics Core) ---
ions = ['Ca','Mg','Na','Cl','SO4','HCO3']
for ion in ions:
    df[f'log_{ion}'] = np.log10(df[ion] + 1e-6)

# --- 4 NEW engineered features ---
MW = {'Ca':40.08, 'Mg':24.31, 'Na':22.99, 'Cl':35.45, 'SO4':96.06, 'HCO3':61.02}
Z  = {'Ca':2,     'Mg':2,     'Na':1,     'Cl':1,     'SO4':2,     'HCO3':1}

def ionic_strength(row):
    I = 0.0
    for ion in ions:
        m = row[ion] / (1000.0 * MW[ion])   # ppm -> mol/L
        I += m * Z[ion]**2
    return 0.5 * I

df['I_str'] = df.apply(ionic_strength, axis=1)
df['Ca_HCO3_ratio'] = (df['Ca'] / MW['Ca']) / (df['HCO3'] / MW['HCO3'] + 1e-6)
df['Ca_Mg_ratio']   = (df['Ca'] / MW['Ca']) / (df['Mg']  / MW['Mg']  + 1e-6)
df['log_TDS'] = np.log10(df['TDS_ppm'] + 1e-6)

physics_features = [f'log_{ion}' for ion in ions] + ['PH', 'TEMP']   # 8
eng_features     = ['I_str', 'Ca_HCO3_ratio', 'Ca_Mg_ratio', 'log_TDS']  # 4

X_phys = df[physics_features].values
X_eng  = df[eng_features].values
y      = df['Inspection_Result'].values.reshape(-1, 1)
groups = df['Unique_Well'].values

# Scale physics features with the EXISTING Phase-1 scaler
scaler_phys = joblib.load(f'{OUT}/models/scaler.pkl')
X_phys_scaled = scaler_phys.transform(X_phys)

# Fit a NEW scaler for the engineered features
scaler_eng = StandardScaler()
X_eng_scaled = scaler_eng.fit_transform(X_eng)
joblib.dump(scaler_eng, f'{OUT}/models/scaler_eng.pkl')
print(f"Saved engineered-feature scaler → models/scaler_eng.pkl")

# Compute class weight for focal loss alpha
n_pos = y.sum()
n_neg = len(y) - n_pos
alpha_neg = n_pos / len(y)   # weight for class 0
alpha_pos = n_neg / len(y)   # weight for class 1
print(f"Class balance: {int(n_neg)} no-scale ({alpha_neg:.2f}w), {int(n_pos)} scale ({alpha_pos:.2f}w)")


# ════════════════════════════════════════════════════
# 2.  MODEL ARCHITECTURE
# ════════════════════════════════════════════════════

class PhysicsCore(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.network(x)

class ImprovedPINN(nn.Module):
    """Physics core (8→SI) + probability layer (8+SI+4 eng = 13 → prob)"""
    def __init__(self, physics_core, dropout=0.3):
        super().__init__()
        self.physics_core = physics_core
        self.prob_layer = nn.Sequential(
            nn.Linear(13, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x_phys, x_eng):
        si = self.physics_core(x_phys)
        combined = torch.cat((x_phys, si, x_eng), dim=1)   # 8+1+4=13
        return si, self.prob_layer(combined)


class FocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting."""
    def __init__(self, alpha_pos=0.5, gamma=2.0):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = 1.0 - alpha_pos
        self.gamma = gamma
    def forward(self, pred, target):
        bce = -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)
        pt  = target * pred + (1 - target) * (1 - pred)
        alpha_t = target * self.alpha_pos + (1 - target) * self.alpha_neg
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


def make_model(seed=42):
    torch.manual_seed(seed)
    core = PhysicsCore(8)
    core.load_state_dict(torch.load(f'{OUT}/models/physics_core.pth'))
    return ImprovedPINN(core, dropout=0.3)

def optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr                           # Youden's J statistic
    ix = np.argmax(j)
    return thresholds[ix]


# ════════════════════════════════════════════════════
# 3.  5-FOLD GROUPED CROSS-VALIDATION
# ════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  5-FOLD GROUPED CROSS-VALIDATION (IMPROVED)")
print("=" * 60)

gkf = GroupKFold(n_splits=5)
fold_metrics = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_phys_scaled, y, groups), 1):
    print(f"\n--- Fold {fold} ---")

    Xp_tr, Xp_te = X_phys_scaled[tr_idx], X_phys_scaled[te_idx]
    Xe_tr, Xe_te = X_eng_scaled[tr_idx],  X_eng_scaled[te_idx]
    y_tr,  y_te  = y[tr_idx],             y[te_idx]

    Xp_tr_t = torch.tensor(Xp_tr, dtype=torch.float32)
    Xe_tr_t = torch.tensor(Xe_tr, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
    Xp_te_t = torch.tensor(Xp_te, dtype=torch.float32)
    Xe_te_t = torch.tensor(Xe_te, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xp_tr_t, Xe_tr_t, y_tr_t), batch_size=32, shuffle=True)
    criterion = FocalLoss(alpha_pos=alpha_pos, gamma=2.0)

    # --- ensemble of 5 models per fold ---
    ensemble_probs = np.zeros(len(te_idx))
    for seed in range(5):
        model = make_model(seed=fold * 100 + seed)
        opt = optim.Adam([
            {'params': model.physics_core.parameters(), 'lr': 1e-4},
            {'params': model.prob_layer.parameters(),   'lr': 3e-3}
        ], weight_decay=1e-4)

        model.train()
        for epoch in range(200):
            for bp, be, by in loader:
                opt.zero_grad()
                _, prob = model(bp, be)
                loss = criterion(prob, by)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            _, probs = model(Xp_te_t, Xe_te_t)
            ensemble_probs += probs.numpy().flatten()

    ensemble_probs /= 5.0   # average

    # --- optimal threshold ---
    y_flat = y_te.flatten()
    best_t = optimal_threshold(y_flat, ensemble_probs)
    preds  = (ensemble_probs >= best_t).astype(float)

    acc  = accuracy_score(y_flat, preds)
    auc_ = roc_auc_score(y_flat, ensemble_probs)
    prec = precision_score(y_flat, preds, zero_division=0)
    rec  = recall_score(y_flat, preds, zero_division=0)
    f1   = f1_score(y_flat, preds, zero_division=0)

    fold_metrics.append({'fold': fold, 'acc': acc, 'auc': auc_,
                         'prec': prec, 'rec': rec, 'f1': f1, 'thresh': best_t})
    print(f"  Accuracy  : {acc*100:.2f}%  (threshold={best_t:.3f})")
    print(f"  ROC-AUC   : {auc_:.4f}")
    print(f"  Precision : {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}")

    fpr, tpr, _ = roc_curve(y_flat, ensemble_probs)
    interp_tpr = np.interp(mean_fpr, fpr, tpr); interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    ax_roc.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold} (AUC={auc_:.2f})')

# ── summary ──
mdf = pd.DataFrame(fold_metrics)
print("\n" + "=" * 60)
print("  IMPROVED MODEL — SUMMARY")
print("=" * 60)
for col in ['acc','auc','prec','rec','f1']:
    print(f"  {col:>10s}:  {mdf[col].mean():.4f}  ± {mdf[col].std():.4f}")

# ── ROC plot ──
mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_tpr  = np.std(tprs, axis=0)
ax_roc.plot(mean_fpr, mean_tpr, 'b', lw=2, label=f'Mean ROC (AUC={mean_auc:.2f})')
ax_roc.fill_between(mean_fpr,
                     np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1),
                     color='grey', alpha=0.2, label='± 1 std')
ax_roc.plot([0,1],[0,1],'r--',lw=2,alpha=0.6,label='Chance')
ax_roc.set(xlim=[-0.05,1.05], ylim=[-0.05,1.05],
           title='Improved PINN — Grouped 5-Fold ROC', xlabel='FPR', ylabel='TPR')
ax_roc.legend(loc='lower right'); ax_roc.grid(True, ls='--', alpha=0.5)
fig_roc.savefig(f'{OUT}/improved_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close(fig_roc)

# ── accuracy bar chart ──
fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
ax_bar.bar(range(1, 6), mdf['acc'], color='teal', alpha=0.7)
ax_bar.axhline(mdf['acc'].mean(), color='r', ls='--', lw=2,
               label=f'Mean = {mdf["acc"].mean()*100:.1f}%')
ax_bar.set(xlabel='Fold', ylabel='Accuracy',
           title='Improved PINN — 5-Fold Accuracies', ylim=(0, 1))
ax_bar.legend()
fig_bar.savefig(f'{OUT}/improved_accuracies.png', dpi=300, bbox_inches='tight')
plt.close(fig_bar)


# ════════════════════════════════════════════════════
# 4.  TRAIN FINAL ENSEMBLE ON ALL DATA & SAVE
# ════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TRAINING FINAL ENSEMBLE (5 models, all data)")
print("=" * 60)

Xp_all_t = torch.tensor(X_phys_scaled, dtype=torch.float32)
Xe_all_t = torch.tensor(X_eng_scaled,  dtype=torch.float32)
y_all_t  = torch.tensor(y,             dtype=torch.float32)
full_loader = DataLoader(TensorDataset(Xp_all_t, Xe_all_t, y_all_t), batch_size=32, shuffle=True)
criterion = FocalLoss(alpha_pos=alpha_pos, gamma=2.0)

os.makedirs(f'{OUT}/models/ensemble', exist_ok=True)
for seed in range(5):
    model = make_model(seed=seed)
    opt = optim.Adam([
        {'params': model.physics_core.parameters(), 'lr': 1e-4},
        {'params': model.prob_layer.parameters(),   'lr': 3e-3}
    ], weight_decay=1e-4)

    model.train()
    for epoch in range(200):
        for bp, be, by in full_loader:
            opt.zero_grad()
            _, prob = model(bp, be)
            loss = criterion(prob, by)
            loss.backward()
            opt.step()

    path = f'{OUT}/models/ensemble/pinn_{seed}.pth'
    torch.save(model.state_dict(), path)
    print(f"  Saved model {seed} → {path}")

# find global optimal threshold from full-data predictions
all_probs = np.zeros(len(y))
for seed in range(5):
    m = make_model(seed=seed)
    m.load_state_dict(torch.load(f'{OUT}/models/ensemble/pinn_{seed}.pth'))
    m.eval()
    with torch.no_grad():
        _, p = m(Xp_all_t, Xe_all_t)
        all_probs += p.numpy().flatten()
all_probs /= 5.0
global_thresh = optimal_threshold(y.flatten(), all_probs)
print(f"\n  Global optimal threshold: {global_thresh:.4f}")
joblib.dump(global_thresh, f'{OUT}/models/optimal_threshold.pkl')

# final accuracy on training data (sanity check)
final_preds = (all_probs >= global_thresh).astype(float)
print(f"  Full-data accuracy (sanity): {accuracy_score(y.flatten(), final_preds)*100:.2f}%")
print(f"  Full-data ROC-AUC (sanity) : {roc_auc_score(y.flatten(), all_probs):.4f}")

# ── Confusion matrix for the improved model ──
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y.flatten(), final_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Scale', 'Scale'], yticklabels=['No Scale', 'Scale'],
            annot_kws={"size": 16, "weight": "bold"}, ax=ax_cm)
ax_cm.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax_cm.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax_cm.set_title('Improved Ensemble PINN — Confusion Matrix', fontsize=14, fontweight='bold')
fig_cm.savefig(f'{OUT}/improved_conf_matrix.png', dpi=300, bbox_inches='tight')
plt.close(fig_cm)

# ── Comparison table ──
print("\n" + "=" * 60)
print("  BEFORE vs AFTER COMPARISON")
print("=" * 60)
print(f"  {'Metric':<25s} {'Old PINN':>12s} {'Improved':>12s}")
print(f"  {'-'*25} {'-'*12} {'-'*12}")
print(f"  {'CV Mean Accuracy':<25s} {'70.69%':>12s} {mdf['acc'].mean()*100:.2f}%".rjust(12))
print(f"  {'CV Mean ROC-AUC':<25s} {'0.6711':>12s} {mdf['auc'].mean():.4f}".rjust(12))
print(f"  {'CV Mean F1':<25s} {'N/A':>12s} {mdf['f1'].mean():.4f}".rjust(12))
print(f"  {'CV Mean Precision':<25s} {'N/A':>12s} {mdf['prec'].mean():.4f}".rjust(12))
print(f"  {'CV Mean Recall':<25s} {'N/A':>12s} {mdf['rec'].mean():.4f}".rjust(12))
print(f"  {'Threshold':<25s} {'0.5000':>12s} {mdf['thresh'].mean():.4f}".rjust(12))

print("\n✓ All improved models, scalers, and plots saved.")
print("  Next: update src/app.py to serve the ensemble.")
