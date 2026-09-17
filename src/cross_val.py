import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import copy

print("--- RUNNING 5-FOLD GROUPED CROSS VALIDATION ---")

# 1. Load Data
data_path = '/Users/Nelo1.0/Desktop/Project2/data/raw/al_hajri_2020_field_data.xlsx'
df = pd.read_excel(data_path)

# Create unique well ID for grouping
df['Unique_Well'] = df['Field'].astype(str) + "_" + df['Well_Number'].astype(str)

# Feature Engineering
df['Cl_ppm'] = df['TDS_ppm'] - (df['Na_ppm'] + df['Ca_ppm'] + df['Mg_ppm'] + df['SO4_ppm'] + df['HCO3_ppm'])
df['Cl_ppm'] = df['Cl_ppm'].apply(lambda x: x if x > 0 else 100)
df['Temp_C'] = 60.0
df = df.rename(columns={'Na_ppm': 'Na', 'Ca_ppm': 'Ca', 'Mg_ppm': 'Mg', 'SO4_ppm': 'SO4', 'HCO3_ppm': 'HCO3', 'pH': 'PH', 'Cl_ppm': 'Cl', 'Temp_C': 'TEMP'})
df = df.dropna(subset=['Ca', 'Mg', 'Na', 'Cl', 'SO4', 'HCO3', 'PH', 'TEMP', 'Inspection_Result', 'Unique_Well'])

ions = ['Ca', 'Mg', 'Na', 'Cl', 'SO4', 'HCO3']
for ion in ions:
    df[f'log_{ion}'] = np.log10(df[ion] + 1e-6)

features = [f'log_{ion}' for ion in ions] + ['PH', 'TEMP']
X = df[features].values
y = df['Inspection_Result'].values.reshape(-1, 1)
groups = df['Unique_Well'].values

scaler = joblib.load('/Users/Nelo1.0/Desktop/Project2/models/scaler.pkl')
X_scaled = scaler.transform(X)

# 2. Define Model Architecture
class PhysicsCore(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.network(x)

class HybridPINN(nn.Module):
    def __init__(self, physics_core, input_dim):
        super().__init__()
        self.physics_core = physics_core
        self.prob_layer = nn.Sequential(
            nn.Linear(input_dim + 1, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        si = self.physics_core(x)
        return self.prob_layer(torch.cat((x, si), dim=1))

# Base weights (Phase 1)
phase1_path = '/Users/Nelo1.0/Desktop/Project2/models/physics_core.pth'

# 3. K-Fold CV
gkf = GroupKFold(n_splits=5)
fold = 1
accuracies = []
aucs = []

# For average ROC plot
tprs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(8, 6))

for train_idx, test_idx in gkf.split(X_scaled, y, groups):
    print(f"Training Fold {fold}...")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    
    # Initialize fresh model for this fold
    physics_core = PhysicsCore(8)
    physics_core.load_state_dict(torch.load(phase1_path))
    model = HybridPINN(physics_core, 8)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam([
        {'params': model.physics_core.parameters(), 'lr': 1e-4}, 
        {'params': model.prob_layer.parameters(), 'lr': 5e-3}
    ])
    
    # Train
    model.train()
    for epoch in range(120):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t).numpy().flatten()
        test_preds = (test_probs >= 0.5).astype(float)
        
    y_test_flat = y_test.flatten()
    acc = accuracy_score(y_test_flat, test_preds)
    auc_score = roc_auc_score(y_test_flat, test_probs)
    
    accuracies.append(acc)
    aucs.append(auc_score)
    
    # ROC Plotting
    fpr, tpr, _ = roc_curve(y_test_flat, test_probs)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold} (AUC = {auc_score:.2f})')
    
    fold += 1

print("\n--- RESULTS ---")
print(f"Mean Accuracy: {np.mean(accuracies)*100:.2f}% ± {np.std(accuracies)*100:.2f}%")
print(f"Mean ROC-AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# Plot Average ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_tpr = np.std(tprs, axis=0)

ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=0.8)
ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Grouped 5-Fold Cross-Validation ROC Curve")
ax.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('/Users/Nelo1.0/Desktop/Project2/kfold_roc_curve.png', dpi=300, bbox_inches='tight')
print("Saved ROC curve to kfold_roc_curve.png")

# Plot Bar Chart for Accuracies
plt.figure(figsize=(6, 4))
plt.bar(range(1, 6), accuracies, color='teal', alpha=0.7)
plt.axhline(np.mean(accuracies), color='r', linestyle='dashed', linewidth=2, label=f'Mean = {np.mean(accuracies)*100:.1f}%')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('5-Fold Grouped Cross-Validation Accuracies')
plt.ylim(0, 1.0)
plt.legend()
plt.savefig('/Users/Nelo1.0/Desktop/Project2/kfold_accuracies.png', dpi=300, bbox_inches='tight')
print("Saved Bar Chart to kfold_accuracies.png")
