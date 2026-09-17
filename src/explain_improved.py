import torch, torch.nn as nn, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from captum.attr import IntegratedGradients
from matplotlib.patches import Patch

OUT = '/Users/Nelo1.0/Desktop/Project2'

class PhysicsCore(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(d,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x): return self.network(x)

class ImprovedPINN(nn.Module):
    def __init__(self, pc, dropout=0.0):
        super().__init__()
        self.physics_core = pc
        self.prob_layer = nn.Sequential(nn.Linear(13,128),nn.SiLU(),nn.Dropout(dropout),nn.Linear(128,64),nn.SiLU(),nn.Dropout(dropout),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,xp,xe):
        si = self.physics_core(xp)
        return self.prob_layer(torch.cat((xp,si,xe),dim=1))

# Wrapper: single tensor -> scalar output for Captum
class Wrapper(nn.Module):
    def __init__(self, pinn):
        super().__init__()
        self.pinn = pinn
    def forward(self, x):
        return self.pinn(x[:,:8], x[:,8:]).squeeze(-1)

# ── Data ──
df = pd.read_excel(f'{OUT}/data/raw/al_hajri_2020_field_data.xlsx')
df['Cl_ppm'] = (df['TDS_ppm'] - (df['Na_ppm']+df['Ca_ppm']+df['Mg_ppm']+df['SO4_ppm']+df['HCO3_ppm'])).clip(lower=100)
df['Temp_C'] = 60.0
df = df.rename(columns={'Na_ppm':'Na','Ca_ppm':'Ca','Mg_ppm':'Mg','SO4_ppm':'SO4','HCO3_ppm':'HCO3','pH':'PH','Cl_ppm':'Cl','Temp_C':'TEMP'})
ions = ['Ca','Mg','Na','Cl','SO4','HCO3']
for i in ions: df[f'log_{i}'] = np.log10(df[i]+1e-6)
MW={'Ca':40.08,'Mg':24.31,'Na':22.99,'Cl':35.45,'SO4':96.06,'HCO3':61.02}
ZZ={'Ca':2,'Mg':2,'Na':1,'Cl':1,'SO4':2,'HCO3':1}
df['I_str'] = df.apply(lambda r: 0.5*sum((r[i]/(1000*MW[i]))*ZZ[i]**2 for i in ions), axis=1)
df['Ca_HCO3_ratio'] = (df['Ca']/MW['Ca'])/(df['HCO3']/MW['HCO3']+1e-6)
df['Ca_Mg_ratio'] = (df['Ca']/MW['Ca'])/(df['Mg']/MW['Mg']+1e-6)
df['log_TDS'] = np.log10(df['TDS_ppm']+1e-6)

Xp = joblib.load(f'{OUT}/models/scaler.pkl').transform(df[[f'log_{i}' for i in ions]+['PH','TEMP']].values)
Xe = joblib.load(f'{OUT}/models/scaler_eng.pkl').transform(df[['I_str','Ca_HCO3_ratio','Ca_Mg_ratio','log_TDS']].values)
X_all = torch.tensor(np.hstack([Xp, Xe]), dtype=torch.float32)
baseline = torch.zeros_like(X_all)

labels = ['log₁₀(Ca)','log₁₀(Mg)','log₁₀(Na)','log₁₀(Cl)','log₁₀(SO₄)','log₁₀(HCO₃)',
          'pH','Temperature','Ionic Strength','Ca/HCO₃ Ratio','Ca/Mg Ratio','log₁₀(TDS)']

# ── Ensemble-averaged attributions ──
ensemble_attr = np.zeros((len(df), 12))
for seed in range(5):
    print(f"Model {seed}...")
    m = ImprovedPINN(PhysicsCore(8), 0.0)
    m.load_state_dict(torch.load(f'{OUT}/models/ensemble/pinn_{seed}.pth'), strict=False)
    m.eval()
    w = Wrapper(m)
    ig = IntegratedGradients(w)
    attr = ig.attribute(X_all, baselines=baseline, n_steps=100)
    ensemble_attr += attr.detach().numpy()
ensemble_attr /= 5.0

mean_abs = np.abs(ensemble_attr).mean(axis=0)
mean_norm = mean_abs / mean_abs.max()

# ── Plot 1: Feature importance bar chart ──
fig, ax = plt.subplots(figsize=(9, 6))
order = np.argsort(mean_norm)
colors = ['#2dd4bf' if i < 8 else '#f59e0b' for i in order]
ax.barh([labels[i] for i in order], mean_norm[order], color=colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Normalised Attribution (Integrated Gradients)', fontsize=12, fontweight='bold')
ax.set_title('Improved Ensemble PINN — Feature Importance', fontsize=14, fontweight='bold')
ax.legend(handles=[Patch(facecolor='#2dd4bf',label='Physics Features'),Patch(facecolor='#f59e0b',label='Engineered Features')], loc='lower right')
ax.grid(axis='x', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{OUT}/improved_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# ── Plot 2: Attribution heatmap (50 samples) ──
fig2, ax2 = plt.subplots(figsize=(10, 6))
np.random.seed(42)
idx = np.random.choice(len(df), 50, replace=False)
sns.heatmap(ensemble_attr[idx], cmap='RdBu_r', center=0, xticklabels=labels, yticklabels=False,
            ax=ax2, cbar_kws={'label':'Attribution'})
ax2.set_title('Per-Sample Feature Attributions (50 random wells)', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/attribution_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nFeature Importance:")
for i in np.argsort(mean_norm)[::-1]:
    print(f"  {labels[i]:>20s}:  {mean_norm[i]:.4f}")
print("\n✓ Done")
