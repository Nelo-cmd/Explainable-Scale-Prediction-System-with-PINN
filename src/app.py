import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import numpy as np
import joblib

app = FastAPI(title="Wellbore Scale Explorer", description="PINN Ensemble API")

# ── Resolve project root relative to this file ──
BASE = Path(__file__).resolve().parent.parent

class PhysicsCore(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1))
    def forward(self, x): return self.network(x)

class ImprovedPINN(nn.Module):
    def __init__(self, physics_core, dropout=0.0):
        super().__init__()
        self.physics_core = physics_core
        self.prob_layer = nn.Sequential(
            nn.Linear(13, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, xp, xe):
        si = self.physics_core(xp)
        return si, self.prob_layer(torch.cat((xp, si, xe), dim=1))

# ── Load artefacts ──
print("Loading ensemble...")
scaler_phys = joblib.load(BASE / 'models' / 'scaler.pkl')
scaler_eng  = joblib.load(BASE / 'models' / 'scaler_eng.pkl')
threshold   = joblib.load(BASE / 'models' / 'optimal_threshold.pkl')

ensemble = []
for i in range(5):
    core = PhysicsCore(8)
    m = ImprovedPINN(core, dropout=0.0)
    m.load_state_dict(torch.load(BASE / 'models' / 'ensemble' / f'pinn_{i}.pth',
                                 map_location='cpu'), strict=False)
    m.eval()
    ensemble.append(m)
print(f"Loaded 5 models.  Threshold: {threshold:.4f}")

MW = {'Ca':40.08,'Mg':24.31,'Na':22.99,'Cl':35.45,'SO4':96.06,'HCO3':61.02}
ZZ = {'Ca':2,'Mg':2,'Na':1,'Cl':1,'SO4':2,'HCO3':1}
ions = ['Ca','Mg','Na','Cl','SO4','HCO3']
FEAT_NAMES = ['Ca²⁺','Mg²⁺','Na⁺','Cl⁻','SO₄²⁻','HCO₃⁻','pH','Temp',
              'Ionic Str.','Ca/HCO₃','Ca/Mg','TDS']

def prep(steps):
    X_phys, X_eng, raw_vals = [], [], []
    for s in steps:
        xp = [np.log10(s['Ca']+1e-6), np.log10(s['Mg']+1e-6),
              np.log10(s['Na']+1e-6), np.log10(s['Cl']+1e-6),
              np.log10(s['SO4']+1e-6), np.log10(s['HCO3']+1e-6),
              s['PH'], s['TEMP']]
        I = 0.5*sum((s[i]/(1000*MW[i]))*ZZ[i]**2 for i in ions)
        r1 = (s['Ca']/MW['Ca'])/(s['HCO3']/MW['HCO3']+1e-6)
        r2 = (s['Ca']/MW['Ca'])/(s['Mg']/MW['Mg']+1e-6)
        lt = np.log10(s.get('TDS', sum(s[i] for i in ions))+1e-6)
        X_phys.append(xp)
        X_eng.append([I, r1, r2, lt])
        raw_vals.append([s['Ca'],s['Mg'],s['Na'],s['Cl'],s['SO4'],s['HCO3'],
                         s['PH'],s['TEMP'], I, r1, r2, 10**lt])
    Xp = scaler_phys.transform(np.array(X_phys))
    Xe = scaler_eng.transform(np.array(X_eng))
    return Xp, Xe, raw_vals

@app.post("/simulate")
async def simulate(req: Request):
    data = await req.json()
    steps = data.get('steps', [])
    if not steps: return {"si":[],"prob":[],"attr":[],"feat_names":FEAT_NAMES,"raw":[]}

    Xp, Xe, raw_vals = prep(steps)
    Xp_t = torch.tensor(Xp, dtype=torch.float32)
    Xe_t = torch.tensor(Xe, dtype=torch.float32)

    si_sum  = torch.zeros(len(steps))
    prob_sum = torch.zeros(len(steps))
    attr_sum = np.zeros((len(steps), 12))

    for m in ensemble:
        xp = Xp_t.clone().requires_grad_(True)
        xe = Xe_t.clone().requires_grad_(True)
        si, prob = m(xp, xe)
        si_sum  += si.flatten().detach()
        prob_sum += prob.flatten().detach()
        prob.sum().backward()
        gp = (xp.grad * xp).detach().numpy()
        ge = (xe.grad * xe).detach().numpy()
        attr_sum += np.hstack([gp, ge])

    return {
        "si":   (si_sum  / 5.0).tolist(),
        "prob": (prob_sum / 5.0).tolist(),
        "attr": (attr_sum / 5.0).tolist(),
        "feat_names": FEAT_NAMES,
        "raw": raw_vals
    }

@app.get("/health")
async def health():
    return {"status": "ok", "models": len(ensemble), "threshold": threshold}

app.mount("/", StaticFiles(directory=str(BASE / 'static'), html=True), name="static")
print("API ready → http://localhost:8000")
