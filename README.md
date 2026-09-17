# Wellbore Carbonate-Scale Explorer (PINN)

An interactive, explainable AI system for predicting CaCO₃ scale deposition in petroleum wellbores. Built with a dual-phase Physics-Informed Neural Network (PINN) ensemble, served via FastAPI, and visualised with a 3D Three.js dashboard.

## Architecture

- **Phase 1 — Physics Core:** Dense MLP pre-trained on 44,177 USGS brine samples to emulate the Pitzer (HMW84) thermodynamic equations
- **Phase 2 — Probability Layer:** Fine-tuned on 505 field observations (Al-Hajri 2020) using Focal Loss, Dropout, and 4 domain-engineered features
- **Ensemble:** 5 independently seeded PINNs, predictions averaged at inference
- **XAI:** Live Gradient×Input attributions computed per depth step

## Quick Start (Local)

```bash
# Create environment
conda create -n pinn_env python=3.10 -y
conda activate pinn_env

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

## Deploy to Render (Free Tier)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your GitHub repo — Render will detect `render.yaml` automatically
4. Click **Apply** — the app will build and deploy

The live URL will be `https://wellbore-scale-explorer.onrender.com` (or similar).

## Deploy with Docker

```bash
docker build -t wellbore-pinn .
docker run -p 8000:8000 wellbore-pinn
```

## Project Structure

```
├── src/
│   ├── app.py                  # FastAPI backend with XAI
│   ├── train_improved.py       # Improved ensemble training
│   ├── explain_improved.py     # Captum Integrated Gradients
│   └── ...
├── static/
│   └── index.html              # Three.js + XAI dashboard
├── models/
│   ├── ensemble/               # 5 PINN weights (.pth)
│   ├── scaler.pkl              # Physics feature scaler
│   ├── scaler_eng.pkl          # Engineered feature scaler
│   └── optimal_threshold.pkl   # Youden's J threshold
├── Dockerfile
├── render.yaml
├── requirements.txt
└── README.md
```

## Key Metrics (5-Fold Grouped CV)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.732 ± 0.079 |
| Precision | 89.2% ± 3.6% |
| F1 Score | 0.743 ± 0.099 |
| Accuracy | 70.1% ± 10.1% |
