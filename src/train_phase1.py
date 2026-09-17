import subprocess
import sys
import os

# Ensure dependencies are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ['torch', 'scikit-learn', 'joblib']:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load Data
data_path = '/Users/Nelo1.0/Desktop/Project2/data/processed/usgs_caco3_44k_pitzer_labeled.csv'
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

# Drop NaNs just to be safe
df = df.dropna(subset=['Ca', 'Mg', 'Na', 'Cl', 'SO4', 'HCO3', 'PH', 'TEMP', 'True_SI'])

# 2. Preprocessing
# Ion concentrations vary wildly (from 1 mg/L to 200,000 mg/L). 
# We log10 transform them so the neural network can learn the relationships easily.
ions = ['Ca', 'Mg', 'Na', 'Cl', 'SO4', 'HCO3']
for ion in ions:
    df[f'log_{ion}'] = np.log10(df[ion] + 1e-6) # add 1e-6 to avoid log(0)

features = [f'log_{ion}' for ion in ions] + ['PH', 'TEMP']
target = 'True_SI'

X = df[features].values
y = df[target].values.reshape(-1, 1)

# Standardize features (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 3. Define the PINN Physics Core Architecture
class PhysicsCore(nn.Module):
    def __init__(self, input_dim):
        super(PhysicsCore, self).__init__()
        # A deep, expressive network to map chemistry to thermodynamics
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(), # Swish activation: Smooth and infinitely differentiable (perfect for PINNs)
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)

model = PhysicsCore(input_dim=len(features))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
epochs = 50
print(f"\nTraining Physics Core on {len(X_train)} samples for {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch+1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
            test_loss = criterion(test_preds, y_test_t).item()
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss (MSE): {epoch_loss/len(train_loader):.4f} | Test Loss (MSE): {test_loss:.4f}")

# 5. Evaluation
model.eval()
with torch.no_grad():
    final_preds = model(X_test_t)
    mae = torch.mean(torch.abs(final_preds - y_test_t)).item()
print(f"\nFinal Mean Absolute Error (MAE) on Test Set: {mae:.4f} SI units")

# 6. Save Model and Scaler
os.makedirs('/Users/Nelo1.0/Desktop/Project2/models', exist_ok=True)
torch.save(model.state_dict(), '/Users/Nelo1.0/Desktop/Project2/models/physics_core.pth')
joblib.dump(scaler, '/Users/Nelo1.0/Desktop/Project2/models/scaler.pkl')

print("Saved weights to models/physics_core.pth")
print("Saved scaler to models/scaler.pkl")
