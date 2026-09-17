import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib

print("--- CHEMISTRY-ONLY BASELINE (ABLATION STUDY) ---")

# 1. Load Data
data_path = '/Users/Nelo1.0/Desktop/Project2/data/raw/al_hajri_2020_field_data.xlsx'
df = pd.read_excel(data_path)

# 2. Feature Engineering
df['Cl_ppm'] = df['TDS_ppm'] - (df['Na_ppm'] + df['Ca_ppm'] + df['Mg_ppm'] + df['SO4_ppm'] + df['HCO3_ppm'])
df['Cl_ppm'] = df['Cl_ppm'].apply(lambda x: x if x > 0 else 100)
df['Temp_C'] = 60.0
df = df.rename(columns={
    'Na_ppm': 'Na', 'Ca_ppm': 'Ca', 'Mg_ppm': 'Mg',
    'SO4_ppm': 'SO4', 'HCO3_ppm': 'HCO3', 'pH': 'PH',
    'Cl_ppm': 'Cl', 'Temp_C': 'TEMP'
})
df = df.dropna(subset=['Ca', 'Mg', 'Na', 'Cl', 'SO4', 'HCO3', 'PH', 'TEMP', 'Inspection_Result'])

ions = ['Ca', 'Mg', 'Na', 'Cl', 'SO4', 'HCO3']
for ion in ions:
    df[f'log_{ion}'] = np.log10(df[ion] + 1e-6)

features = [f'log_{ion}' for ion in ions] + ['PH', 'TEMP']
X = df[features].values
y = df['Inspection_Result'].values.reshape(-1, 1)

scaler = joblib.load('/Users/Nelo1.0/Desktop/Project2/models/scaler.pkl')
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# 3. Define Baseline Model (Standard MLP, No Physics)
class BaselineMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

model = BaselineMLP(input_dim=8)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training Loop
for epoch in range(150):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()

# 5. Evaluation
model.eval()
with torch.no_grad():
    test_probs = model(X_test_t)
    test_preds = (test_probs >= 0.5).float()

test_preds_list = [p[0] for p in test_preds.tolist()]
test_probs_list = [p[0] for p in test_probs.tolist()]
y_test_list = y_test.flatten().tolist()

acc = accuracy_score(y_test_list, test_preds_list)
auc = roc_auc_score(y_test_list, test_probs_list)

print(f"Accuracy: {acc*100:.2f}%")
print(f"ROC-AUC:  {auc:.4f}")
