import numpy as np

# --- 1. THERMODYNAMIC CONSTANTS ---
def _log10(x): return np.log10(x)
def logK1(T): return (-356.3094 - 0.06091964 * T + 21834.37 / T + 126.8339 * _log10(T) - 1684915.0 / T ** 2)
def logK2(T): return (-107.8871 - 0.03252849 * T + 5151.79 / T + 38.92561 * _log10(T) - 563713.9 / T ** 2)
def logKsp_calcite(T): return (-171.9065 - 0.077993 * T + 2839.319 / T + 71.595 * _log10(T))
def logKsp_aragonite(T): return (-171.9773 - 0.077993 * T + 2903.293 / T + 71.595 * _log10(T))
def K1(T): return 10.0 ** logK1(T)
def K2(T): return 10.0 ** logK2(T)
def Ksp_calcite(T): return 10.0 ** logKsp_calcite(T)
def Ksp_aragonite(T): return 10.0 ** logKsp_aragonite(T)

# --- 2. PITZER CONSTANTS ---
N_A, E_CHG, EPS0, KB, BVAL = 6.02214076e23, 1.602176634e-19, 8.8541878128e-12, 1.380649e-23, 1.2
ZI = {"Na": 1, "Ca": 2, "Mg": 2, "Cl": -1, "SO4": -2, "HCO3": -1, "CO3": -2}
CATIONS = ["Na", "Ca", "Mg"]
ANIONS = ["Cl", "SO4", "HCO3", "CO3"]

BETA = {
    ("Na", "Cl"):  (0.0765, 0.2664, 0.0, 0.00127),
    ("Na", "SO4"): (0.01958, 1.113, 0.0, 0.00497),
    ("Na", "HCO3"): (0.0277, 0.0411, 0.0, 0.0),
    ("Na", "CO3"): (0.0399, 1.389, 0.0, 0.0044),
    ("Ca", "Cl"):  (0.3159, 1.614, 0.0, -0.00034),
    ("Ca", "SO4"): (0.20, 3.1973, -54.24, 0.0),
    ("Ca", "HCO3"): (0.4, 2.977, 0.0, 0.0),
    ("Ca", "CO3"): (0.0, 0.0, 0.0, 0.0),
    ("Mg", "Cl"):  (0.35235, 1.6815, 0.0, 0.00519),
    ("Mg", "SO4"): (0.221, 3.343, -37.23, 0.025),
    ("Mg", "HCO3"): (0.329, 0.6072, 0.0, 0.0),
    ("Mg", "CO3"): (0.0, 0.0, 0.0, 0.0),
}
THETA = {
    ("Na", "Ca"): 0.07, ("Na", "Mg"): 0.07, ("Ca", "Mg"): 0.007,
    ("Cl", "SO4"): 0.02, ("Cl", "HCO3"): 0.03, ("Cl", "CO3"): -0.02,
    ("SO4", "HCO3"): 0.01, ("SO4", "CO3"): 0.02, ("HCO3", "CO3"): -0.04,
}
PSI = {
    ("Na", "Ca", "Cl"): -0.007, ("Na", "Mg", "Cl"): -0.012,
    ("Ca", "Mg", "Cl"): -0.012, ("Na", "Cl", "SO4"): 0.0014,
    ("Na", "Cl", "HCO3"): -0.015, ("Na", "Cl", "CO3"): 0.0085,
    ("Na", "SO4", "CO3"): -0.005, ("Na", "HCO3", "CO3"): -0.04,
    ("Ca", "Cl", "SO4"): -0.018, ("Mg", "Cl", "SO4"): -0.004,
}

# --- 3. HELPER FUNCTIONS ---
def _pair(a, b, table): return table.get((a, b)) or table.get((b, a))

def water_density(T):
    t = T - 273.15
    return (999.83952 + 16.945176 * t - 7.9870401e-3 * t ** 2 - 46.170461e-6 * t ** 3 + 105.56302e-9 * t ** 4 - 280.54253e-12 * t ** 5) / (1 + 16.879850e-3 * t)

def dielectric(T):
    t = T - 273.15
    return 87.740 - 0.40008 * t + 9.398e-4 * t ** 2 - 1.410e-6 * t ** 3

def A_phi(T):
    rho = water_density(T)
    eps = dielectric(T) * EPS0
    term = (E_CHG ** 2 / (eps * KB * T)) ** 1.5
    return (1.0 / 3.0) * np.sqrt(2 * np.pi * N_A * rho) * term * (1.0 / (4 * np.pi)) ** 1.5

def _g(x):
    x = np.where(x == 0, 1e-12, x)
    return 2.0 * (1.0 - (1.0 + x) * np.exp(-x)) / x ** 2
def _gp(x):
    x = np.where(x == 0, 1e-12, x)
    return -2.0 * (1.0 - (1.0 + x + x ** 2 / 2.0) * np.exp(-x)) / x ** 2

def _alpha(zc, za): return (1.4, 12.0) if (abs(zc) == 2 and abs(za) == 2) else (2.0, 12.0)

def _B(a, b, I):
    p = _pair(a, b, BETA)
    if p is None: return 0.0, 0.0
    b0, b1, b2, _ = p
    zc = ZI[a] if a in CATIONS else ZI[b]
    za = ZI[b] if b in ANIONS else ZI[a]
    al1, al2 = _alpha(zc, za)
    sI = np.sqrt(I)
    return b0 + b1 * _g(al1 * sI) + b2 * _g(al2 * sI), (b1 * _gp(al1 * sI) + b2 * _gp(al2 * sI)) / I

def _C(a, b):
    p = _pair(a, b, BETA)
    if p is None: return 0.0
    zc = abs(ZI[a] if a in CATIONS else ZI[b])
    za = abs(ZI[b] if b in ANIONS else ZI[a])
    return p[3] / (2.0 * np.sqrt(zc * za))

# --- 4. FAST J FUNCTION ---
def _J_orig(x):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= 0: continue
        t = np.linspace(1e-4, 40, 4000)
        out[i] = xi / 4.0 - 1.0 + np.trapezoid(1.0 - np.exp(-(xi / t) * np.exp(-t)), t)
    return out

_X_TABLE = None
_J_TABLE = None
def _build_table():
    global _X_TABLE, _J_TABLE
    _X_TABLE = np.logspace(-3, np.log10(200), 800)
    _J_TABLE = np.array([_J_orig(x)[0] for x in _X_TABLE])

def _J(x):
    global _X_TABLE, _J_TABLE
    if _X_TABLE is None: _build_table()
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    pos = x > 0
    if pos.any(): out[pos] = np.interp(x[pos], _X_TABLE, _J_TABLE)
    return out

def _Jp(x):
    h = 1e-4
    return (_J(np.asarray(x, dtype=float) + h) - _J(np.asarray(x, dtype=float) - h)) / (2.0 * h)

def _Etheta(zi, zj, I, Aph):
    if zi == zj: return 0.0, 0.0
    sI = np.sqrt(I)
    xij, xii, xjj = 6.0 * zi * zj * Aph * sI, 6.0 * zi * zi * Aph * sI, 6.0 * zj * zj * Aph * sI
    Jij, Jii, Jjj = _J(xij)[0], _J(xii)[0], _J(xjj)[0]
    Jpij, Jpii, Jpjj = _Jp(xij)[0], _Jp(xii)[0], _Jp(xjj)[0]
    Eth = (zi * zj / (4.0 * I)) * (Jij - 0.5 * Jii - 0.5 * Jjj)
    Ethp = -(Eth / I) + (zi * zj / (8.0 * I ** 2)) * (xij * Jpij - 0.5 * xii * Jpii - 0.5 * xjj * Jpjj)
    return Eth, Ethp

# --- 5. ACTIVITY COEFFICIENTS ---
def activity_coefficients(m, T=298.15, use_etheta=True):
    Aph = A_phi(T)
    cats = [c for c in CATIONS if m.get(c, 0) > 0]
    ans = [a for a in ANIONS if m.get(a, 0) > 0]
    I = 0.5 * sum(m[i] * ZI[i] ** 2 for i in cats + ans)
    if I <= 0: return {i: 1.0 for i in cats + ans}
    sI, Z = np.sqrt(I), sum(m[i] * abs(ZI[i]) for i in cats + ans)

    F = -Aph * (sI / (1 + BVAL * sI) + (2.0 / BVAL) * np.log(1 + BVAL * sI))
    for c in cats:
        for a in ans:
            F += m[c] * m[a] * _B(c, a, I)[1]
    
    if use_etheta:
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)): F += m[cats[i]] * m[cats[j]] * _Etheta(ZI[cats[i]], ZI[cats[j]], I, Aph)[1]
        for i in range(len(ans)):
            for j in range(i + 1, len(ans)): F += m[ans[i]] * m[ans[j]] * _Etheta(ZI[ans[i]], ZI[ans[j]], I, Aph)[1]

    gamma = {}
    for M in cats:
        zM = ZI[M]
        ln = zM ** 2 * F
        for a in ans: ln += m[a] * (2 * _B(M, a, I)[0] + Z * _C(M, a))
        for c in cats:
            if c == M: continue
            phi = (_pair(M, c, THETA) or 0.0) + (_Etheta(zM, ZI[c], I, Aph)[0] if use_etheta else 0.0)
            psi_sum = sum(m[a] * (PSI.get(tuple(sorted([M, c])) + (a,)) or PSI.get((min(M, c), max(M, c), a)) or 0.0) for a in ans)
            ln += m[c] * (2 * phi + psi_sum)
        for i in range(len(ans)):
            for j in range(i + 1, len(ans)): ln += m[ans[i]] * m[ans[j]] * (PSI.get((M, ans[i], ans[j])) or PSI.get((M, ans[j], ans[i])) or 0.0)
        for c in cats:
            for a in ans: ln += abs(zM) * m[c] * m[a] * _C(c, a)
        gamma[M] = float(np.exp(ln))

    for X in ans:
        zX = ZI[X]
        ln = zX ** 2 * F
        for c in cats: ln += m[c] * (2 * _B(c, X, I)[0] + Z * _C(c, X))
        for a in ans:
            if a == X: continue
            phi = (_pair(X, a, THETA) or 0.0) + (_Etheta(zX, ZI[a], I, Aph)[0] if use_etheta else 0.0)
            psi_sum = sum(m[c] * (PSI.get((c, min(X, a), max(X, a))) or PSI.get((c, max(X, a), min(X, a))) or 0.0) for c in cats)
            ln += m[a] * (2 * phi + psi_sum)
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)): ln += m[cats[i]] * m[cats[j]] * (PSI.get((cats[i], cats[j], X)) or PSI.get((cats[j], cats[i], X)) or 0.0)
        for c in cats:
            for a in ans: ln += abs(zX) * m[c] * m[a] * _C(c, a)
        gamma[X] = float(np.exp(ln))

    return gamma

