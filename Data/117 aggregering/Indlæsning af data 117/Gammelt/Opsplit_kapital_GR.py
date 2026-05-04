import numpy as np
import pandas as pd
import sys
from scipy.optimize import minimize

# ── Importer data_formater fra begge mapper ───────────────────────────────────────
sys.path.insert(0, "../Indlæsning af data")
import data_formater69 as dfm_69
import calib69 as calib

sys.path.insert(0, ".")
import data_formater as dfm_117

# ── Konstanter ───────────────────────────────────────────────────────────────────
BRANCHER = ["100010", "100030", "100040x100050"]
N = len(BRANCHER)
t = "TID"
i = "ANVENDELSE"

TILGANG_BRUTTOOVERSKUD = "Bruttooverskud af produktion og blandet indkomst"
TILGANG_LONSUM         = "Aflønning af ansatte"
TILGANG_ANDRESKATTER   = "Andre produktionsskatter, netto"
TILGANG_KOMPONENTER    = [TILGANG_BRUTTOOVERSKUD, TILGANG_LONSUM, TILGANG_ANDRESKATTER]

mapping_117 = {
    "010000 Landbrug og gartneri-(Tilgang)":                 "010000",
    "010000 Landbrug og gartneri- (Anvendelse)":             "010000",
    "100010 Slagterier-(Tilgang)":                           "100010",
    "100010 Slagterier- (Anvendelse)":                       "100010",
    "100030 Mejerier-(Tilgang)":                             "100030",
    "100030 Mejerier- (Anvendelse)":                         "100030",
    "100040x100050 Anden fødevareindustri (100040, 100050)": "100040x100050",
    "REST_TILGANG Øvrige brancher":                          "REST",
    "REST_ANVENDELSE Øvrige brancher":                       "REST",
}

# ── Hjælpefunktioner ─────────────────────────────────────────────────────────────
def _fra_input(path, mapping, tilgang_filter):
    df_input = pd.read_csv(path)
    df_input[i] = df_input[i].replace(mapping)
    data = df_input[
        (df_input["TILGANG1"].isin(tilgang_filter))
        & (df_input["PRISENHED"] == "Løbende priser")
    ]
    return data.groupby([i, t])["INDHOLD"].sum().fillna(0).to_frame()

def _series(df, branche):
    serie = df.xs(branche, level=i)["INDHOLD"]
    serie.name = "INDHOLD"
    return serie

# ── Indlæs BVT fra CSV (117-data) ────────────────────────────────────────────────
PATH_117 = "../Nationalregnskab_117/Data117/input_landbrugsdata.csv"
bvt_117  = _fra_input(PATH_117, mapping_117, TILGANG_KOMPONENTER)

# ── Gamma: produktionsværdi-andel per underbranche ───────────────────────────────
vY_prod_agg = dfm_69.Y['Xt'].loc['10120'] * dfm_69.P['Pt'].loc['10120']

gamma_dict = {}
for b in BRANCHER:
    vY_prod_b = dfm_117.Y['Xt'].loc[b] * dfm_117.P['Pt'].loc[b]
    tid = vY_prod_b.index.intersection(vY_prod_agg.index)
    gamma_dict[b] = vY_prod_b.loc[tid] / vY_prod_agg.loc[tid]

# ── TIDER: skæring af alle dataserier ────────────────────────────────────────────
pK_series = calib.P_K.loc['10120']

TIDER = sorted(
    set(tid)
    .intersection(dfm_69.I['Xt'].loc['10120'].index.tolist())
    .intersection(pK_series.dropna().index.tolist())
)
T = len(TIDER)

# ── Aggregerede størrelser fra 10120 ─────────────────────────────────────────────
K_agg = dfm_69.K['Xt'].loc['10120'].loc[TIDER].values
I_agg = dfm_69.I['Xt'].loc['10120'].loc[TIDER].values
delta = calib.delta.loc['10120'].loc[TIDER].values
pK    = calib.P_K.loc['10120'].loc[TIDER].values

# ── Per-branche data ─────────────────────────────────────────────────────────────
gamma = np.array([[gamma_dict[b].loc[t] for t in TIDER] for b in BRANCHER])  # (N, T)

vBVT = np.array([
    _series(bvt_117, b).loc[TIDER].values
    for b in BRANCHER
])  # (N, T)

vOthCost = np.array([
    (dfm_117.L['Xt'].loc[b] * dfm_117.w['Pt'].loc[b]).loc[TIDER].values
    for b in BRANCHER
])  # (N, T)

# ── Initialfordeling ─────────────────────────────────────────────────────────────
K0     = gamma * K_agg[np.newaxis, :]  # (N, T)
I0     = gamma * I_agg[np.newaxis, :]  # (N, T)
K0_tm1 = gamma[:, 0] * K_agg[0]       # (N,)

lambda_K = 0.01

# ── Pak/udpak ────────────────────────────────────────────────────────────────────
def unpack(x):
    K = x[:N*T].reshape(N, T)
    I = x[N*T:].reshape(N, T)
    return K, I

# ── Objektfunktion ───────────────────────────────────────────────────────────────
def objective_K_only(x):
    K = x.reshape(N, T)
    K_lag = np.concatenate([K0_tm1[:, np.newaxis], K[:, :-1]], axis=1)

    denom  = pK[np.newaxis, :] * K_lag + vOthCost
    markup = vBVT / denom - 1.0

    markup_penalty  = np.sum(markup ** 2)
    capital_penalty = np.sum((K[:, :-1] / K0[:, :-1] - 1.0) ** 2)

    return markup_penalty + lambda_K * capital_penalty

# Nedre grænse: K_{i,t} >= (1-δ_t) * K_{i,t-1}
# Da K_{i,t-1} er en del af optimeringen kan vi ikke sætte dette som bounds direkte
# — det bliver en ulighedsbibetingelse i stedet
constraints = []
for t_idx in range(1, T):
    for i_idx in range(N):
        def non_neg_I(x, i=i_idx, t=t_idx):
            K = x.reshape(N, T)
            return K[i, t] - (1 - delta[t]) * K[i, t-1]  # >= 0
        constraints.append({"type": "ineq", "fun": non_neg_I})

result = minimize(
    objective_K_only,
    K0.ravel(),
    method="SLSQP",
    constraints=constraints,
    options={"maxiter": 10000, "ftol": 1e-12, "disp": True},
)
K_opt = result.x.reshape(N, T)

# Beregn I residualt
K_lag_opt = np.concatenate([K0_tm1[:, np.newaxis], K_opt[:, :-1]], axis=1)
I_opt = K_opt - (1 - delta[np.newaxis, :]) * K_lag_opt


# ── Resultater ────────────────────────────────────────────────────────────────────

rows = [
    {
        "ANVENDELSE": b,
        "TID":    tid,
        "K":      K_opt[i_idx, t_idx],
        "I":      I_opt[i_idx, t_idx],
        "markup": vBVT[i_idx, t_idx] / (pK[t_idx] * K_lag_opt[i_idx, t_idx] + vOthCost[i_idx, t_idx]) - 1.0,
        "gamma":  gamma[i_idx, t_idx],
    }
    for i_idx, b in enumerate(BRANCHER)
    for t_idx, tid in enumerate(TIDER)
]

df_out = pd.DataFrame(rows)
print(df_out)
# df_out.to_csv("../Nationalregnskab_117/Data117/gronreform_output.csv", index=False)