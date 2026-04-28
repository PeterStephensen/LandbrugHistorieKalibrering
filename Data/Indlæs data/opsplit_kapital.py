"""
opsplit_kapital.py
==================
Splitter kapitalapparatets 10120-aggregat (Føde-, drikke- og tobaksvareindustri)
ud i tre undersektorer vha. gamma-vægte beregnet fra BVT og lønsum.

Metode (fra Opsplit_kapital_BVT.py):
    gamma_b = alpha * (BVT_b / BVT_10120) + (1-alpha) * (lønsum_b / lønsum_10120)

    BVT  = Bruttooverskud + Afloening af ansatte + Andre produktionsskatter, netto
    Data til BVT og lønsum hentes fra NAIO3 (input) på begge niveauer.

Brug:
    from import_data import load_all
    from opsplit_kapital import beregn_gamma, opsplit_kapital, beregn_kapital_117

    data        = load_all()
    gamma       = beregn_gamma(data["input"], data["input_117"])
    kapital_117 = opsplit_kapital(data["kapital"], gamma)

    # Eller i ét kald:
    kapital_117 = beregn_kapital_117(data)
"""

import pandas as pd
import numpy as np
from omregning import (
    BRANCHE_MAPPING,
    ANDEN_FODE_NAVN,
    beregn_prisindeks_kapital,
    beregn_mængdeindeks_kapital,
)

# ===========================================================================
# Konstanter
# ===========================================================================

# BVT-komponenter (TILGANG1-vaerdier fra NAIO3)
TILGANG_BVT = [
    "Bruttooverskud af produktion og blandet indkomst",
    "Aflønning af ansatte",
    "Andre produktionsskatter, netto",
]
TILGANG_LØNSUM = "Aflønning af ansatte"

# 10120-aggregatets navn i kapital-datasættet (fra load_kapital)
KAPITAL_10120 = "10120 Føde-, drikke- og tobaksvareindustri"

# Kort kode -> fuldt branche-navn i output
# Navnene matcher hvad load_lontimer_117 / load_input_117 bruger
GAMMA_BRANCHER = {
    "100010":     "100010 Slagterier",
    "100030":     "100030 Mejerier",
    "anden_fode": ANDEN_FODE_NAVN,
}


# ===========================================================================
# Beregning af gamma-vægte
# ===========================================================================

def beregn_gamma(
    df_input_69:  pd.DataFrame,
    df_input_117: pd.DataFrame,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Beregner gamma-vægte til opsplitning af 10120-aggregatet på tre undersektorer.

    For hver branche b i {100010, 100030, anden_fode}:
        gamma_b = alpha  * (BVT_b  / BVT_10120)
                + (1-alpha) * (lønsum_b / lønsum_10120)

    Parametre
    ----------
    df_input_69  : data["input"]     fra load_all()  (NAIO3, 69-niveau)
    df_input_117 : data["input_117"] fra load_all()  (NAIO3, 117-niveau)
    alpha        : vægt på BVT-forholdet (default 0.5)

    Returnerer
    ----------
    DataFrame med kolonner: ANVENDELSE (kort kode), TID, gamma
    """
    def _aggreger(df, tilgang_filter):
        d = df.copy()
        d["ANVENDELSE"] = d["ANVENDELSE"].replace(BRANCHE_MAPPING)
        return (
            d.loc[
                d["TILGANG1"].isin(tilgang_filter) &
                (d["PRISENHED"] == "Løbende priser")
            ]
            .groupby(["ANVENDELSE", "TID"])["INDHOLD"]
            .sum()
            .fillna(0)
        )

    bvt_69     = _aggreger(df_input_69,  TILGANG_BVT)
    lønsum_69  = _aggreger(df_input_69,  [TILGANG_LØNSUM])
    bvt_117    = _aggreger(df_input_117, TILGANG_BVT)
    lønsum_117 = _aggreger(df_input_117, [TILGANG_LØNSUM])

    bvt_10120    = bvt_69.xs("10120",    level="ANVENDELSE")
    lønsum_10120 = lønsum_69.xs("10120", level="ANVENDELSE")

    rows = []
    for kort_kode in GAMMA_BRANCHER:
        bvt_b    = bvt_117.xs(kort_kode,    level="ANVENDELSE")
        lønsum_b = lønsum_117.xs(kort_kode, level="ANVENDELSE")

        tid = (
            bvt_b.index
            .intersection(bvt_10120.index)
            .intersection(lønsum_b.index)
            .intersection(lønsum_10120.index)
        )

        bvt_forhold    = bvt_b.loc[tid]    / bvt_10120.loc[tid]
        lønsum_forhold = lønsum_b.loc[tid] / lønsum_10120.loc[tid]
        gamma = alpha * bvt_forhold + (1 - alpha) * lønsum_forhold

        for tidspunkt, g in gamma.items():
            rows.append({"ANVENDELSE": kort_kode, "TID": tidspunkt, "gamma": g})

    return pd.DataFrame(rows)


# ===========================================================================
# Opsplitning af kapital
# ===========================================================================

def opsplit_kapital(
    df_kapital: pd.DataFrame,
    df_gamma:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Splitter 10120-raekken i kapital-datasættet ud i tre undersektorer
    vha. gamma-vægtene fra beregn_gamma().

    10120-aggregatet erstattes af 100010, 100030 og anden_fode.
    01000 (landbrug) og REST forbliver uændrede.

    Parametre
    ----------
    df_kapital : data["kapital"]       fra load_all()
    df_gamma   : output fra beregn_gamma()

    Returnerer
    ----------
    DataFrame med samme kolonner som df_kapital
    """
    gamma_idx = df_gamma.set_index(["ANVENDELSE", "TID"])["gamma"]

    mask_10120 = df_kapital["BRANCHE"] == KAPITAL_10120
    kap_10120  = df_kapital.loc[mask_10120].copy()
    kap_rest   = df_kapital.loc[~mask_10120].copy()

    splits = []
    for kort_kode, fuldt_navn in GAMMA_BRANCHER.items():
        del_df = kap_10120.copy()
        del_df["BRANCHE"] = fuldt_navn

        for tid, grp_idx in del_df.groupby("TID").groups.items():
            if (kort_kode, tid) in gamma_idx.index:
                del_df.loc[grp_idx, "INDHOLD"] *= gamma_idx.loc[(kort_kode, tid)]
            else:
                del_df.loc[grp_idx, "INDHOLD"] = np.nan

        splits.append(del_df)

    return pd.concat([kap_rest] + splits, ignore_index=True)


# ===========================================================================
# Alt-i-en funktion
# ===========================================================================

def beregn_kapital_117(
    data:  dict,
    alpha: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """
    Beregner opsplittet kapital på 117-niveau samt afledt prisindeks og
    maengdeindeks.

    Parametre
    ----------
    data  : dict returneret af import_data.load_all()
    alpha : vægt på BVT-forholdet i gamma-beregningen (default 0.5)

    Returnerer
    ----------
    dict med nøglerne:
      "gamma"               – splitting-vægte pr. branche og aar
      "kapital_117"         – opsplittet kapital-DataFrame
      "prisindeks_kap_117"  – kædet prisindeks (Pt)
      "mængdeindeks_kap_117" – maengdeindeks (Xt)
    """
    print("Beregner gamma-vægte ...")
    gamma = beregn_gamma(data["input"], data["input_117"], alpha=alpha)

    print("Opsplitter kapital til 117-niveau ...")
    kapital_117 = opsplit_kapital(data["kapital"], gamma)

    print("Beregner prisindeks og maengdeindeks for kapital (117-niveau) ...")
    pt  = beregn_prisindeks_kapital(kapital_117)
    xt  = beregn_mængdeindeks_kapital(kapital_117)

    print("Færdig!")
    return {
        "gamma":                  gamma,
        "kapital_117":            kapital_117,
        "prisindeks_kap_117":     pt,
        "mængdeindeks_kap_117":  xt,
    }


if __name__ == "__main__":
    from import_data import load_all
    data    = load_all()
    results = beregn_kapital_117(data)
    for name, df in results.items():
        print(f"\n{'='*50}")
        print(f"  {name}: {df.shape[0]} rækker x {df.shape[1]} kolonner")
        print(df.head(3).to_string(index=False))