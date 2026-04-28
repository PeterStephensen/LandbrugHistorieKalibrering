import pandas as pd

t = "TID"
i = "ANVENDELSE"

TILGANG_BRUTTOOVERSKUD = "Bruttooverskud af produktion og blandet indkomst"
TILGANG_LONSUM = "Aflønning af ansatte"
TILGANG_ANDRESKATTER = "Andre produktionsskatter, netto"

TILGANG_KOMPONENTER = [TILGANG_BRUTTOOVERSKUD, TILGANG_LONSUM, TILGANG_ANDRESKATTER]

# Vægt på BVT-forholdet — resten (1 - W_BVT) går til lønsumsforholdet
alpha = 0.5

mapping_69 = {
    "01000 Landbrug og gartneri-(Tilgang)": "01000",
    "01000 Landbrug og gartneri- (Anvendelse)": "01000",
    "01000 Landbrug og gartneri": "01000",
    "10120 Føde-, drikke- og tobaksvareindustri-(Tilgang)": "10120",
    "10120 Føde-, drikke- og tobaksvareindustri- (Anvendelse)": "10120",
    "10120 Føde-, drikke- og tobaksvareindustri": "10120",
    "REST_TILGANG Øvrige brancher": "REST",
    "REST_ANVENDELSE Øvrige brancher": "REST",
}

mapping_117 = {
    "010000 Landbrug og gartneri-(Tilgang)": "010000",
    "010000 Landbrug og gartneri- (Anvendelse)": "010000",
    "100010 Slagterier-(Tilgang)": "100010",
    "100010 Slagterier- (Anvendelse)": "100010",
    "100030 Mejerier-(Tilgang)": "100030",
    "100030 Mejerier- (Anvendelse)": "100030",
    "100040x100050 Anden fødevareindustri (100040, 100050)": "100040x100050",
    "REST_TILGANG Øvrige brancher": "REST",
    "REST_ANVENDELSE Øvrige brancher": "REST",
}


def _fra_input(path: str, mapping: dict, tilgang_filter: list) -> pd.DataFrame:
    df_input = pd.read_csv(path)
    df_input[i] = df_input[i].replace(mapping)
    data = df_input[
        (df_input["TILGANG1"].isin(tilgang_filter))
        & (df_input["PRISENHED"] == "Løbende priser")
    ]
    return (
        data
        .groupby([i, t])["INDHOLD"]
        .sum()
        .fillna(0)
        .to_frame()
    )


def _load_alle(path: str, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    bvt = _fra_input(path, mapping, TILGANG_KOMPONENTER)
    lonsum = _fra_input(path, mapping, [TILGANG_LONSUM])
    return bvt, lonsum


bvt_69, lonsum_69 = _load_alle(
    "../Nationalregnskab/Data69/input_landbrugsdata.csv", mapping_69
)
bvt_117, lonsum_117 = _load_alle(
    "../Nationalregnskab_117/Data117/input_landbrugsdata.csv", mapping_117
)


def _series(df: pd.DataFrame, branche: str) -> pd.Series:
    serie = df.xs(branche, level=i)["INDHOLD"]
    serie.name = "INDHOLD"
    return serie


# Nævnere: 10120 fra 69-opdelingen
bvt_10120   = _series(bvt_69,   "10120")
lonsum_10120 = _series(lonsum_69, "10120")

# Tællere: de tre 117-brancher
gamma_brancher = ["100010", "100030", "100040x100050"]
gamma_dict = {}
for branche in gamma_brancher:
    bvt_b    = _series(bvt_117,   branche)
    lonsum_b = _series(lonsum_117, branche)

    tid = bvt_b.index.intersection(bvt_10120.index).intersection(lonsum_b.index).intersection(lonsum_10120.index)

    bvt_forhold    = bvt_b.loc[tid]    / bvt_10120.loc[tid]
    lonsum_forhold = lonsum_b.loc[tid] / lonsum_10120.loc[tid]

    gamma_dict[branche] = alpha * bvt_forhold + (1 - alpha) * lonsum_forhold

s = pd.concat(gamma_dict, names=[i, t]).sort_index()
s.name = "gamma"
s_df = s.to_frame()
s_df.to_csv("../Nationalregnskab_117/Data117/s.csv", index=True)