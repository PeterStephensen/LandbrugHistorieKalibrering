import pandas as pd

t = "TID"
i = "ANVENDELSE"

TILGANG_BRUTTOOVERSKUD = "Bruttooverskud af produktion og blandet indkomst"

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


def _bruttooverskud_fra_input(path: str, mapping: dict) -> pd.DataFrame:
    df_input = pd.read_csv(path)
    df_input[i] = df_input[i].replace(mapping)
    bruttooverskud_data = df_input[
        (df_input["TILGANG1"] == TILGANG_BRUTTOOVERSKUD)
        & (df_input["PRISENHED"] == "Løbende priser")
    ]
    bruttooverskud = bruttooverskud_data.pivot_table(
        index=[i, t], values="INDHOLD"
    ).fillna(0)
    bruttooverskud.index.names = [i, t]
    return bruttooverskud


bruttooverskud_69 = _bruttooverskud_fra_input(
    "../Nationalregnskab/Data69/input_landbrugsdata.csv", mapping_69
)
bruttooverskud_117 = _bruttooverskud_fra_input(
    "../Nationalregnskab_117/Data117/input_landbrugsdata.csv", mapping_117
)


def _bruttooverskud_series(bruttooverskud_df: pd.DataFrame, branche: str) -> pd.Series:
    serie = bruttooverskud_df.xs(branche, level=i)["INDHOLD"]
    serie.name = "INDHOLD"
    return serie


# Nævner i gamma: 10120 fra 69-opdelingen
bruttooverskud_10120 = _bruttooverskud_series(bruttooverskud_69, "10120")

# Tællere i gamma: de tre 117-brancher
gamma_brancher = ["100010", "100030", "100040x100050"]
gamma_dict = {}
for branche in gamma_brancher:
    bruttooverskud_branche = _bruttooverskud_series(bruttooverskud_117, branche)
    common_tid = bruttooverskud_branche.index.intersection(bruttooverskud_10120.index)
    gamma_dict[branche] = bruttooverskud_branche.loc[common_tid] / bruttooverskud_10120.loc[common_tid]

# Samme struktur som i calib: én samlet serie med MultiIndex (ANVENDELSE, TID)
gamma = pd.concat(gamma_dict, names=[i, t]).sort_index()
gamma.name = "gamma"
gamma_df = gamma.to_frame()
gamma_df.to_csv("../Nationalregnskab_117/Data117/gamma.csv", index=True)