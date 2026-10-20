
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "vehicles.csv"
OUTPUT_DIR = Path("eda_results")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, low_memory=False)

# Convertir les variables qui doivent être numériques
for col in ["price", "year", "odometer", "lat", "long"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n=== 1. VARIABLE CIBLE : PRICE ===")
price = df["price"].dropna()
print(price.describe(percentiles=[
    .01, .05, .25, .50, .75, .90, .95, .99, .999
]))
print("Prix <= 0 :", (df["price"] <= 0).sum())
print("Prix > 100 000 :", (df["price"] > 100_000).sum())
print("Prix > 1 000 000 :", (df["price"] > 1_000_000).sum())

# Histogrammes du prix brut et du prix positif en échelle log
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(price[price > 0], bins=100)
axes[0].set_title("Prix positifs - échelle originale")
axes[0].set_xlabel("Prix ($)")
axes[0].set_ylabel("Nombre d'annonces")

positive_price = price[price > 0]
axes[1].hist(np.log10(positive_price), bins=100)
axes[1].set_title("Log10 du prix positif")
axes[1].set_xlabel("Log10(prix)")
axes[1].set_ylabel("Nombre d'annonces")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_distributions.png", dpi=150)
plt.close()

print("\n=== 2. VARIABLES NUMERIQUES ===")
numeric_cols = ["year", "odometer", "lat", "long"]
print(df[numeric_cols].describe(
    percentiles=[.01, .05, .25, .5, .75, .95, .99]
).T)

for col in numeric_cols:
    values = df[col].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(values, bins=60)
    axes[0].set_title(f"Distribution : {col}")
    axes[1].boxplot(values, vert=False)
    axes[1].set_title(f"Boxplot : {col}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{col}_distribution.png", dpi=150)
    plt.close()

print("\n=== 3. VALEURS MANQUANTES ===")
missing = pd.DataFrame({
    "count": df.isna().sum(),
    "percent": (df.isna().mean() * 100).round(2)
}).sort_values("percent", ascending=False)
print(missing.to_string())
missing.to_csv(OUTPUT_DIR / "missing_values.csv")

# Prix selon la présence ou l'absence d'une information
for col in [
    "year", "manufacturer", "model", "condition", "cylinders",
    "fuel", "odometer", "title_status", "drive", "size", "type"
]:
    if col not in df.columns:
        continue
    tmp = df[["price", col]].copy()
    tmp["information_status"] = np.where(
        tmp[col].isna(), "Missing", "Present"
    )
    summary = tmp.groupby(
        "information_status", observed=False
    )["price"].agg(["count", "median", "mean"])
    print(f"\nPrix selon présence de {col} :")
    print(summary)

print("\n=== 4. VARIABLES CATEGORIELLES ===")
categorical_cols = [
    "manufacturer", "condition", "cylinders", "fuel",
    "transmission", "drive", "size", "type",
    "title_status", "paint_color", "state"
]

for col in categorical_cols:
    if col not in df.columns:
        continue

    print(f"\n--- {col} ---")
    print("Nombre de catégories :", df[col].nunique(dropna=True))
    print(df[col].value_counts(dropna=False).head(15))

    # Prix médian pour les catégories fréquentes
    tmp = df.loc[df[col].notna() & (df["price"] > 0),
                 [col, "price"]]
    frequent = tmp[col].value_counts().head(15).index
    summary = (
        tmp[tmp[col].isin(frequent)]
        .groupby(col, observed=True)["price"]
        .agg(["count", "median"])
        .sort_values("median", ascending=False)
    )
    print("Prix médian des catégories fréquentes :")
    print(summary)

print("\n=== 5. DOUBLONS DE VIN ===")
if "VIN" in df.columns:
    vin = df["VIN"].dropna().astype(str).str.strip().str.upper()
    print("VIN non manquants :", len(vin))
    print("VIN uniques :", vin.nunique())
    print("Lignes dont le VIN apparaît plusieurs fois :",
          vin.duplicated(keep=False).sum())
    print("Occurrences répétées après la première :",
          vin.duplicated().sum())

    vin_counts = vin.value_counts()
    print("VIN les plus fréquents :")
    print(vin_counts.head(20))

print("\n=== 6. RELATIONS AVEC LE PRIX ===")
valid_price = df["price"] > 0
num_for_corr = ["price", "year", "odometer"]
corr_df = df.loc[valid_price, num_for_corr].copy()

print("\nCorrélation de Pearson :")
print(corr_df.corr(method="pearson"))

print("\nCorrélation de Spearman :")
print(corr_df.corr(method="spearman"))

for col in ["year", "odometer"]:
    tmp = df.loc[
        valid_price & df[col].notna(), [col, "price"]
    ].copy()
    # Limiter uniquement l'affichage graphique, pas les données source
    if len(tmp) > 100_000:
        tmp = tmp.sample(100_000, random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(tmp[col], tmp["price"], alpha=0.15, s=8)
    plt.title(f"{col} vs price (prix positifs)")
    plt.xlabel(col)
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{col}_vs_price.png", dpi=150)
    plt.close()

print("\nEDA terminée. Graphiques et tableaux dans :", OUTPUT_DIR)
