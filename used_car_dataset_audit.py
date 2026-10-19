import pandas as pd
import numpy as np

CSV_PATH = "vehicles.csv"

df = pd.read_csv(CSV_PATH, low_memory=False)

print("=" * 70)
print("1. DATASET SIZE")
print("=" * 70)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:,.1f} MB")

print("\n" + "=" * 70)
print("2. COLUMN TYPES AND NON-MISSING VALUES")
print("=" * 70)
df.info()

print("\n" + "=" * 70)
print("3. MISSING VALUES")
print("=" * 70)
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
}).sort_values("missing_percent", ascending=False)
print(missing[missing["missing_count"] > 0].to_string())

print("\n" + "=" * 70)
print("4. EXACT DUPLICATE ROWS")
print("=" * 70)
print("Exact duplicate rows:", f"{df.duplicated().sum():,}")

# Check likely listing identifiers, if present.
for col in ["id", "url", "VIN"]:
    if col in df.columns:
        nonempty = df[col].dropna()
        print(f"Duplicate non-null {col} values:", f"{nonempty.duplicated().sum():,}")

print("\n" + "=" * 70)
print("5. NUMERIC SUMMARY AND DISTRIBUTIONS")
print("=" * 70)
numeric = df.select_dtypes(include=np.number)
if not numeric.empty:
    print(numeric.describe(percentiles=[.01, .25, .5, .75, .95, .99]).T.to_string())

print("\n" + "=" * 70)
print("6. UNIQUE VALUES IN CATEGORICAL COLUMNS")
print("=" * 70)
for col in df.select_dtypes(include=["object", "category"]).columns:
    print(f"\n{col}: {df[col].nunique(dropna=True):,} unique non-missing values")
    print(df[col].value_counts(dropna=False).head(8).to_string())

print("\n" + "=" * 70)
print("7. POTENTIALLY INVALID VALUES")
print("=" * 70)

# These are screening rules, not automatic proof that a record is wrong.
if "price" in df.columns:
    price = pd.to_numeric(df["price"], errors="coerce")
    print("Price <= 0:", f"{(price <= 0).sum():,}")
    print("Price missing/non-numeric:", f"{price.isna().sum():,}")
    print("Price > $1,000,000:", f"{(price > 1_000_000).sum():,}")

if "year" in df.columns:
    year = pd.to_numeric(df["year"], errors="coerce")
    print("Year missing/non-numeric:", f"{year.isna().sum():,}")
    print("Year before 1900:", f"{((year < 1900) & year.notna()).sum():,}")
    print("Year after 2026:", f"{((year > 2026) & year.notna()).sum():,}")

if "odometer" in df.columns:
    odo = pd.to_numeric(df["odometer"], errors="coerce")
    print("Odometer < 0:", f"{((odo < 0) & odo.notna()).sum():,}")
    print("Odometer > 1,000,000:", f"{((odo > 1_000_000) & odo.notna()).sum():,}")
    print("Odometer missing/non-numeric:", f"{odo.isna().sum():,}")

for col in ["lat", "long"]:
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        print(f"{col}: missing/non-numeric = {vals.isna().sum():,}")

if "lat" in df.columns:
    lat = pd.to_numeric(df["lat"], errors="coerce")
    print("Latitude outside [-90, 90]:", f"{((~lat.between(-90, 90)) & lat.notna()).sum():,}")
if "long" in df.columns:
    lon = pd.to_numeric(df["long"], errors="coerce")
    print("Longitude outside [-180, 180]:", f"{((~lon.between(-180, 180)) & lon.notna()).sum():,}")

print("\n" + "=" * 70)
print("8. EXTREME NUMERIC VALUES (IQR SCREEN)")
print("=" * 70)
for col in numeric.columns:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 4:
        continue
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        continue
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = ((s < lower) | (s > upper)).sum()
    print(f"{col}: {count:,} values outside [{lower:,.2f}, {upper:,.2f}] ({count / len(s) * 100:.2f}%)")

print("\n" + "=" * 70)
print("9. POSSIBLE TEXT/CATEGORY INCONSISTENCIES")
print("=" * 70)
for col in ["manufacturer", "condition", "fuel", "transmission", "drive",
            "type", "title_status", "paint_color", "state"]:
    if col in df.columns:
        s = df[col].dropna().astype(str)
        print(f"\n{col}:")
        print("Distinct values after trimming whitespace:", s.str.strip().nunique())
        print("Values with leading/trailing spaces:", f"{(s != s.str.strip()).sum():,}")
        print("Common values with inconsistent letter case (sample):")
        print(s[s.str.lower().duplicated(keep=False)].value_counts().head(8).to_string())

print("\n" + "=" * 70)
print("10. SAMPLE RECORDS TO MANUALLY INSPECT")
print("=" * 70)
cols = [c for c in ["price", "year", "manufacturer", "model", "condition",
                    "odometer", "fuel", "transmission", "state", "posting_date"]
        if c in df.columns]
print(df[cols].sample(min(10, len(df)), random_state=42).to_string(index=False))

print("\n" + "=" * 70)
print("AUDIT FINISHED — no data was changed.")
print("=" * 70)
