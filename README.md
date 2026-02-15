# finance

## Rio TabNet mortality scraper

Use `scrape_tabnet_rio_mortality.py` to extract mortality counts from Rio TabNet (`sim_apos2005`) with:

- Row: `Bairro Residencia`
- Column: `Faixa Edade`
- Filters: `Munic Residencia = 330455 Rio de Janeiro`, `UF Res = RJ`
- All categories for neighborhood, age range, and cause (cause is materialized in a dedicated CSV column).

### 1) Generate the CSV

```bash
python scrape_tabnet_rio_mortality.py --output rio_mortality_by_neighborhood_cause_age.csv
```

### 2) Load the generated file ("arquivo")

#### Option A: Python + pandas

```python
import pandas as pd

path = "rio_mortality_by_neighborhood_cause_age.csv"
df = pd.read_csv(path, dtype={"neighborhood_code": "string"})

# Keep identifier columns and convert age-range columns to numeric
id_cols = ["neighborhood_code", "neighborhood", "cause"]
age_cols = [c for c in df.columns if c not in id_cols]
df[age_cols] = df[age_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("int64")

print(df.head())
print(df.shape)
```

#### Option B: Python stdlib only (no pandas)

```python
import csv

path = "rio_mortality_by_neighborhood_cause_age.csv"
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"rows={len(rows)}")
print(f"columns={reader.fieldnames}")
print(rows[0])
```

### Optional flags

- `--max-causes 5` for debug runs with fewer causes.
- `--sleep 0.2` pause between requests (default `0.2s`).
