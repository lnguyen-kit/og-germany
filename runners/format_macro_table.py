import pandas as pd

# 1) CSV einlesen (dein Screenshot hat vorne die Index-Spalte mit 0,1,2,...)
df = pd.read_csv("ogdeu_example_output.csv", index_col=0)

# 2) numerische Spalten (alle außer "Variable")
num_cols = df.columns[1:]   # ab "2025" bis "SS"

# von 0.011 -> 1.11 (Prozent) und runden
df[num_cols] = (df[num_cols] * 100).round(2)

# 3) Spaltennamen anpassen wie im Paper
cols = df.columns.tolist()
cols[-2] = "Avg. 10-yr"     # statt "2025-2034"
cols[-1] = "Steady state"   # statt "SS"
df.columns = cols

# 4) neue CSV mit den „schönen“ Werten
df.to_csv("ogdeu_macro_table_pretty.csv", index=False)
print("Fertige Tabelle in ogdeu_macro_table_pretty.csv gespeichert.")
