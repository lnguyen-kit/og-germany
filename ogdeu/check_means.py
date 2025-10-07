# check_means.py
import numpy as np
from taxes import _load_cps

DE_TARGETS = {
    "ps":   np.array([0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.99], dtype=float),
    "q_de": np.array([32526,37944,42700,47244,52159,58214,65843,77105,97680,213286], dtype=float),
    # "meanDE": 62235.0,  # optional
}
DE_COMP_TARGETS = {
    "seinc": 45469.0,  # Selbständigkeit
    "rent":   7444.0,  # Vermietung/Verpachtung
    "cap":    6137.0,  # Kapital (Zins+Div+CG)
}

base = _load_cps(apply_qmap=True, targets=DE_TARGETS, component_targets=DE_COMP_TARGETS)
w = base["weight"]

wmean = lambda x: float(np.average(np.asarray(x, float), weights=w))
posneg = lambda x: (int((np.asarray(x)>0).sum()), int((np.asarray(x)<0).sum()))

print("== Gew. Mittelwerte ==")
print(f"wage : {wmean(base['wage']):,.2f}   pos/neg={posneg(base['wage'])}")
print(f"seinc: {wmean(base['seinc']):,.2f}   pos/neg={posneg(base['seinc'])}")
print(f"rent : {wmean(base['rent']):,.2f}    pos/neg={posneg(base['rent'])}")
print(f"cap  : {wmean(base['cap']):,.2f}     pos/neg={posneg(base['cap'])}")
