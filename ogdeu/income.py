# ogdeu/income.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from ogcore import utils

# feste Defaults
E_DEFAULT = 20
S_DEFAULT = 80
J_DEFAULT = 7

# kleine Numerik-Sicherheiten
_EPS   = 1e-12
_XTOL  = 1e-10
_ITERS = 60

# OG-USA-Koeffizienten (J=7)
const = np.array([3.41, 0.69689692, -0.78761958, -1.11, -0.93939272, 1.60, 1.89], dtype=float)
one   = np.array([-0.09720122, 0.05995294, 0.17654618, 0.21168263, 0.21638731, 0.04500235, 0.09229392], dtype=float)
two   = np.array([ 0.00247639,-0.00004086,-0.00240656,-0.00306555,-0.00321041, 0.00094253, 0.00012902], dtype=float)
three = np.array([-0.00001842,-0.00000521, 0.00001039, 0.00001438, 0.00001579,-0.00001470,-0.00001169], dtype=float)

# Altersfaktor φ_s (0..100) – dein Vektor
fctr = np.array(
    [*([0.0]*20),
     0.5741,0.6108,0.6289,0.6245,0.6270,0.6348,0.6356,0.6494,0.6586,0.6638,0.6664,0.6598,
     0.6484,0.6427,0.6316,0.6272,0.6246,0.6244,0.6222,0.6204,0.6181,0.6196,0.6172,0.6188,
     0.6200,0.6211,0.6163,0.6115,0.6078,0.5989,0.5967,0.5933,0.5905,0.5841,0.5714,0.5535,
     0.5334,0.5076,0.4760,0.4448,0.4073,0.3713,0.3230,0.2636,0.1917,0.1412,0.1002,0.0789,
     0.0678,0.0652,0.0646,0.0670,0.0630,0.0585,0.0541,0.0521,0.0538,0.0535,0.0577,0.0531,0.0472,
     *([0.0]*20)], dtype=float
)
assert fctr.shape[0] == 101, "fctr muss Länge 101 (Alter 0..100) haben."

def _validate_and_norm_weights(age_wgts: np.ndarray, lambdas: np.ndarray, S: int, J: int) -> Tuple[np.ndarray, np.ndarray]:
    age_w = np.asarray(age_wgts, float).reshape(-1)
    lam   = np.asarray(lambdas, float).reshape(-1)
    assert age_w.size == S, f"age_wgts/omega_SS Länge {age_w.size}, erwartet {S}"
    assert lam.size   == J, f"lambdas Länge {lam.size}, erwartet {J}"
    sw = age_w.sum(); sl = lam.sum()
    age_w = age_w / (sw if sw > _EPS else 1.0)
    lam   = lam   / (sl if sl > _EPS else 1.0)
    return age_w, lam

def lep_base_array(S: int = S_DEFAULT, J: int = J_DEFAULT,
                   age_start: int = 20, age_end: int = 80) -> np.ndarray:
    """
    Basis-LEP (S×J) aus OG-USA-Polynomen × deutschem Altersfaktor.
    Für S=80: Alter 21..80 (60 Zeilen) > 0, Alter 81..100 = 0.
    """
    assert J == 7, "Diese Version ist für J=7 ausgelegt."
    assert S >= (age_end - age_start), "S muss >= Anzahl aktiver Altersjahre (hier 60) sein."

    age_work = age_end - age_start  # = 60
    ages = np.arange(age_start+1, age_end+1).reshape(age_work, 1)  # 21..80 (60x1)
    ages = np.tile(ages, (1, J))                                   # 60x7

    log_paths = const + one*ages + two*(ages**2) + three*(ages**3) # 60x7
    abil_paths = np.exp(log_paths) * fctr[age_start+1:age_end+1].reshape(age_work, 1)

    e = np.zeros((S, J), dtype=float)
    e[:age_work, :] = np.maximum(abil_paths, _EPS)  # 21..80 aktiv
    # 81..100 = 0 (Ruhestand ohne Erwerb)
    return e

def normalize_mean_one(e: np.ndarray, age_wgts: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Skaliert so, dass Σ_{s,j} e_{s,j} ω_s λ_j = 1."""
    S, J = e.shape
    denom = float((e * age_wgts.reshape(S,1) * lambdas.reshape(1,J)).sum())
    return e / (denom if denom > _EPS else 1.0)

def make_e(age_wgts: np.ndarray, lambdas: np.ndarray,
           gini_target: float, S: int = S_DEFAULT, J: int = J_DEFAULT) -> np.ndarray:
    """
    Finale e-Matrix (S×J):
      1) e0 = lep_base_array()
      2) Normierung auf Mittelwert 1
      3) Power-Rescaling (p*) für Ziel-Gini
    """
    age_w, lam = _validate_and_norm_weights(age_wgts, lambdas, S, J)

    e0 = lep_base_array(S=S, J=J)
    e0 = normalize_mean_one(e0, age_w, lam)

    def gini_of_p(p: float) -> float:
        em = normalize_mean_one(np.maximum(e0, _EPS) ** float(p), age_w, lam)
        return utils.Inequality(em, age_w, lam, S, J).gini()

    g0 = gini_of_p(1.0)
    lo, hi = (0.05, 1.0) if gini_target < g0 else (1.0, 3.0)

    # robuster Bisect (mit minimaler Bracketerweiterung)
    f = lambda p: gini_of_p(p) - gini_target
    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo * fhi > 0 and tries < 20:
        if gini_target < g0:
            lo = max(lo/2.0, 0.01)
        else:
            hi *= 1.5
        flo, fhi = f(lo), f(hi)
        tries += 1

    a, b = lo, hi
    for _ in range(_ITERS):
        m = 0.5*(a+b)
        fm = f(m)
        if abs(fm) < _XTOL or (b-a) < _XTOL:
            p_star = m
            break
        if f(a)*fm <= 0:
            b = m
        else:
            a = m
    else:
        p_star = 0.5*(a+b)  # Fallback

    e_final = normalize_mean_one(np.maximum(e0, _EPS) ** p_star, age_w, lam)
    return e_final

def make_e_and_meta(age_wgts: np.ndarray, lambdas: np.ndarray,
                    gini_target: float, S: int = S_DEFAULT, J: int = J_DEFAULT):
    """Wie make_e, zusätzlich Rückgabe (p_star, gini_base)."""
    age_w, lam = _validate_and_norm_weights(age_wgts, lambdas, S, J)
    e0 = normalize_mean_one(lep_base_array(S=S, J=J), age_w, lam)

    def gini_of_p(p: float) -> float:
        em = normalize_mean_one(np.maximum(e0, _EPS) ** float(p), age_w, lam)
        return utils.Inequality(em, age_w, lam, S, J).gini()

    g0 = gini_of_p(1.0)
    lo, hi = (0.05, 1.0) if gini_target < g0 else (1.0, 3.0)
    f = lambda p: gini_of_p(p) - gini_target
    flo, fhi = f(lo), f(hi)
    tries = 0
    while flo * fhi > 0 and tries < 20:
        if gini_target < g0:
            lo = max(lo/2.0, 0.01)
        else:
            hi *= 1.5
        flo, fhi = f(lo), f(hi)
        tries += 1

    a, b = lo, hi
    for _ in range(_ITERS):
        m = 0.5*(a+b)
        fm = f(m)
        if abs(fm) < _XTOL or (b-a) < _XTOL:
            p_star = m
            break
        if f(a)*fm <= 0:
            b = m
        else:
            a = m
    else:
        p_star = 0.5*(a+b)

    e_final = normalize_mean_one(np.maximum(e0, _EPS) ** p_star, age_w, lam)
    return e_final, float(p_star), float(g0)

__all__ = ["lep_base_array", "normalize_mean_one", "make_e", "make_e_and_meta",
           "E_DEFAULT", "S_DEFAULT", "J_DEFAULT"]
