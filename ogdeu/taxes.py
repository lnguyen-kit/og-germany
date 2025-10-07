# get_micro_data_DE_from_CPS.py
import os, pickle
import numpy as np
import pandas as pd
from dask import delayed, compute
import dask.multiprocessing
import pkg_resources # um die taxcalc_version auszulesen 



try:
    import taxcalc as tc  # nur CPS laden --> wir nutzen nur tc.Records.cps_constructor() zum Laden der CPS-Daten
except Exception as e:
    raise RuntimeError("Bitte  taxcalc installieren. Grund: " + str(e))

try:
    from ogcore import utils  # mkdirs wie OG-IND
except Exception:
    class _U:
        @staticmethod
        def mkdirs(p): os.makedirs(p, exist_ok=True)
    utils = _U()

DEFAULT_START_YEAR = 2025 #brauche ich das wirklich ? 
CUR_PATH = os.getcwd()


# Signaturgleichheit zu OG-IND
def get_calculator(
    baseline: bool, # Gibt an, ob die Berechnung mit der aktuellen Gesetzeslage ("baseline") durchgeführt wird
    calculator_start_year: int, # Startjahr für die Steuerberechnung
    reform: dict | None = None, # Änderungen am Steuergesetz, die berücksichtigt werden sollen, wenn none dann wird Baseline verwendet
    data=None, gfactors=None, weights=None, #  Steuerdaten als DataFrame oder Dateipfad, Wachstumsfaktoren, um Daten über die Zeit zu extrapolieren,  Gewichte für die Datensätze
    records_start_year: int = DEFAULT_START_YEAR, #  Startjahr für die Datensätze (Standard ist DEFAULT_START_YEAR)
):
    # (Optionales Logging)
    if baseline:
        print("Running current law policy baseline" if not reform else f"Baseline policy is: {reform}")
    else:
        print("Running with current law as reform" if not reform else f"Reform policy is: {reform}")

    
    policy = dict(
        # PIT via §32a (Grundtarif) – implementiert in est_grundtarif_2025()
        #
        #-----------------------------------------------------------------------
        #in deiner Funktion est_grundtarif_2025(zve, splitting):
        #splitting=False → normaler Grundtarif (§32a EStG) auf das zu versteuernde Einkommen (bei dir: pit_base).
        #splitting=True → Einkommen halbieren, Tarif anwenden, verdoppeln (das ist das Splittingverfahren).
        # ---------------------------------------------------------------------
        # #Schaltet Ehegattensplitting ein/aus
        splitting =True,
        cap_rate=0.25,           #    # Kapital, cap flat 
        cap_allow=1000.0,           # z.B. 1000/2000 – minimal: 0

        # Numerik
        #er kleine Schritt für die Finite-Difference-Ableitungen (MTRx/MTRy).
        # MTRx ≈ (T(x+ε) − T(x)) / ε
        # MTRy ≈ (T(y+ε) − T(y)) / ε
        eps=1.0, 
        # Drifts, Drifts (Einkommenswachstum über Jahre)
        #Jährliche Wachstumsraten der CPS-Einkommen im Zeitverlauf 
        drift_w=0.02,
        drift_k=0.03,
        # Budgetfenster
        #Was es ist: Anzahl der Jahre, die du erzeugst
        T=1,
        # Skalierung
        # Was es ist: Ein Multiplikator auf alle Einkommen (Arbeit & Kapital), vor der Steuerberechnung
        # Einheiten anpassen (z. B. CPS in USD → Tarif in EUR ≈ 0.9–1.0 als grober Umrechnungsfaktor)
        scale_income=0.95,
        # --- ETR-Steuerung (NEU) ---
        # "pos": Nenner = nur positive Einkommen (lab^+ + cap^+)
        # "market": Nenner = lab + cap
        etr_den_mode="pos",
        # Floor gegen Mini-Nenner in der ETR-Quote
        etr_market_floor=100.0,

        
    )

    if reform:
        policy.update(reform)
    policy["start_year"] = int(calculator_start_year)
    return policy # vorher calc1: gibt ein Calculator-Objekt zurück, das für das gewünschte Startjahr konfiguriert ist



'''
Helferfunktion für das Quantile Mapping
'''

def _quantile_map_wage(w, weight, ps, q_de, meanDE=None, positive_only=True):
    w = np.asarray(w, float).copy(); g = np.asarray(weight, float) # sicherstellen dass w und weight float array sind, Copierne vom Orginal 
    sel = (w > 0) if positive_only else np.isfinite(w) # nur Beschäftigte Mappen deren Lohn größer 0 ist , wenn positive_only =True
    if sel.sum() == 0: return w # falls es keinen Fall zum Mappen gibt, dann das Origianl zurückgeben

    '''
    -----------------------
    np.argsort(w[sel]) liefert Indizies/Psoition, mit denne man w(sel) aufsteigend sortieren kann ,Bespiel: w[sel] = [32000, 150000, 47000] → argsort gibt [0, 2, 1]
    mit diesen Indizies sortieren wir die Löhne aufsteigend und die dazugehörigen gewichte --> Lohn und gewicht jeder person paarweise ausgerichtet 
    
    sel = (w > 0)                # nur Beschäftigte
    s   = np.argsort(w[sel])     # Sortierindizes nach Lohn
    xs  = w[sel][s]              # sortierte Löhne
    gs  = g[sel][s]              # dazu passende, sortierte Gewichte, also zu den sortierten Löhnen passenden Gewichte
    cdf = np.cumsum(gs) / gs.sum()

    
    ------------------------
    '''
    s = np.argsort(w[sel]); xs = w[sel][s]; gs = g[sel][s] 
    # CDF /kumulierte Verteilungsfunktion bei Lohnen ist F(x)=P(Lohn≤x) 
    cdf = np.cumsum(gs) / gs.sum()
    p   = np.interp(w[sel], xs, cdf, left=0.0, right=1.0)

    ps = np.asarray(ps, float); q_de = np.asarray(q_de, float)
    w_sel = np.interp(p, ps, q_de, left=q_de[0], right=q_de[-1])

    if (meanDE is not None) and np.isfinite(meanDE):
        cur = np.average(w_sel, weights=gs)
        if cur > 0: w_sel *= (float(meanDE) / cur)

    out = w.copy(); out[sel] = w_sel
    out[~np.isfinite(out)] = 0.0; out[out < 0] = 0.0
    return out



def _wquantile(x, w, ps):
    x = np.asarray(x, float); w = np.asarray(w, float); ps = np.asarray(ps, float)
    s = np.argsort(x); xs, ws = x[s], w[s]
    cdf = np.cumsum(ws) / ws.sum()
    return np.interp(ps, cdf, xs)


def scale_to_weighted_mean(x, w, target_mean):
    x = np.asarray(x, float); w = np.asarray(w, float)
    cur = np.average(x, weights=w)
    if cur == 0:
        return x  # oder: raise ValueError("aktuelles Mittel = 0")
    return x * (float(target_mean) / cur)


# ---------- CPS laden ----------

def _load_cps(apply_qmap: bool=False, targets: dict | None=None,
              component_targets: dict|None=None):
    recs = tc.Records.cps_constructor()
    age   = np.asarray(getattr(recs, "age_head"), dtype=int)
    wage  = np.asarray(getattr(recs, "e00200"),   dtype=float)   # Lohn/Gehalt
    seinc = np.asarray(getattr(recs, "e00900"),   dtype=float)   # Selbständig
    farm = np.asarray(getattr(recs, "e02100"), float)

    cg    = np.asarray(getattr(recs, "e01100"), float) #capital gain distribution
    intr  = np.asarray(getattr(recs, "e00300"),   dtype=float)  #Zinsen 
    div   = np.asarray(getattr(recs, "e00600"),   dtype=float)  # Dividenden
    wgt   = np.asarray(recs.s006,                 dtype=float)

    mars  = np.asarray(getattr(recs, "MARS"),     dtype=int)
    married = (mars == 2)

    # --- optional: Löhne "deutsch" machen (Quantile-Mapping) ---
    if apply_qmap:
        assert targets is not None and all(k in targets for k in ("ps","q_de"))
        wage_mapped = _quantile_map_wage(
            wage, wgt, ps=targets["ps"], q_de=targets["q_de"], meanDE=targets.get("meanDE")
        )
    else:
        wage_mapped = wage
    
    # --- Nur Mittelwerte treffen (gewichtete Destatis-Durchschnitte) ---
    # Erwartet: component_targets = {"seinc": 45469.0, "farm": 17002.0, "cap": 6137.0}
    if component_targets:
        if "seinc" in component_targets:
            seinc = scale_to_weighted_mean(seinc, wgt, component_targets["seinc"])
        if "farm" in component_targets:
            farm  = scale_to_weighted_mean(farm,  wgt, component_targets["farm"])
        # Kapital = Zinsen+Dividenden+real. Kursgewinne
        cap_raw = intr + div + cg
        if "cap" in component_targets:
            cap_raw = scale_to_weighted_mean(cap_raw, wgt, component_targets["cap"])
    else:
        cap_raw = intr + div + cg

    

    # --- Achsen sauber trennen ---
    payroll_base = wage_mapped              # SV-Bemessung NUR auf Lohn, also nur Sozailbeträge basieren auf Lohn/wage und nicht auf Gesamtarbeiseinkommen
    lab          = wage_mapped + seinc      # PIT-/Arbeits-Achse 
    cap          = cap_raw   # Kapital-Achse INKL. Miete

    return dict(
        age=age, weight=wgt, married=married,
        wage=wage_mapped, wage_raw=wage, seinc=seinc,
        intr=intr, div=div, cg=cg, farm=farm,
        payroll_base=payroll_base, lab=lab, cap=cap
    )

# ---------- §32a Grundtarif 2025 ----------
def est_grundtarif_2025(zve, splitting=False, rounded =True):

    # zve wird (vektorisiert) zu float gemacht und auf volle Euro abgerundet (gesetzlich: Abrundung auf volle Euro bei Bemessungsgrundlage und Ergebnis).
    # rudnen mit Floor()

    
    arr = np.asarray(zve, dtype=float)
    # CHANGE: Basis nur runden, wenn rounded=True
    x = np.floor(arr) if rounded else arr
   

    # für jedes EInkommmen x_ erhält man den jeweiligen Tarifbetrag gemäß § 32a für 2025
    def grundtarif(x_):
        # §32a EStG 2025 – Parameter
        # y: Zehntausendstel des über 12.096 € liegenden Teils
        # z: Zehntausendstel des über 17.443 € liegenden Teils

        # Zone 1: bis 12.096 € -> Steuer 0 (durch zeros_like abgedeckt)
        y = (x_ - 12096.0) / 10000.0
        z = (x_ - 17443.0) / 10000.0

        
        tax = np.zeros_like(x_)
        
        # Zone 2: 12.097 – 17.443 €
        m2 = (x_ >= 12097.0) & (x_ <= 17443.0)
        tax[m2] = (932.30 * y[m2] + 1400.0) * y[m2]
        

        # Zone 3: 17.444 – 68.480 €
        m3 = (x_ >= 17444.0) & (x_ <= 68480.0)
        tax[m3] = (176.64 * z[m3] + 2397.0) * z[m3] + 1015.13
        
    
        # Zone 4: 68.481 – 277.825 €
        m4 = (x_ >= 68481.0) & (x_ <= 277825.0)
        tax[m4] = 0.42 * x_[m4] - 10911.92

        # Zone 5: ab 277.826 €
        m5 = (x_ >= 277826.0)
        tax[m5] = 0.45 * x_[m5] - 19246.67
        
        # CHANGE: Ergebnis nur runden, wenn rounded=True
        return np.floor(tax) if rounded else tax

    if splitting:
        # CHANGE: beim Splitting Basis ggf. ungerundet halbieren
        return 2.0 * grundtarif(np.floor(x / 2.0) if rounded else (x / 2.0))
    else:
        return grundtarif(x)
    




# --- Sozialabgaben 2025: Deutschland gesamt (ohne Kinder/Sachsen) ---
# Diese Funktion bildet die AN-Beiträge mit zwei echten Deckeln ab:
#  - KV/PV bis BBG_KV_PV
#  - RV/ALV bis BBG_RV_ALV
# Sie liefert (Beitrag, MTR-Komponente) zurück.
def payroll_de_2025(
    lab_income,
    *,
    # Beitragsbemessungsgrenzen (Jahreswerte 2025)
    BBG_KV_PV: float = 66_150.0,   # KV/PV
    BBG_RV_ALV: float = 96_600.0,  # RV/ALV

    # Arbeitnehmer-Anteile (Deutschland gesamt; ohne Kinderzuschlag/Region)
    kv_rate_employee: float = 0.073,     # 7,3 % KV (AN-Hälfte)
    kv_add_rate_employee: float = 0.0125,# Ø hälftiger Zusatzbeitrag 2025 ≈ 1,25 %
    pv_rate_employee: float = 0.018,     # 1,8 % PV (AN-Anteil; ohne Kinderzuschlag)
    rv_rate_employee: float = 0.093,     # 9,3 % RV (AN)
    alv_rate_employee: float = 0.013     # 1,3 % ALV (AN)
):
    lab = np.asarray(lab_income, dtype=float)

    # Sozialbeiträge nur auf NICHT-negative Löhne erheben
    lab_pos = np.maximum(lab, 0.0)

    # KV-AN gesamt = 7,3 % + hälftiger Zusatzbeitrag
    rate_kv_total = kv_rate_employee + float(kv_add_rate_employee)

    # Bemessungsgrundlagen
    # Bemessungsgrundlagen mit Clip statt min(lab,...), damit nie negativ
    base1 = np.clip(lab_pos, 0.0, BBG_KV_PV)  # bis KV/PV-Deckel
    base2 = np.clip(lab_pos, 0.0, BBG_RV_ALV)  # zwischen den Deckeln

    contrib = (rate_kv_total + pv_rate_employee) * base1 \
            + (rv_rate_employee + alv_rate_employee) * base2


    # Grenzbeitrag (MTR der Sozialbeiträge)
    #  - bis 66.150 €: KV+PV+RV+ALV
    #  - 66.150–96.600 €: nur RV+ALV
    #  - > 96.600 €: 0
    
    mtr = np.where(lab_pos <= 0.0, 0.0, np.where(lab_pos < BBG_KV_PV,
                 (rate_kv_total + pv_rate_employee) + (rv_rate_employee + alv_rate_employee),
                 np.where(lab_pos < BBG_RV_ALV,
                          (rv_rate_employee + alv_rate_employee),
                          0.0))
    )
    
    return contrib, mtr



# ---------- Steuerrechnung ----------
def apply_de_tax(lab, cap, P, split_mask=None, payroll_base=None):
    eps = float(P.get("eps", 1.0))
    split = np.zeros_like(lab, dtype=bool) if split_mask is None else np.asarray(split_mask, dtype=bool)

    def _etr(T, lab, cap, P):
        mode_den = str(P.get("etr_den_mode", "pos"))
        den_floor = float(P.get("etr_market_floor", 100.0))
        den = (np.clip(lab, 0.0, None) + np.clip(cap, 0.0, None)) if mode_den=="pos" else (lab + cap)
        etr = np.zeros_like(T, dtype=float)
        m = den > den_floor
        etr[m] = T[m]/den[m]
        return np.clip(etr, 0.0, 2.0)

    def pit_from_base(base, rounded):
        pit = np.empty_like(base, dtype=float)
        pit[split]  = 2.0 * est_grundtarif_2025(0.5 * base[split], rounded=rounded)
        pit[~split] = est_grundtarif_2025(      base[~split], rounded=rounded)
        return pit

    def payroll_from_wage(w_vec):
        contrib, mtr = payroll_de_2025(
            w_vec,
            BBG_KV_PV=float(P.get("BBG_KV_PV", 66_150.0)),
            BBG_RV_ALV=float(P.get("BBG_RV_ALV", 96_600.0)),
            kv_rate_employee=float(P.get("kv_rate_employee", 0.073)),
            kv_add_rate_employee=float(P.get("kv_add_rate_employee", 0.0125)),
            pv_rate_employee=float(P.get("pv_rate_employee", 0.018)),
            rv_rate_employee=float(P.get("rv_rate_employee", 0.093)),
            alv_rate_employee=float(P.get("alv_rate_employee", 0.013)),
        )
        return contrib, mtr

    cap_rate  = float(P.get("cap_rate", 0.25))
    cap_allow = float(P.get("cap_allow", 0.0))
    def cap_tax_from_cap(cap_vec): return cap_rate * np.clip(cap_vec - cap_allow, 0.0, None)

    # --- Level-Steuern (PIT gerundet) ---
    pit = pit_from_base(lab, rounded=True)
    wage_for_sv = payroll_base if payroll_base is not None else lab  # Fallback
    payroll, mtr_payroll = payroll_from_wage(wage_for_sv)
    cap_tax = cap_tax_from_cap(cap)
    T = pit + payroll + cap_tax
    etr = _etr(T, lab, cap, P)

    # --- Marginals ungerundet ---
    T0x = pit_from_base(lab, rounded=False) + payroll_from_wage(wage_for_sv)[0] + cap_tax_from_cap(cap)
    T1x = pit_from_base(lab + eps, rounded=False) + payroll_from_wage(wage_for_sv + eps)[0] + cap_tax_from_cap(cap)
    mtrx = np.clip((T1x - T0x) / eps, 0.0, 0.99)

    T0y = T0x
    T1y = pit_from_base(lab, rounded=False) + payroll_from_wage(wage_for_sv)[0] + cap_tax_from_cap(cap + eps)
    mtry = np.clip((T1y - T0y) / eps, 0.0, 0.99)

    return T, payroll, etr, mtrx, mtry


def cps_de_advance(policy: dict, year: int, base):
    start_year = int(policy["start_year"])
    t = year - start_year #start_year wurde in get_calculator(...) gesetzt (z. B. 2022). t ist, wie viele Jahre du vorangeschritten bist.
    if t < 0: raise ValueError("year < start_year")

    #Einkommen auf Zieljahr hochskalieren: So bekommst du für jedes Jahr ein plausibel „fortgeschriebenes“ Einkommenspanel, ohne die Mikrodaten neu zu gewichten.
    # OG ind ia hat es mit Gorth factors gemacht , vielleicht asuch so machen weil CPs auch growth facotr mit sich bringt 
    lab = base["lab"] * ((1.0 + float(policy.get("drift_w", 0.02))) ** t) * float(policy.get("scale_income", 1.0)) # Arbeitseinkommen wächst jährlich mit drift_w (Standard 2 %), also (1,02)^t 
    cap = base["cap"] * ((1.0 + float(policy.get("drift_k", 0.03))) ** t) * float(policy.get("scale_income", 1.0)) # Kapitaleinkommen wächst um 3 prozent , scale_income skaliert beide zusätzlich (z. B. USD→EUR-Umrechnung oder Level-Kalibrierung)

    T, payroll, etr, mtrx, mtry = apply_de_tax(lab, cap, policy, split_mask=base["married"], payroll_base=base.get("payroll_base", None)) # apply_de_tax(lab, cap, policy ist mein Steur Engine , also rechnet trafilce Est auf einer passendne Besmmsungsgrundlage und leifert T, pyroll, et und mtrx und mtry

    N = lab.shape[0] # N ist die Länge der Arrays – wichtig, um unten Vektoren der richtigen Länge zu bauen (z. B. die year-Spalte).
    
    # Das ist ein Dictionary von Arrays (je eine Spalte) mit exakt den 11 Spalten, die OG-Core/txfunc erwartet
    # In get_data(...) wird dieses Dict je Jahr in einen DataFrame gegossen und dann als {str(year): df} zusammengebaut – genau wie OG-IND
    return dict(
        mtr_labinc=mtrx, mtr_capinc=mtry, age=base["age"],
        total_labinc=lab, total_capinc=cap, market_income=lab + cap,
        total_tax_liab=T, payroll_tax_liab=payroll, etr=etr,
        year=np.full(N, year, dtype=int), weight=base["weight"]
    )


def get_data(
    baseline: bool = False,
    start_year: int = DEFAULT_START_YEAR,
    reform: dict | None = None,
    data=None, path: str = CUR_PATH,
    client=None, num_workers: int = 1,
):


    policy = get_calculator(
        baseline=baseline,
        calculator_start_year=start_year,
        reform=(reform or {}),
        data=data,
    )


    # Destatis-Zahlen  als Zielwerte
    DE_TARGETS = {
    "ps":   np.array([0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.99], dtype=float),
    "q_de": np.array([32526,37944,42700,47244,52159,58214,65843,77105,97680,213286], dtype=float),
    #"meanDE": 62235.0,  # optional, nicht relevant 
}
    # Sicherheitschecks:
    assert len(DE_TARGETS["ps"]) == len(DE_TARGETS["q_de"])
    assert np.all(np.diff(DE_TARGETS["ps"]) > 0)


    # Wichtig: Wenn du in Euro mappst, setz scale_income=1.0,
    # sonst würdest du den frisch getroffenen Mean wieder verschieben.
    policy["scale_income"] = 1.0

    DE_COMP_TARGETS = {
        "seinc": 45469.0,  # selbständige Arbeit
        "farm":   17002.0,  # Landwirtschaft
        "cap":    6137.0,  # Kapitalvermögen (Zins+Div+CG)
    }

    base = _load_cps(apply_qmap=True, targets=DE_TARGETS,
                     component_targets=DE_COMP_TARGETS)


 #   diagnose_wage_vs_targets(base, DE_TARGETS, atol=100.0)     # trifft P10..P99? Mean?
  #  diagnose_before_after(base, DE_TARGETS)                   # optional: Vorher/Nachher


    # 3) Fenster & Obergrenze wie OG-IND
    T = int(policy.get("T", 1))
    TC_LAST_YEAR = int(policy.get("max_year", start_year + T - 1))  # optional wie OG-IND
    if start_year > TC_LAST_YEAR:
        raise RuntimeError("Start year is beyond configured last year.")

    year_stop = min(TC_LAST_YEAR, start_year + T - 1)
    years = list(range(start_year, year_stop + 1))

    # 4) Dask: lazy tasks bauen
    lazy_values = [delayed(cps_de_advance)(policy, yr, base) for yr in years]

    # 5) Rechnen: mit client oder mit multiprocessing.get (OG-IND-Style)
    if client:
        futures = client.compute(lazy_values, num_workers=num_workers)
        results = client.gather(futures)
    else:
        results = compute(
            *lazy_values,
            scheduler=dask.multiprocessing.get,
            num_workers=num_workers,
        )

    micro_data_dict = {str(yr): pd.DataFrame(res) for yr, res in zip(years, results)}

    # Pickel Seichern, ordner anlegen , dateinnamen --> JAhresdic aus Dataframes
    utils.mkdirs(path)
    pkl_path = os.path.join(path, "micro_data_baseline.pkl" if baseline else "micro_data_policy.pkl")
    with open(pkl_path, "wb") as f: pickle.dump(micro_data_dict, f)

    taxcalc_version = pkg_resources.get_distribution("taxcalc").version

    return micro_data_dict, f"CPS-DE-v1 (tc {taxcalc_version})"

