# runners/run_baseline.py
from ogcore.parameters import Specifications
from ogcore.SS import run_SS
import os, pickle, traceback
import numpy as np
from ogdeu.demographics import omega_ss_germany_2024
from ogdeu.income import make_e
from ogdeu.income import make_e_and_meta



def main():
    print(">> Building specifications…")
    p = Specifications()

    p.E = 20
    p.S = 80
    p.J = 7

    # --- Workaround: erwarteten Pickle-Pfad einmal anlegen ---
    # (deine OG-Core-Version sucht hier: OUTPUT_BASELINE/SS/SS_vars.pkl)
    legacy_base = "OUTPUT_BASELINE"
    os.makedirs(os.path.join(legacy_base, "SS"), exist_ok=True)
    ss_pickle = os.path.join(legacy_base, "SS", "SS_vars.pkl")
    if not os.path.exists(ss_pickle):
        with open(ss_pickle, "wb") as f:
            pickle.dump({}, f)   # leerer Dict reicht zum Start

    # --- Korrekt geformte Overrides (gemäß deinem Schema-Dump) ---
    
    overrides = {
    "start_year": 2020,
    "delta_annual": [{"value": 0.06}],
    "g_y_annual":  [{"value": 0.015}],
    "frisch":      [{"value": 0.4}],
    "debt_ratio_ss": [{"value": 0.60}],
    "Z":     [{"value": [[1.0]]}],
    "tau_c": [{"value": [[0.19]]}],
}


    print(">> Applying overrides…")
    p.update_specifications(overrides)

    # (optional) eigenes Output-Verzeichnis zusätzlich erstellen
    outdir = "outputs_deu"
    os.makedirs(os.path.join(outdir, "SS"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "TPI"), exist_ok=True)
    # Manche Versionen beachten 'output_base' als Attribut:
    try:
        p.output_base = outdir
    except Exception:
        pass

    print(">> Solving steady state (SS)…")


    lam = np.array([0.25, 0.25, 0.20, 0.10, 0.10, 0.09, 0.01], dtype=float)
    omega = omega_ss_germany_2024()

    p.omega_SS = omega
    p.lambdas  = lam

    e, p_star, g0 = make_e_and_meta(omega, lam, gini_target=0.311, S=p.S, J=lam.size)
    print(f"[LEP] p*={p_star:.6f}, gini_base={g0:.6f}")
    p.e = e


    # --- Inputs sichern (vor dem SS-Lauf) ---
    save_dir = os.path.join(outdir, "SS")  # outdir hast du oben bereits angelegt
    np.save(os.path.join(save_dir, "e_matrix.npy"), p.e)
    np.save(os.path.join(save_dir, "omega_SS.npy"), p.omega_SS)
    np.save(os.path.join(save_dir, "lambdas.npy"), p.lambdas)


    # --- Shape-Sicherung für ogcore 0.14.10 Broadcasting ---


    # 1) rho: sicherstellen, dass es ein Zeitpfad (T+S, S) ist
    # --- Shape-Fix für get_BQ in ogcore 0.14.10 ---



# 1) rho: als 3D (T+S, S, 1), damit get_BQ via [-1, :] → (S,1) erhält
    rho_arr = np.asarray(p.rho)
    if rho_arr.ndim == 2 and rho_arr.shape == (p.T + p.S, p.S):
        p.rho = rho_arr[:, :, None]                 # (T+S, S, 1)
    elif rho_arr.ndim == 1 and rho_arr.shape == (p.S,):
    # Falls irgendwo zu (S,) reduziert wurde → Zeitpfad künstlich replizieren
        p.rho = np.tile(rho_arr.reshape(1, p.S, 1), (p.T + p.S, 1, 1))
# jetzt liefert p.rho[-1, :] → (S,1)

# 2) lambdas: Zeilenvektor (1, J), damit (S,1) * (1,J) → (S,J)
    p.lambdas = np.asarray(p.lambdas, dtype=float).reshape(1, p.J)

# 3) omega_SS: Spaltenvektor (S,1), damit (S,1) * (S,J) → (S,J)
    p.omega_SS = np.asarray(p.omega_SS, dtype=float).reshape(p.S, 1)

# (optional) Checks
    print("[SHAPES3] rho", np.asarray(p.rho).shape,
        "lambdas", p.lambdas.shape,
        "omega_SS", p.omega_SS.shape,
        "e", p.e.shape)


    ss = run_SS(p, client=None)

    print(">> SS done. Keys (sample):", list(ss.keys())[:20])
    me = ss.get("max_euler_error") or ss.get("max_euler_error_ss")
    print(">> Max Euler Error:", me)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
