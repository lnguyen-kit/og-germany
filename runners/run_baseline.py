from ogcore.SS import run_SS
from ogcore import TPI
from ogdeu.calibrate import Calibration
import os, pickle, shutil

def main():
    # 1) Specs & Demografie
    calib = Calibration()
    p = calib.build_specs()
    calib.attach_demography(p, download_path="data/demography")

    # (nur zur Sicherheit)
    os.makedirs(os.path.join(p.output_base, "SS"), exist_ok=True)
    os.makedirs(os.path.join(p.output_base, "TPI"), exist_ok=True)

    # 2) Steady State rechnen
    ss = run_SS(p, client=None)

    # 3) SS sauber speichern (ohne utils.save_pickle)
    ss_dir = os.path.join(p.output_base, "SS")
    ss_vars = os.path.join(ss_dir, "SS_vars.pkl")
    with open(ss_vars, "wb") as f:
        pickle.dump(ss, f)

    # 4) Baseline-Alias anlegen, den TPI erwartet
    ss_baseline = os.path.join(ss_dir, "SS_baseline_vars.pkl")
    shutil.copyfile(ss_vars, ss_baseline)

    # 5) TPI laufen lassen (liest jetzt SS_baseline_vars.pkl)
    tpi = TPI.run_TPI(p, client=None)
    print("TPI done. keys:", list(tpi.keys())[:10])

if __name__ == "__main__":
    main()
