# ogdeu/calibrate.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from ogcore.parameters import Specifications


@dataclass
class Calibration:
    # Pfade & Settings
    overrides_file: Path = Path("policy/baseline_overrides.json")
    baseline: bool = True
    output_base: str = "OUTPUT_BASELINE"

    def build_specs(self) -> Specifications:
        specs = Specifications(baseline=self.baseline, output_base=self.output_base)
        if self.overrides_file.exists():
            with self.overrides_file.open("r", encoding="utf-8") as f:
                overrides = json.load(f)
            specs.update_specifications(overrides)
        else:
            print(f"[WARN] overrides file not found: {self.overrides_file.resolve()}")
        return specs


    # ohne e 
    def attach_demography(self, specs: Specifications, download_path: str | None = "data/demography"):
        from .demographics import get_demog_S

        demog_S = get_demog_S(specs, download_path=download_path, graph=False)

        # ParamTools-Format bauen: {"param": [{"value": ...}]}
        def pt_wrap(val):
            arr = np.asarray(val)
            return [{"value": arr.tolist()}] if arr.ndim >= 1 else [{"value": float(arr)}]

        demog_overrides = {k: pt_wrap(v) for k, v in demog_S.items()}
        specs.update_specifications(demog_overrides)

        # optional für spätere Nutzung:
        self.demographic_params = demog_S
        return demog_S
