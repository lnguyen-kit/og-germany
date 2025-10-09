# Need to fix references to Calculator, reform json, and substitute new tax
# function call
import multiprocessing
from distributed import Client
import os
import json
import time
import copy

# from taxcalc import Calculator
from ogdeu.calibrate1 import Calibration
from ogcore.parameters import Specifications
from ogcore import output_tables as ot
from ogcore import output_plots as op
from ogcore import parameter_plots as pp
from ogcore.execute import runner
from ogcore.utils import safe_read_pickle


def main():
    # Define parameters to use for multiprocessing
    client = None
    num_workers = min(multiprocessing.cpu_count(), 7)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, "OG-Germany-Example", "OUTPUT_BASELINE")
    reform_dir = os.path.join(CUR_DIR, "OG-Germany-Example", "OUTPUT_REFORM")

    """
    ---------------------------------------------------------------------------
    Run baseline policy
    ---------------------------------------------------------------------------
    """
    # Set up baseline parameterization
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )

    # Update parameters for baseline from default json file
    p.update_specifications(
        json.load(
            open(
                os.path.join(
                    CUR_DIR, "..", "ogdeu", "ogdeu_default_parameters.json"
                )
            )
        )
    )

    # Update parameters from calibrate.py Calibration class
    #BW ist „number of years in the budget window (the period over which tax policy is assumed to vary)“.
    # Wenn du die Steuerpolitik nur für ein Jahr schätzt/fixierst, genügt BW=1
    p.BW = 1 
    p.tax_func_type = "GS"
    c = Calibration(p, estimate_tax_functions=True, client=client)
    client = Client()

    # Einträge omega, gn_ss, usw. kommen aus dme Dictionary d der Calibration klasse 
    # nur eine Teilmenge aus d weren in updated_params gepackt und per p.udate_specifications(updated_params) überschrieben 
    #andere Parameter wie die Makroparameter gamma, g_y_annaule usw. werden gehören zu den default werten aus der Json file (auch in p geladen)
    #udpated_params Dictionary wird aus ausgewählten schlüsseln von d gebaut 

    # omega, g_n_ss, omega_SS, rho, g_n, imm_rates und omega_s_preTP werden endogen im demogrphics.py ermittelt (sehr lang für json)
    # e, etr_params, mtrx_params, mtry_params, mean_income_data und fact_tax_paroll wird auch endogen ermittelt (sehr lang für json)
    d = c.get_dict()
    updated_params = {
        #demogrphie und e
        "omega": d["omega"],
        "g_n_ss": d["g_n_ss"],
        "omega_SS": d["omega_SS"],
        "rho": d["rho"],
        "g_n": d["g_n"],
        "imm_rates": d["imm_rates"],
        "omega_S_preTP": d["omega_S_preTP"],

        "e": d["e"],
        # Hasuhaltsparamter
        "etr_params": d["etr_params"],
        "mtrx_params": d["mtrx_params"],
        "mtry_params": d["mtry_params"],
        "mean_income_data": d["mean_income_data"],
        "frac_tax_payroll": d["frac_tax_payroll"],
    }
    # Ditionary wird in p gemerged 
    p.update_specifications(updated_params)

    print (">>>>> Running baseline")
    # Run model
    start_time = time.time()
    runner(p, time_path=True, client=client)
    print("baseline run time = ", time.time() - start_time)

    """
    ---------------------------------------------------------------------------
    Run reform policy
    ---------------------------------------------------------------------------
    """

    # create new Specifications object for reform simulation
    p2 = copy.deepcopy(p)
    p2.baseline = False
    p2.output_base = reform_dir

    '''
    #Create a PIT reform
    pit_reform = {
        2020: {
            "_std_deduction": [50000],
            "_rebate_thd": [500000],
            "_rebate_ceiling": [12500],
        }
    }
    # create new calibration object for reform simulation
    c2 = Calibration(
        p2, pit_reform=pit_reform, estimate_tax_functions=True, client=client
    )
    # update tax function parameters in Specifications Object
    d = c2.get_dict()
    # additional parameters to change
    updated_params_ref = {
        "cit_rate": [[0.35]],
        "etr_params": d["etr_params"],
        "mtrx_params": d["mtrx_params"],
        "mtry_params": d["mtry_params"],
        "mean_income_data": d["mean_income_data"],
        "frac_tax_payroll": d["frac_tax_payroll"],
    }
    p2.update_specifications(updated_params_ref)
    
    '''
    new_cit = [[0.35]]
    p2.update_specifications({"cit_rate" : new_cit})

    # Run model
    print (">>>> running reform (cit only)")
    start_time = time.time()
    runner(p2, time_path=True, client=client)
    print("reform run time = ", time.time() - start_time)
    client.close()

    """
    ---------------------------------------------------------------------------
    Save some results of simulations
    ---------------------------------------------------------------------------
    """
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, "TPI", "TPI_vars.pkl")
    )
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, "model_params.pkl")
    )
    ans = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=base_params.start_year,
    )

    # create plots of output
    op.plot_all(
        base_dir, reform_dir, os.path.join(CUR_DIR, "OG-Germany_example_plots")
    )

    print("Percentage changes in aggregates:", ans)
    # save percentage change output to csv file
    ans.to_csv("ogdeu_example_output.csv")


if __name__ == "__main__":
    # execute only if run as a script
    main()