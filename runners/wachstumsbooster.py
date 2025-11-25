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
    base_dir = os.path.join(CUR_DIR, "OG-Germany-TEST3", "OUTPUT_BASELINE")
    reform_dir = os.path.join(CUR_DIR, "OG-Germany-TEST3", "OUTPUT_REFORM")

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

    p.BW = 8

    
    client = Client(processes=False)
    print("Dask Client läuft im Threaded-Modus")
    print("Dashboard:", client.dashboard_link)
    print("Dask Client läuft im Threaded-Modus (processes=False)")
    c = Calibration(p, estimate_tax_functions=True, client=client )
   

    d = c.get_dict()
    updated_params = {
        #demogrphie und e

        "cit_rate" : [
            [0.30],
            ],
        "delta_tau_annual": [
            [0.08140],
            ],
        "omega": d["omega"],
        "g_n_ss": d["g_n_ss"],
        "omega_SS": d["omega_SS"],
        "rho": d["rho"],
        "g_n": d["g_n"],
        "imm_rates": d["imm_rates"],
        "omega_S_preTP": d["omega_S_preTP"],

        "initial_guess_r_SS": 0.05,
        "initial_guess_TR_SS": 0.02,
        "nu":0.4,

        "e": d["e"],
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


    print("Closing Baseline Client to free memory...")
    time.sleep(5) # Kurze Pause, damit das OS aufräumen kann

    """
    ---------------------------------------------------------------------------
    Run reform policy
    ---------------------------------------------------------------------------
    """

    # create new Specifications object for reform simulation
    #p2 = copy.deepcopy(p)

    p2 = Specifications(
        baseline=False,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=reform_dir,
    )
    p2.update_specifications(
        json.load(
            open(
                os.path.join(
                    CUR_DIR, "..", "ogdeu", "ogdeu_default_parameters.json"
                )
            )
        )
    )


    p2.BW = 8  # 2025–2032
    
    '''
    c2 = Calibration(
        p2, estimate_tax_functions=True, client=client
    )
    '''
    c2 = Calibration(
        p2, estimate_tax_functions=True, client=client
    )

    client.close()
    del client
    client = Client(processes=False)


    d = c2.get_dict()

    updated_params_reform = {
        "cit_rate" : [
            [0.30], 
            [0.30],  
            [0.30],  
            [0.29],  
            [0.28],  
            [0.27],  
            [0.26],  
            [0.25],  
            ],
        "delta_tau_annual": [
            [0.08140],  
            [0.08250],  
            [0.08340],  
            [0.08310],  
            [0.08280],  
            [0.08250], 
            [0.08220], 
            [0.08200],  
            ],
            
        "omega": d["omega"],
        "g_n_ss": d["g_n_ss"],
        "omega_SS": d["omega_SS"],
        "rho": d["rho"],
        "g_n": d["g_n"],
        "imm_rates": d["imm_rates"],
        "omega_S_preTP": d["omega_S_preTP"],

        "initial_guess_r_SS": 0.05,
        "initial_guess_TR_SS": 0.02,
        "nu":0.4,

        "e": d["e"],
        "etr_params": d["etr_params"],
        "mtrx_params": d["mtrx_params"],
        "mtry_params": d["mtry_params"],
        "mean_income_data": d["mean_income_data"],
        "frac_tax_payroll": d["frac_tax_payroll"],
    }


    p2.update_specifications(updated_params_reform)
    # Run model
    print (">>>> running reform (cit only)")
    start_time = time.time()
    #runner(p2, time_path=True, client=client)
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
    ans.to_csv("ogdeu_example_output2.csv")


if __name__ == "__main__":
    # execute only if run as a script
    main()

