import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.interpolate as si
from ogcore import parameter_plots as pp
from linearmodels.panel import PanelOLS
from ogcore.parameters import Specifications
from ogcore import utils




# Shape‐Faktor φ_s für Deutschland: Alter 0–19 = 0, Alter 20–80 = aus NTA (gerundet auf 4 Dezimalstellen), Alter 81–100 = 0
fctr = np.array([
    # Alter 0–19
    *[0.0]*20,
    # Alter 20–43 (24 Werte)
    0.5741, 0.6108, 0.6289, 0.6245, 0.6270, 0.6348, 0.6356, 0.6494,
    0.6586, 0.6638, 0.6664, 0.6598, 0.6484, 0.6427, 0.6316, 0.6272,
    0.6246, 0.6244, 0.6222, 0.6204, 0.6181, 0.6196, 0.6172, 0.6188,
    # Alter 44–67 (24 Werte)
    0.6200, 0.6211, 0.6163, 0.6115, 0.6078, 0.5989, 0.5967, 0.5933,
    0.5905, 0.5841, 0.5714, 0.5535, 0.5334, 0.5076, 0.4760, 0.4448,
    0.4073, 0.3713, 0.3230, 0.2636, 0.1917, 0.1412, 0.1002, 0.0789,
    # Alter 68–80 (13 Werte)
    0.0678, 0.0652, 0.0646, 0.0670, 0.0630, 0.0585, 0.0541,
    0.0521, 0.0538, 0.0535, 0.0577, 0.0531, 0.0472,
    # Alter 81–100
    *[0.0]*20
])


def get_e_orig(age_wgts, abil_wgts, plot_path=None):

    # Return and error if age_wgts is not a vector of size (80,)
    if age_wgts.shape[0] != 80:
        err = "Vector age_wgts does not have 80 elements."
        raise RuntimeError(err)
    # Return and error if abil_wgts is not a vector of size (7,)
    if abil_wgts.shape[0] != 7:
        err = "Vector abil_wgts does not have 7 elements."
        raise RuntimeError(err)
    
    # 1) Generate polynomials using USA data and use them to get income profiles for
    #    ages 21 to 80.
    one = np.array(
        [
            -0.09720122,
            0.05995294,
            0.17654618,
            0.21168263,
            0.21638731,
            0.04500235,
            0.09229392,
        ]
    )
    two = np.array(
        [
            0.00247639,
            -0.00004086,
            -0.00240656,
            -0.00306555,
            -0.00321041,
            0.00094253,
            0.00012902,
        ]
    )
    three = np.array(
        [
            -0.00001842,
            -0.00000521,
            0.00001039,
            0.00001438,
            0.00001579,
            -0.00001470,
            -0.00001169,
        ]
    )
    const = np.array(
        [
            3.41e00,
            0.69689692,
            -0.78761958,
            -1.11e00,
            -0.93939272,
            1.60e00,
            1.89e00,
        ]
    )

    #vector of ages 
    ages_short = np.tile(np.linspace(21, 80, 60).reshape((60, 1)), (1, 7))
    log_abil_paths = (
        const
        + (one * ages_short)
        + (two * (ages_short**2))
        + (three * (ages_short**3))
    )


    #original ability paths
    abil_paths = np.exp(log_abil_paths)
    


    #Multipy the original ability paths by the adjustment factor (fctr)
    # --- (2) Shift via fctr --
    abil_paths_shifted = (abil_paths * fctr[21:81].reshape(60, 1)) 

    # --- (3) Regressions-Setup (Panel OLS) ---
    #Prepare the dataset
    data = pd.DataFrame(abil_paths_shifted)
    data['age'] = data.loc[:,"age"] = np.arange(start=21, stop=81)  
    data.columns = data.columns.astype(str)
    data = pd.melt(data, id_vars='age', value_vars=['0', '1', '2', '3', '4', '5', '6'])

    data['age2'] = data['age'] ** 2
    data['age3'] = data['age'] ** 3
    data['ln_earn_rate'] = np.log(data['value'])

    #create 2 level index for panel ols, create dummies, and clean up
    data['ageind']=data['age']
    data['varind']=data['variable']
    data = data.set_index(["varind", "ageind"])

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = pd.get_dummies(data, columns=['variable'])


    model_results = {
        "Names": [
            "Constant",
            "",
            "Head Age",
            "",
            "Head Age^2",
            "",
            "Head Age^3",
            "",
            "R-Squared",
            "Observations",
        ]
    }
    #cats_pct = ["0-25", "26-50", "51-70", "71-80", "81-90", "91-99", "100"]
    cats_pct = ["variable_0", "variable_1", "variable_2", "variable_3", "variable_4", "variable_5", "variable_6"]
    # long_model_results is used to report the results and save in a file
    long_model_results = {
        "Lifetime Income Group": [],
        "Constant": [],
        "Age": [],
        "Age^2": [],
        "Age^3": [],
        "Observations": [],
    }

    # temp_model_results is used to pass the coefficients to the calculation below
    temp_model_results = {
        "Lifetime Income Group": [],
        "Constant": [],
        "Age": [],
        "Age^2": [],
        "Age^3": [],
    }

    #run the model for each J (cats_pct)
    for i, group in enumerate(cats_pct):  
        data2 = data[data[group] == 1].copy()
        data2["ones"] = np.ones(len(data2.index))
        mod = PanelOLS(
            data2.ln_earn_rate, data2[["ones", "age", "age2", "age3"]]
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        # print('Summary for lifetime income group ', group)
        # print(res.summary)
        # Save model results to dictionary
        model_results[group] = [
            res.params["ones"],
            res.std_errors["ones"],
            res.params["age"],
            res.std_errors["age"],
            res.params["age2"],
            res.std_errors["age2"],
            res.params["age3"],
            res.std_errors["age3"],
            res.rsquared,
            res.nobs,
        ]
        long_model_results["Lifetime Income Group"].extend([cats_pct[i], ""])
        long_model_results["Constant"].extend(
            [res.params["ones"], res.std_errors["ones"]]
        )
        long_model_results["Age"].extend(
            [res.params["age"], res.std_errors["age"]]
        )
        long_model_results["Age^2"].extend(
            [res.params["age2"], res.std_errors["age2"]]
        )
        long_model_results["Age^3"].extend(
            [res.params["age3"], res.std_errors["age3"]]
        )
        long_model_results["Observations"].extend([res.nobs, ""])
        
        temp_model_results["Constant"].extend(
            [res.params["ones"]]
        )
        temp_model_results["Age"].extend(
            [res.params["age"]]
        )
        temp_model_results["Age^2"].extend(
            [res.params["age2"]]
        )
        temp_model_results["Age^3"].extend(
            [res.params["age3"]]
        )


    const = np.array(
        temp_model_results["Constant"]
    )
    one = np.array(
        temp_model_results["Age"]
    )
    two = np.array(
        temp_model_results["Age^2"]
    )
    three = np.array(
        temp_model_results["Age^3"]
    )

    ages_short_alt = np.tile(np.linspace(21, 80, 60).reshape((60, 1)), (1, 7))
    log_abil_paths_alt = (
        const
        + (one * ages_short_alt)
        + (two * (ages_short_alt**2))
        + (three * (ages_short_alt**3))
    )
    abil_paths_alt = np.exp(log_abil_paths_alt)



    # this exists in OG code, but we need to compute the age_wgts here 
    # Define the ability weights (J)
    abil_wgts = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])

    #We want to create the population weights for each year, to get `age_wgts`
    pop_target ='/home/jovyan/work/un_ge_population.csv'

    # Convert .csv file to Pandas DataFrame
    pop_df = pd.read_csv(
        pop_target,
        sep=",", 
        header=0,    
        usecols=["TimeLabel", "SexId", "Sex", "AgeStart", "Value", "VariantId"],
        float_precision="round_trip",
    )
    pop_df=pop_df.loc[pop_df['VariantId'] == 4]
    pop_df=pop_df.loc[pop_df['SexId'] == 3]
    #to correct: header = 0; separator line delete
    # Rename variables in the population and fertility rates data
    pop_df.rename(
        columns={
            "TimeLabel": "year",
            "SexId": "sex_num",
            "Sex": "sex_str",
            "AgeMid": "age",
            "Value": "pop",
            "VariantId": "variant"
        },
        inplace=True,
    )
    pop_df = (
    pop_df.reset_index()
    )
    pop_df['age_share'] = pop_df['pop'] / pop_df['pop'].sum()
    age_wgts = np.array(pop_df['age_share'])


    print("age_wgts ist:",age_wgts)
    #age_wgts

    # 2) lifetime earnings based on re-estimated coefficients
    e_orig_alt = np.zeros((80, 7))
    e_orig_alt[:60, :] = abil_paths_alt
    e_orig_alt[60:, :] = 0.0

    # Rescale the lifetime earnings path matrix so that the
    #    population weighted average equals 1.
    e_alt = (
        e_orig_alt
        / (e_orig_alt * age_wgts[21:].reshape(80, 1) * abil_wgts.reshape(1, 7)).sum()
    )

    return e_alt 

    #print("e_alt ist:",e_alt)
    #print(len(e_alt))


def match_gini(e_alt_base: np.ndarray, lambdas: np.ndarray, age_wgts : np.ndarray, gini_to_match=0.311, plot=False):
    
    """
    Finde p, so dass (emat_base ** p) den Ziel-Gini trifft. Danach reskalieren auf Mittelwert 1.
    """
    
    
    assert lambdas.shape[0] == 7
    assert age_wgts.shape[0] == 80
    '''
    # brauche ich was? 
    usa_params = Specifications()
    usa_params.omega_SS = age_wgts.copy()
    usa_params.lambdas  = lambdas.copy()
    usa_params.S        = S
    usa_params.J        = J
    usa_params.E        = E.copy()

    '''
    
    # --- NEU: Power-Transformation statt e*exp(a*e) ---
    # (1) Hilfsfunktion: Gini als Funktion von p
    def gini_of_p(p):
        # Falls irgendwo 0-Werte vorkommen:
        e_base = np.maximum(e_alt_base, 1e-12)
        em = e_base ** p
        return utils.Inequality(em, age_wgts, lambdas, 80, 7).gini()

    tgt = float(gini_to_match)
    g0  = gini_of_p(1.0)   # p=1 -> baseline

    # (2) Bracket für p je nach Richtung
    if tgt < g0:
        # Ungleichheit senken -> p in (0,1)
        lo, hi = 0.05, 1.0
    else:
        # Ungleichheit erhöhen -> p > 1
        lo, hi = 1.0, 3.0

    # Sicherheit: Enden ggf. ausweiten, bis Vorzeichenwechsel
    f_lo = gini_of_p(lo) - tgt
    f_hi = gini_of_p(hi) - tgt
    k = 0
    while (f_lo * f_hi > 0) and k < 25:
        if tgt < g0:
            lo = max(lo/2, 1e-4)   # weiter Richtung 0
        else:
            hi *= 1.5              # weiter nach oben
        f_lo = gini_of_p(lo) - tgt
        f_hi = gini_of_p(hi) - tgt
        k += 1
    if f_lo * f_hi > 0:
        raise ValueError(f"Kein Vorzeichenwechsel für p gefunden (g0={g0:.6f}, tgt={tgt:.6f}).")

    # (3) p* finden
    p_star = opt.root_scalar(lambda p: gini_of_p(p) - tgt,
                             method="bisect", bracket=[lo, hi], xtol=1e-10).root

    # (4) E neu und skalieren (Mittelwert 1 mit DE-Gewichten)
    e_new = np.maximum(e_alt_base, 1e-12) ** p_star   # <-- NUR diese Zeile ersetzt dein e*exp(a*e)
    emat_final_scaled = e_new / (
        e_new * age_wgts.reshape(80, 1) * lambdas.reshape(1, 7)
    ).sum()

    return emat_final_scaled 



'''

# von demogrphics.py 
omega_SS_2024 = np.array([
    0.011808, 0.012227, 0.012600, 0.012886, 0.013080, 0.013194, 0.013253, 0.013274,
    0.013271, 0.013254, 0.013235, 0.013218, 0.013204, 0.013200, 0.013202, 0.013207,
    0.013217, 0.013232, 0.013252, 0.013281, 0.013320, 0.013372, 0.013438, 0.013520,
    0.013617, 0.013728, 0.013857, 0.014001, 0.014161, 0.014335, 0.014521, 0.014718,
    0.014923, 0.015132, 0.015339, 0.015541, 0.015732, 0.015907, 0.016062, 0.016193,
    0.016295, 0.016367, 0.016406, 0.016411, 0.016382, 0.016321, 0.016230, 0.016112,
    0.015970, 0.015809, 0.015629, 0.015436, 0.015231, 0.015020, 0.014802, 0.014578,
    0.014357, 0.014126, 0.013880, 0.013596, 0.013271, 0.012906, 0.012479, 0.011984,
    0.011425, 0.010793, 0.010083, 0.009294, 0.008436, 0.007528, 0.006604, 0.005663,
    0.004715, 0.003814, 0.002997, 0.002285, 0.001687, 0.001202, 0.000822, 0.000543
], dtype=float)

age_wgts = np.asarray(omega_SS_2024, dtype=float); age_wgts /= age_wgts.sum()
lambdas  = np.asarray([0.25,0.25,0.20,0.10,0.10,0.09,0.01], dtype=float); lambdas /= lambdas.sum()
gini_de  = 0.311  # Bruchteil, nicht Prozent



# Shapes prüfen
print("e_DE shape:", e_DE.shape)          # sollte (S, J) sein
print("len(lambdas):", np.asarray(lambdas).size)


# Nur für den Plot: 21..80 (inclusive) anzeigen
ages_21_80 = np.arange(21, 81)        # Länge 60
emat_21_80 = e_DE[:len(ages_21_80), :]  # erste 60 Alterszeilen


'''
