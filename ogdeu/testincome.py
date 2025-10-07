import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.interpolate as si
from ogcore import parameter_plots as pp
#from linearmodels.panel import PanelOLS
from ogcore.parameters import Specifications
from ogcore import utils




# Shape‐Faktor φ_s für Deutschland: Alter 0–19 = 0, Alter 20–80 = aus NTA (gerundet auf 4 Dezimalstellen), Alter 81–100 = 0
fctr = np.array([
    # Alter 0–19
    *[0.0]*20,
    # fctr [21:81], Alter 20–80 (60 Werte)
    0.5741, 0.6108, 0.6289, 0.6245, 0.6270, 0.6348, 0.6356, 0.6494,
    0.6586, 0.6638, 0.6664, 0.6598, 0.6484, 0.6427, 0.6316, 0.6272,
    0.6246, 0.6244, 0.6222, 0.6204, 0.6181, 0.6196, 0.6172, 0.6188,
    0.6200, 0.6211, 0.6163, 0.6115, 0.6078, 0.5989, 0.5967, 0.5933,
    0.5905, 0.5841, 0.5714, 0.5535, 0.5334, 0.5076, 0.4760, 0.4448,
    0.4073, 0.3713, 0.3230, 0.2636, 0.1917, 0.1412, 0.1002, 0.0789,
    0.0678, 0.0652, 0.0646, 0.0670, 0.0630, 0.0585, 0.0541,
    0.0521, 0.0538, 0.0535, 0.0577, 0.0531, 0.0472,
    # Alter 81–100
    *[0.0]*20
])




def arctan_func(xvals, a, b, c):
    r"""
    This function generates predicted ability levels given data (xvals)
    and parameters a, b, and c, from the following arctan function:

    .. math::
        y = (-a / \pi) * \arctan(b * x + c) + (a / 2)

    Args:
        xvals (Numpy array): data inputs to arctan function
        a (scalar): scale parameter for arctan function
        b (scalar): curvature parameter for arctan function
        c (scalar): shift parameter for arctan function

    Returns:
        yvals (Numpy array): predicted values (output) of arctan
            function

    """
    yvals = (-a / np.pi) * np.arctan(b * xvals + c) + (a / 2)
    return yvals


def arctan_deriv_func(xvals, a, b, c):
    r"""
    This function generates predicted derivatives of arctan function
    given data (xvals) and parameters a, b, and c. The functional form
    of the derivative of the function is the following:

    .. math::
        y = - (a * b) / (\pi * (1 + (b * xvals + c)^2))

    Args:
        xvals (Numpy array): data inputs to arctan derivative function
        a (scalar): scale parameter for arctan function
        b (scalar): curvature parameter for arctan function
        c (scalar): shift parameter for arctan function

    Returns:
        yvals (Numpy array): predicted values (output) of arctan
            derivative function

    """
    yvals = -(a * b) / (np.pi * (1 + (b * xvals + c) ** 2))
    return yvals


def arc_error(abc_vals, params):
    """
    This function returns a vector of errors in the three criteria on
    which the arctan function is fit to predict extrapolated ability in
    ages 81 to 100.::

        1) The arctan function value at age 80 must match the estimated
           original function value at age 80.
        2) The arctan function slope at age 80 must match the estimated
           original function slope at age 80.
        3) The level of ability at age 100 must be a given fraction
           (abil_deprec) below the ability level at age 80.

    Args:
        abc_vals (tuple): contains (a,b,c)

            * a (scalar): scale parameter for arctan function
            * b (scalar): curvature parameter for arctan function
            * c (scalar): shift parameter for arctan function
        params (tuple): contains (first_point, coef1, coef2, coef3,
            abil_deprec)

            * first_point (scalar): ability level at age 80, > 0
            * coef1 (scalar): coefficient in log ability equation on
                linear term in age
            * coef2 (scalar): coefficient in log ability equation on
                quadratic term in age
            * coef3 (scalar): coefficient in log ability equation on
                cubic term in age
            * abil_deprec (scalar): ability depreciation rate between
                ages 80 and 100, in (0, 1).

    Returns:
        error_vec (Numpy array): errors ([error1, error2, error3])

            * error1 (scalar): error between ability level at age 80
                from original function minus the predicted ability at
                age 80 from the arctan function given a, b, and c
            * error2 (scalar): error between the slope of the original
                function at age 80 minus the slope of the arctan
                function at age 80 given a, b, and c
            * error3 (scalar): error between the ability level at age
                100 predicted by the original model value times
                abil_deprec minus the ability predicted by the arctan
                function at age 100 given a, b, and c

    """
    a, b, c = abc_vals
    first_point, coef1, coef2, coef3, abil_deprec = params
    error1 = first_point - arctan_func(80, a, b, c)
    if (3 * coef3 * 80**2 + 2 * coef2 * 80 + coef1) < 0:
        error2 = (
            3 * coef3 * 80**2 + 2 * coef2 * 80 + coef1
        ) * first_point - arctan_deriv_func(80, a, b, c)
    else:
        error2 = -0.02 * first_point - arctan_deriv_func(80, a, b, c)
    error3 = abil_deprec * first_point - arctan_func(100, a, b, c)
    error_vec = np.array([error1, error2, error3])

    return error_vec


def arctan_fit(first_point, coef1, coef2, coef3, abil_deprec, init_guesses):
    """
    This function fits an arctan function to the last 20 years of the
    ability levels of a particular ability group to extrapolate
    abilities by trying to match the slope in the 80th year and the
    ability depreciation rate between years 80 and 100.

    Args:
        first_point (scalar): ability level at age 80, > 0
        coef1 (scalar): coefficient in log ability equation on linear
            term in age
        coef2 (scalar): coefficient in log ability equation on
            quadratic term in age
        coef3 (scalar): coefficient in log ability equation on cubic
            term in age
        abil_deprec (scalar): ability depreciation rate between
            ages 80 and 100, in (0, 1)
        init_guesses (Numpy array): initial guesses

    Returns:
        abil_last (Numpy array): extrapolated ability levels for ages
            81 to 100, length 20

    """
    params = [first_point, coef1, coef2, coef3, abil_deprec]
    solution = opt.root(arc_error, init_guesses, args=params, method="lm")
    [a, b, c] = solution.x
    old_ages = np.linspace(81, 100, 20)
    abil_last = arctan_func(old_ages, a, b, c)
    return abil_last



def get_e_US(age_wgts, abil_wgts, plot_path=None):

    """
    Überblick 
    1) USA-Polys → Level 21..80
    2) Alterskalierung via fctr (21..80)
    3) Re-fit log(earn) ~ poly(age) je J
    4) Tail 81..100 via Arctan
    5) Reskalieren auf Mittelwert 1 mit _den übergebenen_ Gewichten
    """


    # Sicherheitscheck
    # Return and error if age_wgts is not a vector of size (80,)
    if age_wgts.shape[0] != 80:
        err = "age_wgts muss Länge 80 haben (Altersjahre 21..100)"
        raise RuntimeError(err)
    # Return and error if abil_wgts is not a vector of size (7,)
    if abil_wgts.shape[0] != 7:
        err = "abil_wgts muss Länge 7 haben"
        raise RuntimeError(err)
    
    # 1) Generate polynomials using USA data and use them to get income profiles for ages 21 to 80.
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
    ages_21_80 = np.arange(21,81) # 60
    A = np.tile(ages_21_80.reshape(60,1),(1,7))
    log_abil_paths = const + one*A + two*(A**2) + three*(A**3)
    abil_paths = np.exp(log_abil_paths)  # US ability paths
    

    # (2) Alters-Skalierung mit fctr (siehe oben)
    abil_paths_shifted = (abil_paths * fctr[21:81].reshape(60, 1)) 


    # 3: Shift the original data by the NTA-based factor and prepare data to run a new regression with shifted data 
    #+ Now re-compute the ability paths using the newly estimated coefficients
    '''
    #(3) Regressions-Fit je J
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

    # wert für const
    # const = [0.85076795, -1.86233513, -3.34685163, -3.66923205, -3.49862477, -0.95923205, -0.66923205]


    one = np.array(
        temp_model_results["Age"]
    )

    # Wert für one: 
    # one = [0.03063091, 0.18778507, 0.30437831, 0.33951476, 0.34421944, 0.17283448, 0.22012605]
    two = np.array(
        temp_model_results["Age^2"]
    )
    
    three = np.array(
        temp_model_results["Age^3"]
    )
    '''
    
    const = [0.85076795, -1.86233513, -3.34685163, -3.66923205, -3.49862477, -0.95923205, -0.66923205]
    one = [0.03063091, 0.18778507, 0.30437831, 0.33951476, 0.34421944, 0.17283448, 0.22012605]
    two = [0.00062368, -0.00189357, -0.00425927, -0.00491826, -0.00506312, -0.00091018, -0.00172369]
    three = [-1.71125612e-05, -3.90256117e-06, 1.16974388e-05, 1.56874388e-05, 1.70974388e-05, -1.33925612e-05, -1.03825612e-05]


    #ages_short_alt = np.tile(np.linspace(21, 80, 60).reshape((60, 1)), (1, 7))

    log_abil_paths_alt = (
        const
        + (one * A)
        + (two * (A**2))
        + (three * (A**3))
    )

    abil_paths_alt = np.exp(log_abil_paths_alt)



    # this exists in OG code, but we need to compute the age_wgts here 
    # Define the ability weights (J)
    #abil_wgts = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])

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

    # 4) lifetime earnings based on re-estimated coefficients
    e_orig_alt = np.zeros((80, 7))
    e_orig_alt[:60, :] = abil_paths_alt
    e_orig_alt[60:, :] = 0.0


    # hier bitte noch prüfen 
    # abil_deprec sind werte die steuern, wie stark die tail extrapolation für Alter 81 bis 100 fällt
    # Für Deutschland ist ein Rückgang um etwa die Hälfte in 20 Jahren gut begründbar und konsistent 
    # mit NTA/AGENTA-Profilen (Arbeits­einkommen konzentriert sich auf 20–60 und fällt im hohen Alter stark)

    abil_deprec = np.array([0.47, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5])
    #     Initial guesses for the arctan. They're pretty sensitive.
    # Startwerte für a, b und c der actan Kurve mit der man den Teil von Es für Alter 81 bis 100 glättet
    # solver ermittelt die endgültigen Paramter die alle drei Randbenigungen erfüllen 
    init_guesses = np.array(
        [
            [58, 0.0756438545595, -5.6940142786],
            [27, 0.069, -5],
            [35, 0.06, -5],
            [37, 0.339936555352, -33.5987329144],
            [70.5229181668, 0.0701993896947, -6.37746859905],
            [35, 0.06, -5],
            [35, 0.06, -5],
        ]
    )
    for j in range(7):
        e_orig_alt[60:, j] = arctan_fit(
            e_orig_alt[59, j],
            one[j],
            two[j],
            three[j],
            abil_deprec[j],
            init_guesses[j],
        )


    # 5) Skalieren  lifetime earnings path matrix  auf Mittelwert 1 (genau die übergebenen Gewichte!)
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
    
    assert e_alt_base.shape == (80,7)
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
    #falls Gini Bruch oder Prozent ist 
    if tgt <= 1.0:
        tgt *= 100.0


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
    e_new = np.maximum(e_alt_base, 1e-12) ** p_star   
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
