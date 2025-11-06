import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from CodeLibrary import Simulation


sim = Simulation(AspenFileName="Batch_reactor_allcompsincluded.bkp", WorkingDirectoryPath=r"c:\Users\Gebruiker\Documents\Studie_master_thesis\Aspen-Python\Python_Aspen\Kinetics_literature_modelling", VISIBILITY=True)
# sim = Simulation(AspenFileName="Batch_reactor_volumefixed.bkp", WorkingDirectoryPath=r"c:\Users\Gebruiker\Documents\Studie_master_thesis\Aspen-Python\Python_Aspen\Kinetics_literature_modelling", VISIBILITY=True)
print("Aspen is connected")

Ea_fit_with_python = 17.14
n_fit_with_python = 0.296
m_fit_with_python = 1.178
R = 8.314/1000
V = 0.5
V_m3 = V/1000 #m^3
M_cat_tot= 0.001 #1g

cat_component = {
    "NI": 12.7,
    "C": 0.7,
    "S": 0.1,
    "AL2O3": 74.07,
    "CAO": 3.19,
    "SIO2": 0.21,
    "FE2O3": 0.97,
    "ZNO": 0.26,
    "MGO": 0.17,
    "MN": 0.039,
    "CO": 0.012,
    "K": 0.36,
    "NA": 0.12,
}

wt_catcomps = {component: wt_percentage / 100.0 for component, wt_percentage in cat_component.items()}
mw_cat_components = {
    "HCL": 36.46,
    "H2O": 18.02,
    "NI": 58.69,
    "C": 12.011,
    "S": 32.06,
    "AL2O3": 101.96,
    "CAO": 56.08,
    "SIO2": 60.08,
    "FE2O3": 159.69,
    "ZNO": 81.38,
    "MGO": 40.30,
    "MN": 54.94,
    "CO": 58.93,
    "K": 39.10,
    "NA": 22.99,
}

M_Ni0_kg = M_cat_tot*mw_cat_components["NI"] #kg of Ni at start
N_Ni0 = M_Ni0_kg /mw_cat_components["NI"] #kmol of Ni at start

cat_comps = list(mw_cat_components.keys())

time_points = np.array([0, 10, 20, 30, 45, 60, 90, 120, 150, 180]) #min

T_concentration_lines = 323
HCl0_values = [0.025, 0.05, 0.1, 0.5, 1.0, 2.0]
X_percent_conc_runs = [
    np.array([0.0, 7.8, 16.1, 22.6, 26.7, 31.2, 34.5, 37.1, 39.1, 40.1]),
    np.array([0.0, 10.8, 20.8, 26.9, 37.1, 42.1, 50.5, 57.7, 60.7, 60.9]),
    np.array([0.0, 14.3, 24.7, 32.7, 42.7, 49.2, 57.7, 61.2, 64.2, 67.0]),
    np.array([0.0, 19.3, 36.4, 49.4, 59.7, 68.8, 77.2, 78.5, 79.0, 80.0]),
    np.array([0.0, 25.0, 48.5, 57.3, 66.8, 72.9, 80.9, 83.9, 85.3, 85.4]),
    np.array([0.0, 31.7, 50.8, 59.7, 72.0, 76.6, 83.3, 84.6, 87.2, 87.6]),
]

HCl0_temperature_lines = 1
Temperature_values = [293, 313, 323, 333, 353]
X_percent_temperature_lines = [
    np.array([0.0, 9.7, 18.6, 27.4, 35.7, 41.6, 50.1, 57.0, 64.8, 69.1]),
    np.array([0.0, 18.2, 31.1, 39.5, 48.4, 55.5, 62.2, 68.5, 74.1, 78.9]),
    np.array([0.0, 25.0, 48.5, 57.3, 66.8, 72.9, 80.9, 83.9, 85.3, 85.4]),
    np.array([0.0, 27.3, 52.7, 63.7, 75.1, 81.8, 88.7, 93.5, 94.2, 94.4]),
    np.array([0.0, 37.9, 62.2, 74.3, 82.1, 88.0, 94.6, 96.5, 96.6, 97.9]),
]

Temperature_desired = 353
HCl0_desired = 1
target_index = Temperature_values.index(Temperature_desired)
X_target_data = X_percent_temperature_lines[target_index]

if target_index != -1 and HCl0_desired == HCl0_temperature_lines: #-1 can be any number except for 4 I think.
    all_sims_parameters = [{
        'T': Temperature_desired,
        'HCl0': HCl0_desired,
        'X_percent': X_target_data,
        't_min': time_points
    }]
else:
    all_sims_parameters = []

X_target_foreverything = np.concatenate([run['X_percent'] for run in all_sims_parameters]) / 100


#Aspen function for running sim
def run_aspen_simulations(k_aspen, Ea, n, m):
    X_sim_everything = []
    retry_number = 3

    global sim
    sim.REACTION_POWERLAW_Set_preexpfactor("CURVEFIT", k_aspen)

    for q, test in enumerate(all_sims_parameters):
        T = test['T']
        HCl0_M =test['HCl0']
        t_min_eval = test['t_min']

        T_C = T-273.15#T in *C

        #Some mass calcs
        mass_sol_in_kg = V *1
        n_HCL = HCl0_M*V_m3
        mass_HCl_kg = n_HCL*mw_cat_components["HCL"]
        mass_H2O_kg = mass_sol_in_kg-mass_HCl_kg
        mass_cat_components_kg = {component: M_cat_tot* mw_cat_components[component] for component in cat_component}

        sim.BLK_BATCH_Set_Temperature("REACTOR", T_C) #Aspen reactor T

        X_simulated = []

        for t_min in t_min_eval:

            if t_min == 0:
                X = 0
            else:
                t_hr = t_min / 60
                X = 0

                massflow_HCl = mass_HCl_kg / t_hr #Acid mix
                massflow_H2O = mass_H2O_kg / t_hr #Acid mix
                massflow_cat_comps = {comp: mass_cat_components_kg[comp] / t_hr for comp in cat_component} #Cat flows
                N_Ni_in_kmol_hr = massflow_cat_comps["NI"] / mw_cat_components["NI"] #Initial Ni massflow
                sim.STRM_Set_ComponentFlowRate("HCLFEED", massflow_H2O, "H2O")
                sim.STRM_Set_ComponentFlowRate("HCLFEED", massflow_HCl, "HCL")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["NI"], "NI")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["C"], "C")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["S"], "S")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["AL2O3"], "AL2O3")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["CAO"], "CAO")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["SIO2"], "SIO2")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["FE2O3"], "FE2O3")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["ZNO"], "ZNO")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["MGO"], "MGO")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["MN"], "MN")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["CO"], "CO")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["K"], "K")
                sim.STRM_CISOLID_Set_ComponentFlowRate("CATFEED", massflow_cat_comps["NA"], "NA")

                for poging in range(retry_number):
                    try:
                        sim.BLK_BATCH_Set_Cycletime("REACTOR", t_hr)
                        sim.BLK_BATCH_Set_Calctime("REACTOR", t_hr)
                        sim.BLK_BATCH_Set_Stopcriteria("REACTOR", t_hr)
                        sim.Run()

                        Ni_out_kmoleflow = sim.STRM_CISOLID_Get_MoleFlowPerCompound("OUT", "NI")
                        if Ni_out_kmoleflow is not None and N_Ni_in_kmol_hr > 1e-12:
                            X = 1-(Ni_out_kmoleflow / N_Ni_in_kmol_hr)
                            break

                    except Exception as e:
                        print(f"Aspen call failed for this: T={T}, HCl0={HCl0_M}, t={t_min} min. Retrying with this: (Attempt {poging + 1})...")
                        time.sleep(2 ** poging)

                else:
                    print(f"No result after {retry_number} attempts. Now, will make X equal to 0")

            X_simulated.append(X)

        X_sim_everything.extend(X_simulated)

    return np.array(X_sim_everything)


def desired_value_function(p_aspen_array, Ea, n, m, X_target):
    p_aspen = p_aspen_array[0]
    k_aspen = 10**p_aspen

    try:
        X_simulated = run_aspen_simulations(k_aspen, Ea, n, m)
    except Exception as e:
        print(f"Aspen failed to run or has an error: {e}")
        return np.inf

    Sum_squared_error = np.sum((X_simulated - X_target)**2)
    return Sum_squared_error


#kalibratie
k0_guess_initial = 8.7866e-06
p0_guess = [np.log10(k0_guess_initial)]

boundaries = [(-15.0, -3.0)]

result = differential_evolution(
    desired_value_function,
    bounds=boundaries,
    args=(Ea_fit_with_python, n_fit_with_python, m_fit_with_python, X_target_foreverything),
    strategy='best1bin',
    maxiter=5,
    popsize=5,
    tol=0.01,
    disp=True
)


#Resultaten
if result.success:
    p_aspen_optimal = result.x[0]
    k_aspen_optimal = 10 **p_aspen_optimal
    print(f"Optimal Aspen Pre-exponential Factor (k_Aspen): {k_aspen_optimal:.4e}")
    print(f"Final Sum squared error is: {result.fun:.4e}")

    X_final_sim = run_aspen_simulations(k_aspen_optimal, Ea_fit_with_python, n_fit_with_python, m_fit_with_python)
    sumsquared_res = np.sum((X_target_foreverything - X_final_sim)**2)
    sumsquared_tot = np.sum((X_target_foreverything - np.mean(X_target_foreverything))**2)
    R2 = 1 -(sumsquared_res/sumsquared_tot)
    print(f"R^2 (T={Temperature_desired}K, HCl0={HCl0_desired}M data): {R2:.4f}")

else:
    p_aspen_best = result.x[0]
    k_aspen_best = 10**p_aspen_best
    print("Calibiration complete(Potential Failure/Limit)")
    print(f"Best k_Aspen found (which may actually not be optimal): {k_aspen_best:.4e}")
