import matplotlib.pyplot as plt
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

wt_catcomps = {component: percent / 100.0 for component, percent in cat_component.items()}
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

# M_Ni0_kg = M_cat_tot*mw_cat_components["NI"] #kg of Ni at start
# N_Ni0 = M_Ni0_kg /mw_cat_components["NI"] #kmol of Ni at start
M_Ni0_kg = M_cat_tot * wt_catcomps["NI"] # Correct kg of Ni at start (using weight fraction)
N_Ni0 = M_Ni0_kg / mw_cat_components["NI"] # kmol of Ni at start

cat_comps = list(wt_catcomps.keys())

time_points = np.array([0, 10, 20, 30, 45, 60, 90, 120, 150, 180])

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

all_sims_parameters = []

for T, X_percent in zip(Temperature_values, X_percent_temperature_lines):
    run_title = f"T={T:.1f}K, HCl0={HCl0_temperature_lines:.1f}M"
    all_sims_parameters.append({
        'title': run_title,
        'T': T,
        'HCl0': HCl0_temperature_lines,
        'X_percent': X_percent,
        't_min': time_points,
        'group': 'Temperature',
    })

for HCl0, X_percent in zip(HCl0_values, X_percent_conc_runs):
    if not (T_concentration_lines == 323.0 and HCl0 == 1.0): #323K is already in here, so make sure it is not added again
        run_title = f"T={T_concentration_lines:.1f}K, HCl0={HCl0:.3g}M"
        all_sims_parameters.append({
            'title': run_title,
            'T': T_concentration_lines,
            'HCl0': HCl0,
            'X_percent': X_percent,
            't_min': time_points,
            'group': 'Concentration',
        })


#Aspen
def run_aspen_simulations(k_aspen, Ea, n, m, runs_parameters):
    all_results = {}
    retry_number = 3
    global sim
    global N_Ni0

    sim.REACTION_POWERLAW_Set_preexpfactor("CURVEFIT", k_aspen)
    N_Ni0_initial = N_Ni0 #Initial Ni concentration

    for q, run in enumerate(runs_parameters):
        T = run['T']
        HCl0_M = run['HCl0']
        t_min_eval = run['t_min']
        run_title = run['title']

        T_C = T-273.15

        #Some mass calcs
        mass_sol_in_kg = V *1.0
        n_HCL = HCl0_M*V_m3
        mass_HCl_kg = n_HCL*mw_cat_components["HCL"]
        mass_H2O_kg = mass_sol_in_kg - mass_HCl_kg
        # mass_cat_components_kg = {component: M_cat_tot* mw_cat_components[component] for component in cat_comps}
        # NWC FIXED (CORRECT):
        mass_cat_components_kg = {component: M_cat_tot * wt_catcomps[component] for component in cat_comps}

        sim.BLK_BATCH_Set_Temperature("REACTOR", T_C)
        X_simulated = []

        for t_min in t_min_eval:

            if t_min == 0:
                X = 0.0
            else:
                t_hr = t_min / 60.0
                X = 0.0

                #Masfflows and input for Aspen
                massflow_HCl = mass_HCl_kg / t_hr
                massflow_H2O = mass_H2O_kg / t_hr
                massflow_cat_comps = {comp: mass_cat_components_kg[comp] / t_hr for comp in cat_comps}
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

                        kmolflow_Ni_out = sim.STRM_CISOLID_Get_MoleFlowPerCompound("OUT", "NI")
                        if kmolflow_Ni_out is not None and N_Ni0_initial > 1e-12:
                            n_Ni_remaining = kmolflow_Ni_out*t_hr
                            X = 1-(n_Ni_remaining / N_Ni0_initial)
                            break
                        else:
                            kmolflow_Ni_out = 0  #reset for retry

                    except Exception as e:
                        if poging == 0:
                            print(f"Aspen call failed for this: t={t_min} min. Retrying with this: (Attempt {poging + 1})...")
                        time.sleep(2**poging)

                else:
                    print(f"No result after {retry_number} attempts. Now, will make X = 0")

            X_simulated.append(X)

        X_experimental = run['X_percent'] / 100
        X_simulation = np.array(X_simulated)

        #R^2
        sumsquared_res = np.sum((X_experimental-X_simulation)**2)
        sumsquared_tot = np.sum((X_experimental -np.mean(X_experimental))**2)
        R2 = 1 -(sumsquared_res / sumsquared_tot) if sumsquared_tot > 1e-12 else 1

        all_results[run_title] = {
            'X_sim': X_simulation,
            'X_exp': X_experimental,
            't_min': run['t_min'],
            'group': run['group'],
            'T': T,
            'HCl0': HCl0_M,
            'R2': R2
        }

    return all_results

k_best_found_fit = 7.8771e-05 #input is here! Take this value from Aspen k fitting with the 353K line
final_results = run_aspen_simulations(k_best_found_fit, Ea_fit_with_python, n_fit_with_python, m_fit_with_python, all_sims_parameters)

temperature_runs = [res for title, res in final_results.items() if res['group'] == 'Temperature']
concentration_runs = [res for title, res in final_results.items() if res['group'] == 'Concentration']


#######Plot T influence
plt.figure(figsize=(10, 6))
axis1 = plt.gca()
axis1.tick_params(axis='y', labelsize=14)
axis1.tick_params(axis='x', labelsize=14)
plt.xlabel('Time (minutes)', fontsize = 18)
plt.ylabel('Conversion (%)', fontsize = 18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Conversion versus time at different temperatures')
colors_plot = plt.cm.plasma(np.linspace(0.1, 0.9, len(Temperature_values)))
temperature_runs.sort(key=lambda x: x['T'])

for plott, run in enumerate(temperature_runs):
    T = run['T']
    R2 = run['R2']
    color = colors_plot[plott]

    plt.plot(run['t_min'], run['X_sim']*100, label=f'Sim {T:.0f}K (R$^2$={R2:.3f})', color=color, linestyle='-', linewidth=2)
    plt.plot(run['t_min'], run['X_exp']*100, label=f'Exp {T:.0f}K', color=color, marker='o', linestyle='', markersize=5)

plt.legend(loc='lower right', ncol=2, fontsize=16, title_fontsize = 18)
plt.tight_layout()
plt.show()


#######Plot concentration effect
plt.figure(figsize=(10, 6))
axis2 = plt.gca()
axis2.tick_params(axis='y', labelsize=14)
axis2.tick_params(axis='x', labelsize=14)
plt.xlabel('Time (min)', fontsize = 18)
plt.ylabel('Conversion (%)', fontsize = 18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Conversion versus time at different HCl concentrations')

colors_plot2 = plt.cm.viridis(np.linspace(0.1, 0.9, len(HCl0_values)))
concentration_runs.sort(key=lambda x: x['HCl0'])

for plotje, run in enumerate(concentration_runs):
    HCl0_M = run['HCl0']
    R2 = run['R2']
    color = colors_plot2[plotje]

    plt.plot(run['t_min'], run['X_sim']*100, label=f'Sim {HCl0_M:.3g}M (R$^2$={R2:.3f})', color=color, linestyle='-', linewidth=2)
    plt.plot(run['t_min'], run['X_exp']*100, label=f'Exp {HCl0_M:.3g}M', color=color, marker='^', linestyle='', markersize=5)

plt.legend(loc='lower right', ncol=2, fontsize=16, title_fontsize = 18)
plt.tight_layout()
plt.show()


########################Plot R2 values
temperature_labels = [f"{run['T']:.0f}K" for run in temperature_runs]
temperature_R2_values = [run['R2'] for run in temperature_runs]

concent_labels = [f"{run['HCl0']:.3g}M" for run in concentration_runs]
concent_R2_values = [run['R2'] for run in concentration_runs]


plt.figure(figsize=(8, 5))
plt.bar(temperature_labels, temperature_R2_values, color='skyblue', width=0.6)
axis3 = plt.gca()
axis3.tick_params(axis='y', labelsize=14)
axis3.tick_params(axis='x', labelsize=14)
plt.xlabel('Temperature (K)', fontsize=18)
plt.ylabel('R^2', fontsize=18)
plt.title('R^2 values by temperature', fontsize=14)
plt.ylim(max(0, min(temperature_R2_values) - 0.1), 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
for Rval, R2 in enumerate(temperature_R2_values):
    plt.text(Rval, R2 + 0.01, f'{R2:.3f}', ha='center', va='bottom', fontsize=16)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.bar(concent_labels, concent_R2_values, color='lightcoral', width=0.6)
axis4 = plt.gca()
axis4.tick_params(axis='y', labelsize=14)
axis4.tick_params(axis='x', labelsize=14)
plt.xlabel('HCl Concentration (M)', fontsize=18)
plt.ylabel('R^2', fontsize=18)
plt.title('R^2 values by HCl concentration', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(max(0, min(concent_R2_values) - 0.1), 1.05)
plt.xticks(rotation=0)

for Rvalagain, R2 in enumerate(concent_R2_values):
    plt.text(Rvalagain, R2 + 0.01, f'{R2:.3f}', ha='center', va='bottom', fontsize=16)
plt.tight_layout()
plt.show()


############Total R2
X_simulatie_total = np.concatenate([res['X_sim'] for res in final_results.values()])
X_experiment_total = np.concatenate([res['X_exp'] for res in final_results.values()])

sumsquare_resid = np.sum((X_experiment_total - X_simulatie_total)**2)
sumsquare_totaal = np.sum((X_experiment_total - np.mean(X_experiment_total))**2)
R2_total = 1 - (sumsquare_resid / sumsquare_totaal) if sumsquare_totaal > 1e-12 else 1.0
print(f"Total sum squared error: {sumsquare_resid:.4e}")
print(f"Total R^2 : {R2_total:.4f}")

