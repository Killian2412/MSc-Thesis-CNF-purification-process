import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

#######Input
R = 8.314/1000 #kJ/(mol*K)
V = 0.5  #L
V_m3 = V/ 1000 #m^3
N_Ni0 = 0.0021639/1000 #kmol of Ni initially. 12.7 wt% in 1g.
C_Ni0 = N_Ni0/V_m3 #kmol/m^3

all_simulations_parameters = [] #for storage

#######Data arrays. Data is from Parhi2013 Figure 5 and 6.
time_min_array = np.array([0, 10, 20, 30, 45, 60, 90, 120, 150, 180])
Temperature_concentration_variation = 323.0 #K
Concentration_array = [0.025, 0.05, 0.1, 0.5, 1.0, 2.0] #M, which is kmol/m^3
X_concentration_variation_array = [
    np.array([0.0, 7.8, 16.1, 22.6, 26.7, 31.2, 34.5, 37.1, 39.1, 40.1]), #0.025 M
    np.array([0.0, 10.8, 20.8, 26.9, 37.1, 42.1, 50.5, 57.7, 60.7, 60.9]), #0.05
    np.array([0.0, 14.3, 24.7, 32.7, 42.7, 49.2, 57.7, 61.2, 64.2, 67.0]), #0.1
    np.array([0.0, 19.3, 36.4, 49.4, 59.7, 68.8, 77.2, 78.5, 79.0, 80.0]), #0.5
    np.array([0.0, 25.0, 48.5, 57.3, 66.8, 72.9, 80.9, 83.9, 85.3, 85.4]), #1
    np.array([0.0, 31.7, 50.8, 59.7, 72.0, 76.6, 83.3, 84.6, 87.2, 87.6]), #2
]

HCl0_temperature_variation = 1.0 #M, which is kmol/m^3
Temperature_array = [293, 313, 323, 333, 353] #K
X_temperature_variation_array = [
    np.array([0.0, 9.7, 18.6, 27.4, 35.7, 41.6, 50.1, 57.0, 64.8, 69.1]), #293K
    np.array([0.0, 18.2, 31.1, 39.5, 48.4, 55.5, 62.2, 68.5, 74.1, 78.9]), #313
    np.array([0.0, 25.0, 48.5, 57.3, 66.8, 72.9, 80.9, 83.9, 85.3, 85.4]), #323
    np.array([0.0, 27.3, 52.7, 63.7, 75.1, 81.8, 88.7, 93.5, 94.2, 94.4]), #333
    np.array([0.0, 37.9, 62.2, 74.3, 82.1, 88.0, 94.6, 96.5, 96.6, 97.9]), #353
]


#######For loops. Add data to the all_simulations_parameters.
#Temperature variation
for Temp_of_exp, X_data in zip(Temperature_array, X_temperature_variation_array):
    all_simulations_parameters.append({'T': Temp_of_exp, 'HCl0': HCl0_temperature_variation, 'X_percent': X_data, 't_min': time_min_array})

#Concentration variation
for HCl0_exp, X_data in zip(Concentration_array, X_concentration_variation_array):
    if not (HCl0_exp == HCl0_temperature_variation and Temperature_concentration_variation == 323.0): #Is already in, is 323K run. So do not want to add it again.
        all_simulations_parameters.append({'T': Temperature_concentration_variation, 'HCl0': HCl0_exp, 'X_percent': X_data, 't_min': time_min_array})

#This makes sure that no runs are the same
unique_runs = {}
for simulation in all_simulations_parameters:
    uniquecheck = (simulation['T'], simulation['HCl0'])
    if uniquecheck not in unique_runs:
        unique_runs[uniquecheck] = simulation

all_simulations_parameters = list(unique_runs.values())
all_simulations_parameters.sort(key=lambda p: (p['T'], p['HCl0'])) #Runs are sorted. First by T, then HCL0.
dXdt_experimental_segments = []
X_experimental = []
time_seconds = []

for simrun in all_simulations_parameters:
    time_insec = simrun['t_min']*60
    dXdt_experimental_segments.append(np.gradient(simrun['X_percent'] /100, time_insec))
    X_experimental.extend(simrun['X_percent'] /100)
    time_seconds.extend(time_insec)

dXdt_experimental_everysegment = np.concatenate(dXdt_experimental_segments)
X_experimental_everysegment = np.array(X_experimental)
time_seconds_everysegment = np.array(time_seconds)


#######Kinetic model
def kinetic_model(t, y, A, Ea, n, m, T, HCl0):
    X, N_HCl = y
    k = A*np.exp(-Ea/(R*T)) #Arrhenius law
    C_HCl = max(N_HCl/V_m3, 1e-12) #HCl concentration in kmol/m^3
    C_Ni = max(C_Ni0*(1-X), 1e-12) #Ni concentration in kmol/m^3
    reaction_rate = k*C_HCl**n *C_Ni**m #Reaction rate in kmol/(m^3*s)
    dXdt = (V_m3/ N_Ni0)*reaction_rate
    dN_HCldt = -2*reaction_rate*V_m3 #Change in HCl
    return [dXdt, dN_HCldt]

#Calculation of DX/dt for all experiments
def simulate_dXdt_global(time_seconds_data, A, Ea, n, m):
    dXdt_model_allsims = []

    for simrunn in all_simulations_parameters:
        T = simrunn['T'] #K
        HCl0 = simrunn['HCl0'] #kmol/m^3
        t_eval =simrunn['t_min']*60 #s

        N_HCl0_run = HCl0*V_m3 #kmol of HCl initially there
        y0 = [0.0, N_HCl0_run] #Initial conditions

        solution_ode = solve_ivp(kinetic_model, [t_eval[0], t_eval[-1]], y0, args=(A, Ea, n, m, T, HCl0), t_eval=t_eval, method='LSODA', rtol=1e-5, atol=1e-8)
        X_sim = solution_ode.y[0]
        N_HCl_sim = solution_ode.y[1]
        k =A*np.exp(-Ea/(R*T)) #Calculate dX/dt at each point that is integrated

        for X, N_HCl in zip(X_sim, N_HCl_sim):
            C_HCl = max(N_HCl /V_m3, 1e-12)
            C_Ni = max(C_Ni0*(1-X), 1e-12)
            reaction_rate = k*C_HCl**n *C_Ni**m
            dXdt = (V_m3/N_Ni0)* reaction_rate
            dXdt_model_allsims.append(dXdt)

    return np.array(dXdt_model_allsims)


#######R^2
def R2_calculation(y_actual, y_prediction):
    sumsquared_res = np.sum((y_actual-y_prediction) ** 2)
    sumsquared_tot = np.sum((y_actual-np.mean(y_actual)) **2)
    if sumsquared_tot == 0:
        return 1
    return 1 - (sumsquared_res / sumsquared_tot)


#######Run global fit
# Initial guesses: [A, Ea (in kJ/mol), n, m]
initial_guesses = [4.0e-5, 20, 0.5, 1]
lower_boundaries = [1e-10, 0.1, 0.1, 0]
upper_boundaries = [1e20, 150, 3, 3]
print("Performing the curve fit now")

try:
    popt, pcov = curve_fit(simulate_dXdt_global, time_seconds_everysegment, dXdt_experimental_everysegment, p0=initial_guesses, bounds=(lower_boundaries, upper_boundaries), maxfev=10000, ftol=1e-8, xtol=1e-8)
    A_fit, Ea_fit, n_fit, m_fit = popt

    #Results
    print("Resulting fit parameters:")
    print(f"A = {A_fit:.3e}")
    print(f"Ea = {Ea_fit:.2f} kJ/mol")
    print(f"n = {n_fit:.3f}")
    print(f"m = {m_fit:.3f}")

    #Individual R^2 calculation
    X_simulation_data = []
    dXdt_simulation_all = simulate_dXdt_global(time_seconds_everysegment, A_fit, Ea_fit, n_fit, m_fit) #dX/dt using fit parameters
    segment_start_idx = 0

    #For loop for final ODE solution and R^2 values
    for simrun3 in all_simulations_parameters:
        T = simrun3['T']
        HCl0 =simrun3['HCl0']
        N_HCl0_run = HCl0*V_m3
        t_eval = simrun3['t_min']*60 #s
        y0 = [0, N_HCl0_run]

        solution_ODE = solve_ivp(kinetic_model, [t_eval[0], t_eval[-1]], y0, args=(A_fit, Ea_fit, n_fit, m_fit, T, HCl0), t_eval=t_eval, method='LSODA', rtol=1e-5, atol=1e-8) #ODE solver, gives X over time

        X_experimental_segment = simrun3['X_percent']/100
        X_simulation_segment = solution_ODE.y[0]
        N_HCl_simulation_segment = solution_ODE.y[1] #kmol
        C_HCl_simulatie = N_HCl_simulation_segment/V_m3 #kmol/m^3
        C_Ni_simulatie = C_Ni0*(1 -X_simulation_segment) #kmol/m^3

        #R2 for dX/dt for this segment
        number_of_points = len(t_eval)
        dXdt_experimental_segment = dXdt_experimental_everysegment[segment_start_idx: segment_start_idx + number_of_points]
        dXdt_simulation_segment = dXdt_simulation_all[segment_start_idx: segment_start_idx + number_of_points]
        R2_conversion_rate = R2_calculation(dXdt_experimental_segment, dXdt_simulation_segment)

        #R2 for X versus T for this segment
        R2_conversion = R2_calculation(X_experimental_segment, X_simulation_segment)
        segment_start_idx += number_of_points

        X_simulation_data.append({
            'T': T,
            'HCl0': HCl0,
            'label': f'{int(T)}K, {HCl0:.3g}M',
            't_min': simrun3['t_min'],
            'X_sim': X_simulation_segment,
            'C_HCl_sim': C_HCl_simulatie,
            'C_Ni_sim': C_Ni_simulatie,
            'R2_rate': R2_conversion_rate,
            'R2_conversion': R2_conversion
        })


except RuntimeError as e:
    print("Something went wrong")
    print("Check initial guesses (p0) or bounds for potential issues.")
    exit()


#######Plot function for conversion
def plot_conversion_subset(axis, simulation_data_subset, title_text, key_variable):
    axis.set_ylabel('Conversion (X)')
    axis.set_ylim(0, 1.05)
    axis.set_xlabel('Time (min)')
    axis.grid(True, linestyle='--', alpha=0.7)
    axis.set_title(r'Global Kinetic Model Fit: Conversion vs. Time (' + title_text + r')')

    key_values_plot_conversion = sorted(list(set([data[key_variable] for data in simulation_data_subset])))
    number_of_keys = len(key_values_plot_conversion)
    handles_list = []
    labels_list = []

    for index, data in enumerate(simulation_data_subset):
        experimental_run = next(r for r in all_simulations_parameters if r['T'] == data['T'] and r['HCl0'] == data['HCl0'])
        X_experimental = experimental_run['X_percent']/100

        label = data['label']
        time_in_minutes = data['t_min']

        color_idnumber = key_values_plot_conversion.index(data[key_variable])
        color = plt.colormaps.get_cmap('Dark2')(color_idnumber / max(1, number_of_keys - 1) if number_of_keys > 1 else 0.5)
        axis.plot(time_in_minutes, X_experimental, 'o', markersize=6, alpha=0.6, color=color)
        model_line, = axis.plot(time_in_minutes, data['X_sim'], '-', linewidth=2.5, color=color)
        handles_list.append(model_line)
        labels_list.append(label)

    experimental_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, linestyle='')
    handles_list.insert(0, experimental_handle)
    labels_list.insert(0, 'Experimental Data')
    axis.legend(handles_list, labels_list, title="Legend", loc='lower right', ncol=1, fontsize='small')


#Plot function for concentration of HCl and Ni versus time
def plot_concentrations_subset(axis_HCl, axis_Ni, sim_data_subset, title_tekst, key_variab):
    axis_HCl.set_title('Model Predictions: Concentration versus Time (' + title_tekst + r')')
    axis_HCl.set_ylabel('Acid Concentration HCl in M)', color='blue')
    axis_HCl.tick_params(axis='y', labelcolor='blue')
    axis_HCl.grid(True, linestyle='--', alpha=0.7, axis='x')

    axis_Ni.set_xlabel('Time (min)')
    axis_Ni.set_ylabel('Solid Ni Concentration Ni in kmol\m^3)', color='red')
    axis_Ni.tick_params(axis='y', labelcolor='red')
    axis_Ni.set_ylim(0, C_Ni0*1.05)

    key_values_conc = sorted(list(set([data[key_variab] for data in sim_data_subset])))
    number_of_keys = len(key_values_conc)
    handles_HCl = []
    HCl_labels = []
    Ni_handles = []
    Ni_labels = []

    #If anything is wrong with what shows, it is down here?
    for index, data in enumerate(sim_data_subset):
        label = data['label']
        t_min = data['t_min']
        color_idx = key_values_conc.index(data[key_variab])
        color = plt.colormaps.get_cmap('Dark2')(color_idx / max(1, number_of_keys - 1) if number_of_keys > 1 else 0.5)
        line_HCl, = axis_HCl.plot(t_min, data['C_HCl_sim'], linewidth=2.5, color=color, linestyle='-')
        handles_HCl.append(line_HCl)
        HCl_labels.append(r' $[\mathrm{HCl}]$: ' + label)
        line_Ni, = axis_Ni.plot(t_min, data['C_Ni_sim'], linewidth=2.5, color=color, linestyle='--')
        Ni_handles.append(line_Ni)
        Ni_labels.append(r' $[\mathrm{Ni}]$: ' + label)

    handles_full_list = handles_HCl + Ni_handles
    labels_full_list = HCl_labels + Ni_labels
    axis_HCl.legend(handles_full_list, labels_full_list, title="Legend", loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize='small')


#######Plot: dX/dt versus time
figure1, axis1 = plt.subplots(figsize=(10, 7))

axis1.set_ylabel('Conversion Rate dX\dt * 1000 1\s', fontsize = 18)
axis1.set_ylim(bottom=0)
axis1.set_xlabel('Time (min)', fontsize = 18)
axis1.grid(True, linestyle='--', alpha=0.7)
axis1.set_title('Global Kinetic Model Fit: Rate dX\dt versus time')
axis1.tick_params(axis='y', labelsize=14)
axis1.tick_params(axis='x', labelsize=14)

number_runs = len(all_simulations_parameters)
handles = []
labels = []
segment_start_idxdir = 0

for k, data in enumerate(X_simulation_data):
    t_min = data['t_min']
    label = data['label']
    R2_rate = data['R2_rate']

    number_of_points = len(t_min)
    dXdt_experimental_segment = dXdt_experimental_everysegment[segment_start_idxdir: segment_start_idxdir + number_of_points]
    dXdt_simulation_segment = dXdt_simulation_all[segment_start_idxdir: segment_start_idxdir + number_of_points]
    segment_start_idxdir += number_of_points

    color = plt.colormaps.get_cmap('Spectral')(k / number_runs)
    axis1.plot(t_min, dXdt_experimental_segment * 1000, 'o', markersize=6, alpha=0.6, color=color)
    model_line, = axis1.plot(t_min, dXdt_simulation_segment * 1000, '-', linewidth=2.5, color=color)
    labels.append(f"{label} (R²={R2_rate:.4f})")
    handles.append(model_line)

experimental_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, linestyle='')
handles.insert(0, experimental_handle)
labels.insert(0, 'Experimental Data')
axis1.legend(handles, labels, loc='upper right', ncol=1, fontsize=16, title_fontsize = 18)
plt.tight_layout()
plt.show()


#######Plot: X versus time, T dependence
T_runss_sim = [data for data in X_simulation_data if data['HCl0'] == HCl0_temperature_variation]

figure2, axis2 = plt.subplots(figsize=(10, 7))
plot_conversion_subset(axis2, T_runss_sim, 'Temperature Dependence, HCl molarity is 1 M', 'T')
plt.tight_layout()
plt.show()


#######Plot: X versus time, HCl dependence
Concentrations_runs_sim = [data for data in X_simulation_data if data['T'] == Temperature_concentration_variation] #Only use runs where T = 323K

figure3, axis3 = plt.subplots(figsize=(10, 7))
plot_conversion_subset(axis3, Concentrations_runs_sim, 'HCl Concentration Dependence, T = 323K', 'HCl0')
plt.tight_layout()
plt.show()


#######Plot: R^2 bar chart for dX/dt
figure4, axis4 = plt.subplots(figsize=(12, 7))

run_labels = [data['label'] for data in X_simulation_data]
R2_values_for_rateofDXdt = [data['R2_rate'] for data in X_simulation_data]
colors = plt.colormaps.get_cmap('Spectral')(np.linspace(0, 1, len(run_labels)))

bars_for_plot = axis4.bar(run_labels, R2_values_for_rateofDXdt, color=colors, alpha=0.8)

axis4.set_title('R^2 for dX\dt per individual experiment')
axis4.set_xlabel('Experiment Condition (Temperature, Initial HCl concentration)')
axis4.set_ylabel('R^2')
y_min = 0.65
axis4.set_ylim(y_min, 1.005)

for bar in bars_for_plot:
    y_value = bar.get_height()
    axis4.text(bar.get_x() + bar.get_width() / 2, y_value + 0.002, f'{y_value:.4f}', ha='center', va='bottom', fontsize=9, rotation=45, color='black')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


#######Plot: R^2 bar chart for X
figure5, axis5 = plt.subplots(figsize=(12, 7))
R2_values_fortheconversion = [data['R2_conversion'] for data in X_simulation_data]
colors = plt.colormaps.get_cmap('Spectral')(np.linspace(0, 1, len(run_labels)))  # Use same colors
bars_secondR2plot = axis5.bar(run_labels, R2_values_fortheconversion, color=colors, alpha=0.8)

axis5.set_title('R^2 for X per individual experiment')
axis5.set_xlabel('Experiment Condition (Temperature, Initial HCl concentration)')
axis5.set_ylabel('R^2')
axis5.set_ylim(top=1.0005)

for bar in bars_secondR2plot:
    y_value = bar.get_height()
    axis5.text(bar.get_x() + bar.get_width() / 2, y_value + 0.00005, f'{y_value:.4f}', ha='center', va='bottom', fontsize=9, rotation=45, color='black')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


#######Plot: Concentration versus time. varying temperature
Temp_runs_sim = [data for data in X_simulation_data if data['HCl0'] == HCl0_temperature_variation]
figure6, axis_HCl = plt.subplots(figsize=(12, 7))
axis_Ni = axis_HCl.twinx() #Axis for Ni concentration
plot_concentrations_subset(axis_HCl, axis_Ni, Temp_runs_sim,'Concentrations versus time at varying Temperatures', 'T')
plt.subplots_adjust(bottom=0.3)
plt.show()


#######Plot: Concentration versus time, varying concentration
Concentrat_runs_sim = [data for data in X_simulation_data if data['T'] == Temperature_concentration_variation] #Only 323K runs, so you get only HCl variation
figure7, axis_HCl_7 = plt.subplots(figsize=(12, 7))
axis_Ni_7 = axis_HCl_7.twinx() #Axis for Ni concentration
plot_concentrations_subset(axis_HCl_7, axis_Ni_7, Concentrat_runs_sim,'Concentrations versus time at varying initial HCl concentrations', 'HCl0')
plt.subplots_adjust(bottom=0.3)
plt.show()