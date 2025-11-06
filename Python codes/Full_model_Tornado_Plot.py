import numpy as np
from Demos.BackupSeek_streamheaders import stream_name
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from CodeLibrary import Simulation
from tabulate import tabulate
from matplotlib.ticker import FuncFormatter

#instanciate the class and set the aspen name, file path and visibility
sim = Simulation(AspenFileName= "Final_Aspen_model.bkp", WorkingDirectoryPath= r"c:\Users\Gebruiker\Documents\Studie_master_thesis\Aspen-Python\Python_Aspen\Final_model" ,VISIBILITY=True)

###Techno-economics import

from Reactor_modelling.Final_Python_Models.Techno_economics import cstr_towler_2022
from Reactor_modelling.Final_Python_Models.Techno_economics import rotary_dryer_woods
from Reactor_modelling.Final_Python_Models.Techno_economics import rotary_dryer_towler_2022
from Reactor_modelling.Final_Python_Models.Techno_economics import preliminary_treatment_unit_woods
from Reactor_modelling.Final_Python_Models.Techno_economics import hydrocyclone_woods
from Reactor_modelling.Final_Python_Models.Techno_economics import filter_towler_2022
from Reactor_modelling.Final_Python_Models.Techno_economics import washer_mixersettler_woods
from Reactor_modelling.Final_Python_Models.Techno_economics import single_stage_centrifugal_pump_towler_2010
from Reactor_modelling.Final_Python_Models.Techno_economics import steam_boiler_lbnl
from Reactor_modelling.Final_Python_Models.Techno_economics import air_filter_woods
from Reactor_modelling.Final_Python_Models.Techno_economics import heater_plateframe_towler_2010
from Reactor_modelling.Final_Python_Models.Techno_economics import slurry_heater_floating_head_shell_and_tube_towler_2010
from Reactor_modelling.Final_Python_Models.Techno_economics import capital_cost_calculation
from Reactor_modelling.Final_Python_Models.Techno_economics import opex_variable_production_cost
from Reactor_modelling.Final_Python_Models.Techno_economics import opex_fixed_production_costs
from Reactor_modelling.Final_Python_Models.Techno_economics import working_capital
from Reactor_modelling.Final_Python_Models.Techno_economics import cash_flow_calculation
from Reactor_modelling.Final_Python_Models.Techno_economics import npv_calculation
from Reactor_modelling.Final_Python_Models.Techno_economics import create_cash_flow_table
from Reactor_modelling.Final_Python_Models.Techno_economics import calculate_levelized_cost
from Reactor_modelling.Final_Python_Models.Techno_economics import payback_time_calculation
from Reactor_modelling.Final_Python_Models.Techno_economics import calculate_irr
from Reactor_modelling.Final_Python_Models.Techno_economics import roi_calculation
from Reactor_modelling.Final_Python_Models.Techno_economics import air_cooler_floating_head_shell_and_tube_towler_2010


#############################################################Inputs

#CNF Feed
CNF_Feed_Temperature = 600 #*C, is an assumption based on the temperature in a CMP reactor
CNF_Feed_Pressure = 1 #atm
CNF_Feed_TotalFlow = 3470 #kg/hr

#CNFFEED stream
sim.STRM_CISOLID_Set_Temperature(Streamname="CNFFEED", Temp=CNF_Feed_Temperature)
sim.STRM_CISOLID_Set_Pressure(Streamname="CNFFEED", Pressure=CNF_Feed_Pressure)
sim.STRM_CISOLID_Set_TotalFlowRate("CNFFEED", CNF_Feed_TotalFlow)


#HCL mixture feed
HCL_Feed_Temperature = 20 #*C
HCL_Feed_Pressure = 1 #atm
AcidSolution_Multiplier = 10 #Indicates how much L of Acid solution there is in the feed per kg of CNF in the feed
Acid_Feed_Lflow = CNF_Feed_TotalFlow*AcidSolution_Multiplier #L/hr

#HCLFEED stream
sim.STRM_Set_Temperature(Streamname="HCLFEED", Temp=HCL_Feed_Temperature)
sim.STRM_Set_Pressure(Streamname="HCLFEED", Pressure=HCL_Feed_Pressure)
sim.STRM_Set_TotalFlowRate(Streamname="HCLFEED", TotalFlowRate=Acid_Feed_Lflow)


#General input
plant_hours_per_year = 8000

#CSTR Reactor input code
CSTR_Temperature = 80
sim.BLK_CISTR_Set_Pressure("CSTR", 1)
sim.BLK_CISTR_Set_Temperature("CSTR", CSTR_Temperature)
sim.BLK_CISTR_Set_Specification_type("CSTR", "RES-TIME")
sim.BLK_CISTR_Set_ResidenceTime("CSTR", 3)
T_steam_CSTR_heating_in = 200
T_steam_CSTR_heating_out = 110

#Solids filter 1 and 2 input
submergence_factor = 0.5
filtration_rate = 15 #lb/(ft^2 hr). From Filters and Filtration Handbook, fifth edition. Page 119, value for calcium carbonate.

#Washer input
Washer_water_added = CNF_Feed_TotalFlow #kg/hr. Initially, 2000kg/hr of water with 2283kg/hr CNFFeed results in 1827kg/hr CNF and 456kg/hr of acid in at washer. So decided to make water added just equal to CNF Feed flow.
sim.STRM_Set_ComponentFlowRate("WASLIQIN", Washer_water_added, "H2O")

#Solids cooler input
T_water_in_cooler = 20 #degrees Celsius, is an estimation
Cp_water = 4.184 #kJ/kgK
U_cooler = 150 #W/m^2K #Trojosky 2019
sim.BLK_HEATER_Set_Temperature("CNFCOOLE", CSTR_Temperature) #Is set equal to CSTR temperature
sim.BLK_HEATER_Set_Pressure("CNFCOOLE", 1)

#Slurry heater input
U_slurry_heater = 100 #W/m2K Based on industrial handbook for industrial drying, 2nd edition, page 188
T_steam_slurry_heater_in = 200 #*C, is at 15 bar
T_steam_slurry_heater_out = 110 #*C

#Dryer / slurry heater input
RH_air_in = 0.77 #Average RH of atmospheric air in the Netherlands
p_g_air_in = 2.3392 #kPa, at 20*C. https://www.quadco.engineering/en/know-how/cfd-calculate-water-fraction-humid-air.htm?utm_source=chatgpt.com
RH_dry_air_out = 0.8 #RH of air leaving the dryer
p_g_dry_air_out = 101.42 #kPa, https://www.quadco.engineering/en/know-how/cfd-calculate-water-fraction-humid-air.htm?utm_source=chatgpt.com
p_dryer = 101.325 #kPa
rho_water_100C = 958 #kg/m^3, http://www.bioconsult.ch/Inovatech/W-Lehre/J%20Che%20Eng%20Dat20,%2097.pdf
rho_air_100C = 1.92 #kg/m^3, from Morgan and Shapiro 7th edition, Table A-22
rho_carbon_100C = 2100 #kg/m^3, for graphitized CNF. https://www.us-nano.com/inc/sdetail/984

#Dry air heater input
T_dry_air_heater_in = 20 #degrees Celsius
p_dry_air_in = 101.325 #kPa
Cp_steam = 2010 #J/kgK
T_steam_air_heater_in = 200 #degrees Celsius
T_steam_air_heater_out = 110 #degrees Celsius
U_air_heater = 50 #W/m^2*K, 30-300 for steam to air in finned tubes, based on air side surface area. 400-4000 when based on steam side surface area. So I used 50.

#Steam generator input
h_g_water_saturated_vapor_200 = 2793.2 #kJ/kgK, from Morgan and Shapiro, 7th edition, Table A-2
h_f_water_saturated_liquid_20 = 83.96 #kJ/kg, From Morgan and Shapiro, 7th edition, Table A-2
#The enthalpy of saturated vapor at a given T already includes both the sensible heat and the latent heat of vaporization.
eta_electrical = 0.98 #efficiency of the electric steam generator, https://link.springer.com/article/10.3103/S1068371218070106

#Extra input for dry air
p_air_normal = 101.325 #kPa
T_air_normal = 0 #degrees celsius
rho_air_20C = 1.204 #kg/m^3, https://www.engineeringtoolbox.com/air-density-specific-weight-d_600.html

#Input for air cooler
Cp_water_vapor = 1.9 #kJ/kgK
Cp_hcl_vapor = 0.8 #kJ/kgK
hcl_dissolve_heat = 2100 #kJ/kg
T_water_in_air_cooler = 20 #Celsius
U_air_cooler = 50 #W/m^2K, from slides by Hooman. water to air in finned tubes, based on air-side surface area.

#Input for pump calculations
rho_water_20C = 1000 #kg/m^3

################################################################Dryer calculations and input
p_sat = 2.3393*10**3 #at 20 *C, from engineeringtoolbox
p_v = RH_air_in*p_sat
p_atm = 101325 #Pa
Humidity_air_dryer_in = 0.622*p_v/(p_atm - p_v) #Absolute humidity, kg water / kg air
T_air_dryer_in = 180 #*C
Humidity_air_dryer_out = 0.05 #kg water per kg air
Dryer_safety_factor = 1.5
Cp_carbon = 0.7 #kJ/kgK
Cp_air = 1 #kJ/kgK
latent_heat_of_vaporization = 2260 #kJ/kg
Tout_dryer = 100 #*C

massflow_solid_slurry_dryer_in = 0.7255*CNF_Feed_TotalFlow
massflow_water_slurry_dryer_in = 0.25*massflow_solid_slurry_dryer_in
massflow_dry_air_required = massflow_water_slurry_dryer_in / (Humidity_air_dryer_out - Humidity_air_dryer_in) * Dryer_safety_factor
sim.STRM_Set_ComponentFlowRate("DRYAIR", massflow_dry_air_required, "AIR")

##########################################################################Running the simulation

cnf_selling_price = 25000 ## Between 25.000 and 113.000 USD per tonne in 2017
dust_waste_price = 20 #USD/tonne

#Dictionary for economic parameters
parameters_dict = {
    'Electricity price': 0.10,
    'Water price': 0.10,
    'Operator salary': 3850,
    'Interest rate': 10,
    'HCl buying price': 112.03,
    'Project lifetime': 20,
    'Plant operating hours per year': 8000,
    'Fixed capital costs variance factor': 1,
    'Fixed production costs variance factor': 1,
    'Acid waste price': 316
}



def full_calculation(parameters):

    electricity_price = parameters['Electricity price']
    water_price = parameters['Water price']
    operator_salary = parameters['Operator salary']
    interest_rate = parameters['Interest rate']/100 #Go from percentage (e.g. 10%) to 0.10. Is the number you need.
    hcl_buying_price = parameters['HCl buying price']
    project_lifetime = parameters['Project lifetime']
    plant_hours_per_year = parameters['Plant operating hours per year']
    fixed_capital_costs_variance_factor = parameters['Fixed capital costs variance factor']
    fcop_variance_factor = parameters['Fixed production costs variance factor']
    acid_waste_price = parameters['Acid waste price']

    sim.Run()
    sim.DialogSuppression(TrueOrFalse=True)


    test_rate = sim.STRM_CISOLID_Get_MassFlowPerCompound("CNFFEED", "CARBON")
    test_rate_acid = sim.STRM_Get_MassFlowPerCompound("HCLFEED", "HCL")
    test_rate_washerin = sim.STRM_Get_MassFlowPerCompound("WASLIQIN", "H2O")

    #HCLFEED
    Acid_Feed_HCL_massflow = sim.STRM_Get_MassFlowPerCompound("HCLFEED", "HCL")
    Acid_Feed_H2O_massflow = sim.STRM_Get_MassFlowPerCompound("HCLFEED", "H2O")
    Acid_Feed_total_massflow = Acid_Feed_HCL_massflow + Acid_Feed_H2O_massflow #kg/hr


    ###LMTD calculations of the Solids cooler
    T_water_out_cooler = sim.STRM_CISOLID_Get_Temperature("CNFFEEDC") - 5  # degrees Celsius, estimated to be 5*C lower than T of solid leaving the cooler
    Q_hot_cooler = 1163 * sim.BLK_HEATER_Get_HeatDuty("CNFCOOLE")  # In kW, times 1163 to go from Gcal/hr to kW
    Q_cold_cooler = -Q_hot_cooler
    massflow_water_cooler = (Q_cold_cooler) / (Cp_water * (T_water_out_cooler - T_water_in_cooler)) * 3600  # kg/hr, *3600 from s to hr

    CNF_Feed_Cooled_Temperature = sim.STRM_CISOLID_Get_Temperature("CNFFEEDC")

    Delta_T1_cooler = CNF_Feed_Temperature - T_water_in_cooler
    Delta_T2_cooler = CNF_Feed_Cooled_Temperature - T_water_out_cooler
    Delta_T_lm_cooler = (Delta_T1_cooler - Delta_T2_cooler) / (np.log(Delta_T1_cooler / Delta_T2_cooler))

    Area_cooler = Q_cold_cooler*1000 / (U_cooler * Delta_T_lm_cooler)  # m^2. '-' because the Q from Aspen is a negative number

    # Calculations for the CSTR
    CSTR_volume = sim.BLK_RCSTR_Get_ReactorVolume("CSTR")
    CSTR_duty_Gcalhr = sim.BLK_RCSTR_Get_HeatDuty("CSTR")
    CSTR_duty_kW = 1163*CSTR_duty_Gcalhr
    massflow_steam_required_CSTR = CSTR_duty_kW * 1000 / (Cp_steam * (T_steam_CSTR_heating_in - T_steam_CSTR_heating_out)) * 3600  #kg/hr. *1000 to J. Cp is in J/kgKkgK

    Q_steam_generation_CSTR_heating = h_g_water_saturated_vapor_200 - h_f_water_saturated_liquid_20  # kJ/kg
    Qdot_steam_generation_CSTR_heating = Q_steam_generation_CSTR_heating * massflow_steam_required_CSTR / 3600 #kJ/s (kW)
    E_electrical_steam_generation_CSTR_heating = Qdot_steam_generation_CSTR_heating / eta_electrical
    E_electrical_steam_generation_CSTR_heating_kWh = E_electrical_steam_generation_CSTR_heating * plant_hours_per_year
    E_electrical_steam_generation_CSTR_heating_BTUperhr = E_electrical_steam_generation_CSTR_heating * 3412.142 #BTU/hr
    E_electrical_steam_generation_CSTR_heating_MMBTUperhr = E_electrical_steam_generation_CSTR_heating / 1000000 #MMBTU/hr


    # Calculations for the separator
    volumeflow_mixed_separator_in = sim.STRM_Get_VolumeFlow("CSTROUT")  # m^3/hr
    volumeflow_solid_separator_in = sim.STRM_CISOLID_Get_VolumeFlow("CSTROUT")  # m^3/hr
    volumeflow_total_separator_in = volumeflow_mixed_separator_in + volumeflow_solid_separator_in  # m^3/hr
    volumeflow_total_separator_in_Ls = volumeflow_total_separator_in * 1000 / 3600  # L/s


    # Calculations for the Hydrocyclone
    volumeflow_mixed_hydrocyclone_in = sim.STRM_Get_VolumeFlow("SEPBOT")  # m^3/hr
    volumeflow_solid_hydrocyclone_in = sim.STRM_CISOLID_Get_VolumeFlow("SEPBOT")  # m^3/hr
    volumeflow_total_hydrocyclone_in = volumeflow_mixed_hydrocyclone_in + volumeflow_solid_hydrocyclone_in  # m^3/hr
    volumeflow_total_hydrocyclone_in_Ls = volumeflow_total_hydrocyclone_in * 1000 / 3600  # L/s


    # Size calculations for solids filter 1 and solids filter 2
    filtration_rate_kg = filtration_rate * 4.882427636  # kg/(m^2hr). Conversion from lb/(ft^2 hr) to this.

    massflow_H2O_filter1_removed = sim.STRM_Get_MassFlowPerCompound("FILTLIQ", "H2O")  # kg/hr
    massflow_HCL_filter1_removed = sim.STRM_Get_MassFlowPerCompound("FILTLIQ", "HCL")  # kg/hr
    massflow_CL_filter1_removed = sim.STRM_Get_MassFlowPerCompound("FILTLIQ", "CL-")  # kg/hr
    massflow_NI_filter1_removed = sim.STRM_Get_MassFlowPerCompound("FILTLIQ", "NI++")  # kg/hr
    massflow_liquid_removed_filter1 = massflow_H2O_filter1_removed + massflow_HCL_filter1_removed + massflow_CL_filter1_removed + massflow_NI_filter1_removed  # kg/hr
    filter1_area = massflow_liquid_removed_filter1 / filtration_rate_kg / submergence_factor  # m^2

    massflow_H2O_filter2_removed = sim.STRM_Get_MassFlowPerCompound("FILT2LIQ", "H2O")  # kg/hr
    massflow_HCL_filter2_removed = sim.STRM_Get_MassFlowPerCompound("FILT2LIQ", "HCL")  # kg/hr
    massflow_CL_filter2_removed = sim.STRM_Get_MassFlowPerCompound("FILT2LIQ", "CL-")  # kg/hr
    massflow_NI_filter2_removed = sim.STRM_Get_MassFlowPerCompound("FILT2LIQ", "NI++")  # kg/hr
    massflow_liquid_removed_filter2 = massflow_H2O_filter2_removed + massflow_HCL_filter2_removed + massflow_CL_filter2_removed + massflow_NI_filter2_removed  # kg/hr
    filter2_area = massflow_liquid_removed_filter2 / filtration_rate_kg / submergence_factor  # m^2


    # Calculations for the washer
    volumeflow_mixed_washer_in = sim.STRM_Get_VolumeFlow("FILTSOL")  # m^3/hr
    volumeflow_solid_washer_in = sim.STRM_CISOLID_Get_VolumeFlow("FILTSOL")  # m^3/hr
    volumeflow_total_washer_in = volumeflow_mixed_washer_in + volumeflow_solid_washer_in  # m^3/hr
    volumeflow_total_washer_in_Ls = volumeflow_total_washer_in * 1000 / 3600  # L/s

    volumeflow_water_added_washer = sim.STRM_Get_VolumeFlow("WASLIQIN")  # m^3/hr


    ###Slurry heater calculations and steam generation requirements for slurry heater
    Q_slurry_heater = 1163 * sim.BLK_HEATER_Get_HeatDuty("SOLHEATE")  # In kW, times 1163 to go from Gcal/hr to kW
    massflow_steam_slurry_heater = (Q_slurry_heater * 1000) / (Cp_steam * (T_steam_slurry_heater_in - T_steam_slurry_heater_out)) * 3600  # kg/hr, *1000 for kW to W, *3600 from s to hr
    Q_steam_generation_slurry_heater = h_g_water_saturated_vapor_200 - h_f_water_saturated_liquid_20  # kJ/kg
    Qdot_steam_generation_slurry_heater = Q_steam_generation_slurry_heater * massflow_steam_slurry_heater / 3600  # kJ/s (kW)
    E_electrical_steam_generation_slurry_heater = Qdot_steam_generation_slurry_heater / eta_electrical
    E_electrical_steam_generation_slurry_heater_kWh = E_electrical_steam_generation_slurry_heater * plant_hours_per_year  # Times operating hours to get kWh consumed per year
    E_electrical_steam_generation_slurry_heater_BTUperhr = E_electrical_steam_generation_slurry_heater * 3412.142  # BTU/hr
    E_electrical_steam_generation_slurry_heater_MMBTUperhr = E_electrical_steam_generation_slurry_heater_BTUperhr / 1000000  # MMBTU/hr


    ###LMTD calculations of the slurry heater
    T_slurry_heater_out = sim.STRM_CISOLID_Get_Temperature("DRYERIN")
    T_slurry_heater_in = sim.STRM_CISOLID_Get_Temperature("SEP2SOL")
    Delta_T1_slurry_heater = T_steam_slurry_heater_in - T_slurry_heater_in
    Delta_T2_slurry_heater = T_steam_slurry_heater_out - T_slurry_heater_out
    Delta_T_lm_slurry_heater = (Delta_T1_slurry_heater - Delta_T2_slurry_heater) / (np.log(Delta_T1_slurry_heater / Delta_T2_slurry_heater))
    Area_slurry_heater = Q_slurry_heater * 1000 / (U_slurry_heater * Delta_T_lm_slurry_heater)  # m^2, size of the slurry heater heat exchanger


    ###Size calculations for the dryer
    massflow_carbon_dryer_in = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYERIN", "CARBON")
    massflow_SI_dryer_in = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYERIN", "SI")
    massflow_NI_dryer_in = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYERIN", "NI")
    massflow_MG_dryer_in = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYERIN", "MG")
    massflow_AL_dryer_in = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYERIN", "AL")
    massflow_FE_dryer_in = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYERIN", "FE")
    massflow_CA_dryer_in = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYERIN", "CA")
    massflow_solid_total_dryer_in = massflow_carbon_dryer_in + massflow_SI_dryer_in + massflow_NI_dryer_in + massflow_MG_dryer_in + massflow_AL_dryer_in + massflow_FE_dryer_in + massflow_CA_dryer_in

    massflow_water_dryer_in = sim.STRM_Get_MassFlowPerCompound("DRYERIN", "H2O")
    massflow_HCL_dryer_in = sim.STRM_Get_MassFlowPerCompound("DRYERIN", "HCL")
    massflow_liquid_total_dryer_in = massflow_water_dryer_in + massflow_HCL_dryer_in

    volumeflow_slurry_solid_dryer_in = massflow_solid_total_dryer_in / rho_carbon_100C
    volumeflow_slurry_liquid_dryer_in = massflow_liquid_total_dryer_in / rho_water_100C
    volumeflow_dry_air_dryer_in = massflow_dry_air_required / rho_air_100C
    volumeflow_total_dryer_in = volumeflow_slurry_solid_dryer_in + volumeflow_slurry_liquid_dryer_in + volumeflow_dry_air_dryer_in
    volume_dryer = volumeflow_total_dryer_in  # m^3 (per hr actually), estimation. #Assume 1 hr residence time of liquid, solid and air.
    diameter_dryer = (4 * volume_dryer / (5 * np.pi)) ** (1 / 3)  # m, if L/D ratio is 5. See Article by Mujumdar: CLASSIFICATION AND SELECTION OF INDUSTRIAL DRYERS. Page 40 and Dryer by Murugan et al
    Area_dryer = np.pi * diameter_dryer ** 2 / 4  # m^2, area of the rotary dryer


    ###LMTD calculations of the dry air heater
    T_air_heater_out = sim.STRM_Get_Temperature("DRYAIRHE")  # degrees Celsius
    T_air_heater_in = T_dry_air_heater_in  # degrees Celsius

    Q_cold_air_heater = 1163 * sim.BLK_HEATER_Get_HeatDuty("AIRHEATE")  # In kW, times 1163 to go from Gcal/hr to kW
    Q_hot_air_heater = Q_cold_air_heater
    massflow_steam_air_heater = (Q_hot_air_heater * 1000) / (Cp_steam * (T_steam_air_heater_in - T_steam_air_heater_out)) * 3600  # kg/hr, *1000 for kW to W, *3600 from s to hr
    Delta_T1_air_heater = T_steam_air_heater_in - T_air_heater_in
    Delta_T2_air_heater = T_steam_air_heater_out - T_air_heater_out
    Delta_T_lm_air_heater = (Delta_T1_air_heater - Delta_T2_air_heater) / (np.log(Delta_T1_air_heater / Delta_T2_air_heater))

    Area_air_heater = (Q_cold_air_heater * 1000) / (U_air_heater * Delta_T_lm_air_heater)  # m^2, *1000 to convert the Q from kW to W #U is W/m2K


    ###Calculations for steam generation for dry air heater
    Q_steam_generation_air_heater = h_g_water_saturated_vapor_200 - h_f_water_saturated_liquid_20  # kJ/kg
    Qdot_steam_generation_air_heater =  Q_steam_generation_air_heater * massflow_steam_air_heater / 3600  # kJ/s (kW), /3600 to go from hr to s
    # Would need a pressure of 2.343 bar for the steam to get saturated vapor at 125*C
    E_electrical_air_heater = Qdot_steam_generation_air_heater / eta_electrical
    E_electrical_air_heater_kWh = E_electrical_air_heater * plant_hours_per_year  # Times operating hours to get kWh consumed per year
    E_electrical_air_heater_BTUperhr = E_electrical_air_heater * 3412.142  # BTU/hr
    E_electrical_air_heater_MMBTUperhr = E_electrical_air_heater_BTUperhr / 1000000  # MMBTU/hr

    ###Calculations for steam boiler sizing
    steam_boiler_rating = E_electrical_air_heater_MMBTUperhr + E_electrical_steam_generation_slurry_heater_MMBTUperhr + E_electrical_steam_generation_CSTR_heating_MMBTUperhr # MMBTU/hr
    steam_boiler_electricity_consumption = E_electrical_air_heater_kWh + E_electrical_steam_generation_slurry_heater_kWh + E_electrical_steam_generation_CSTR_heating_kWh #kWh


    ###Calculations for the dry air filter
    massflow_dry_air_in = sim.STRM_Get_VolumeFlow("DRYAIR")
    flow_dry_air_volumetric = massflow_dry_air_in / rho_air_20C  # in m^3/hr
    flow_dry_air_volumetric_pers = flow_dry_air_volumetric / 3600
    flow_dry_air_Nm3perhr = flow_dry_air_volumetric * (p_dry_air_in / p_air_normal) * ((T_air_normal + 273.15) / (T_dry_air_heater_in + 273.15))  # mass flow in Nm^3/hr
    flow_dry_airNm3pers = flow_dry_air_Nm3perhr / 3600  # Nm^3/s, needed for the air filter sizing

    dust_concentration_in_air = 15 * 10 ** -9  # 15 micrograms/m^3. So 15*10^-9 kg per m^3. From Table 0.1 in https://www.who.int/publications/i/item/9789240034228
    dust_waste_flow = flow_dry_air_volumetric * dust_concentration_in_air * plant_hours_per_year / 1000  # tonnes/year. Go to kg/hr to kg/year to tonne/year


    ###Calculations for dry air cooler
    T_dewpoint = sim.BLK_DRYER_Get_DewPoint("DRYER") #degrees Celsius
    T_air_dryer_out = sim.STRM_Get_Temperature("USEDAIR") #degrees Celsius
    Massflow_air_dryer_out = sim.STRM_Get_MassFlowPerCompound("USEDAIR", "AIR")  # kg/hr
    Massflow_water_dryer_out = sim.STRM_Get_MassFlowPerCompound("USEDAIR", "H2O")  # kg/hr
    Massflow_hcl_dryer_out = sim.STRM_Get_MassFlowPerCompound("USEDAIR", "HCL")  # kg/hr
    Qdot_sensible_air = Massflow_air_dryer_out * Cp_air * (T_air_dryer_out - T_dewpoint) #kJ/hr
    Qdot_sensible_water = Massflow_water_dryer_out * Cp_water_vapor * (T_air_dryer_out - T_dewpoint) #kJ/hr
    Qdot_sensible_hcl = Massflow_hcl_dryer_out * Cp_hcl_vapor * (T_air_dryer_out - T_dewpoint) #kJ/hr
    latent_heat_of_condensation = latent_heat_of_vaporization
    Qdot_latent_water = Massflow_water_dryer_out * latent_heat_of_condensation #kJ/hr
    Qdot_dissolve_hcl = Massflow_hcl_dryer_out * hcl_dissolve_heat #kJ/hr
    Qdot_air_cooler = Qdot_sensible_air + Qdot_sensible_water + Qdot_sensible_hcl + Qdot_latent_water + Qdot_dissolve_hcl #kJ/hr
    T_water_out_air_cooler = T_dewpoint - 5
    Massflow_water_air_cooler = Qdot_air_cooler / (Cp_water * (T_water_in_air_cooler - T_water_out_air_cooler))

    Delta_T1_air_cooler = T_air_dryer_out - T_water_in_air_cooler
    Delta_T2_air_cooler = T_dewpoint - T_water_out_air_cooler
    Delta_T_lm_air_cooler = (Delta_T1_air_cooler - Delta_T2_air_cooler) / (np.log(Delta_T1_air_cooler / Delta_T2_air_cooler))
    Area_air_cooler = (Qdot_air_cooler * 1000 / 3600) / (U_air_cooler * Delta_T_lm_air_cooler)  # m^2, *1000 to convert the Q from kW to W #U is W/m2K

    Massflow_acid_waste_condensor = Massflow_water_dryer_out + Massflow_hcl_dryer_out #kg/hr, assumption that all liquid is removed


    ###Calculations for pump sizing
    pump_massflow = massflow_steam_air_heater + massflow_steam_slurry_heater + massflow_steam_required_CSTR # kg/hr
    pump_massflow_Ls = pump_massflow / 3600 / rho_water_20C * 1000  # L/s, 3600 for kg/hr to kg/s, /rho to m^3, *1000 to dm^3 and thus liter


    ##Total water consumption
    water_consumption_yearly = pump_massflow_Ls / 1000 * 60 * 60 * plant_hours_per_year + volumeflow_water_added_washer * plant_hours_per_year + (massflow_water_cooler + Massflow_water_air_cooler)/rho_water_20C*plant_hours_per_year # m^3/year
    #Water used in the HCl mixture not accounted for here, as the price of that is accounted for in the HCl price, which is actually the price for a HCl mixture, hence already including the water.


    ##Total acid waste flow
    hydrocylone_acid_waste = sim.STRM_Get_TotalFlow("HYDR1LIQ")
    filter1_acid_waste = sim.STRM_Get_TotalFlow("FILTLIQ")
    washer_acid_waste = sim.STRM_Get_TotalFlow("WASLIQOU")
    filter2_acid_waste = sim.STRM_Get_TotalFlow("FILT2LIQ")
    condensor_acid_waste = Massflow_acid_waste_condensor
    acid_waste_flow = (hydrocylone_acid_waste + filter1_acid_waste + washer_acid_waste + filter2_acid_waste + condensor_acid_waste) * plant_hours_per_year / 1000  # kg/hr to kg/year to tonne/year

    #CNF production numbers
    Carbon_out = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYEROUT", "CARBON")
    Ni_out = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYEROUT", "NI")
    Si_out = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYEROUT", "SI")
    Mg_out = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYEROUT", "MG")
    Al_out = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYEROUT", "AL")
    Fe_out = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYEROUT", "FE")
    Ca_out = sim.STRM_CISOLID_Get_MassFlowPerCompound("DRYEROUT", "CA")

    CNF_production = Carbon_out + Ni_out + Si_out + Mg_out + Al_out + Fe_out + Ca_out  # kg/hr
    CNF_purity = Carbon_out / CNF_production  # Wt%
    CNF_production_tonnes_per_year = CNF_production / 1000 * plant_hours_per_year  # tonnes/yr


    ################Techno-economic calculations
    solids_cooler_cost, solids_cooler_number = rotary_dryer_towler_2022(Area_cooler)
    CSTR_cost, CSTR_number = cstr_towler_2022(CSTR_volume)
    separator_cost, separator_number = preliminary_treatment_unit_woods(volumeflow_total_separator_in_Ls)
    hydrocyclone_cost, hydrocyclone_number = hydrocyclone_woods(volumeflow_total_hydrocyclone_in_Ls)
    filter1_cost, filter1_number = filter_towler_2022(filter1_area)
    washer_cost, washer_number = washer_mixersettler_woods(volumeflow_total_washer_in_Ls)
    filter2_cost, filter2_number = filter_towler_2022(filter2_area)
    slurry_heater_cost, slurry_heater_number = slurry_heater_floating_head_shell_and_tube_towler_2010(Area_slurry_heater)
    dryer_cost, dryer_number = rotary_dryer_towler_2022(Area_dryer)
    steam_boiler_cost, steam_boiler_number = steam_boiler_lbnl(steam_boiler_rating)
    water_pump_cost, water_pump_number = single_stage_centrifugal_pump_towler_2010(pump_massflow_Ls)
    air_filter_cost, air_filter_number = air_filter_woods(flow_dry_airNm3pers)
    air_heater_cost, air_heater_number = heater_plateframe_towler_2010(Area_air_heater)
    air_cooler_cost, air_cooler_number = air_cooler_floating_head_shell_and_tube_towler_2010(Area_air_cooler)


    equipment_cost_dict = {
        'Solids cooler cost': solids_cooler_cost,
        'CSTR cost': CSTR_cost,
        'Separator cost': separator_cost,
        'Hydrocyclone cost': hydrocyclone_cost,
        'Filter1 cost': filter1_cost,
        'Washer cost': washer_cost,
        'Filter2 cost': filter2_cost,
        'Slurry heater cost': slurry_heater_cost,
        'Dryer cost': dryer_cost,
        'Steam boiler cost': steam_boiler_cost,
        'Water pump cost': water_pump_cost,
        'Air filter cost': air_filter_cost,
        'Air heater cost': air_heater_cost,
        'Air cooler cost': air_cooler_cost
    }

    economic_results = {}

    isbl, fixed_capital_costs = capital_cost_calculation(
        solids_cooler_cost,
        CSTR_cost,
        separator_cost,
        hydrocyclone_cost,
        filter1_cost,
        washer_cost,
        filter2_cost,
        slurry_heater_cost,
        dryer_cost,
        air_filter_cost,
        air_heater_cost,
        steam_boiler_cost,
        water_pump_cost,
        air_cooler_cost
    )
    economic_results['isbl'] = isbl
    fixed_capital_costs_variance = fixed_capital_costs*fixed_capital_costs_variance_factor
    economic_results['fixed capital costs'] = fixed_capital_costs_variance


    vcop, raw_material_cost, electricity_cost, water_cost, acid_waste_cost, dust_waste_cost = opex_variable_production_cost(
        Acid_Feed_total_massflow,
        steam_boiler_electricity_consumption,
        water_consumption_yearly,
        massflow_dry_air_required,
        electricity_price,
        water_price,
        hcl_buying_price,
        plant_hours_per_year,
        acid_waste_flow,
        acid_waste_price,
        dust_waste_flow,
        dust_waste_price
    )
    economic_results['vcop'] = vcop
    economic_results['raw material cost'] = raw_material_cost
    economic_results['electricity cost'] = electricity_cost
    economic_results['water cost'] = water_cost
    economic_results['acid waste cost'] = acid_waste_cost
    economic_results['dust waste cost'] = dust_waste_cost



    fcop, cash_cost_of_production, fcop_dictionary = opex_fixed_production_costs(
        isbl,
        vcop,
        solids_cooler_number,
        CSTR_number,
        separator_number,
        hydrocyclone_number,
        filter1_number,
        washer_number,
        filter2_number,
        slurry_heater_number,
        dryer_number,
        operator_salary
    )
    fcop_variance = fcop*fcop_variance_factor
    economic_results['fcop with variance'] = fcop_variance
    economic_results['ccop'] = cash_cost_of_production
    working_capital_calculated = working_capital(isbl)
    economic_results['working capital'] = working_capital_calculated

    capital_cost_array, prod_array, cash_cost_array, revenue_array, gross_profit_array, depreciation_array, taxable_income_array, tax_paid_array, cash_flow = cash_flow_calculation(
        fixed_capital_costs_variance,
        working_capital_calculated,
        fcop_variance,
        vcop,
        CNF_production_tonnes_per_year,
        cnf_selling_price,
        project_lifetime
    )

    pv_array, npv_array = npv_calculation(cash_flow, interest_rate)
    pv = pv_array[-1]
    npv = npv_array[-1]
    economic_results['npv'] = npv

    # cash_flow_table = create_cash_flow_table(
    #     capital_cost_array,
    #     prod_array,
    #     cash_cost_array,
    #     revenue_array,
    #     gross_profit_array,
    #     depreciation_array,
    #     taxable_income_array,
    #     tax_paid_array,
    #     cash_flow,
    #     pv_array,
    #     npv_array
    # )

    levelized_cost_of_purification = calculate_levelized_cost(capital_cost_array, prod_array, cash_cost_array, interest_rate)
    economic_results['lcop'] = levelized_cost_of_purification

    payback_time = payback_time_calculation(revenue_array, cash_flow, fixed_capital_costs_variance, working_capital_calculated)
    economic_results['payback time'] = payback_time

    irr = calculate_irr(cash_flow)
    economic_results['irr'] = 100*irr #*100 for decimal to percentage

    roi = roi_calculation(fixed_capital_costs_variance, cash_flow, project_lifetime)
    economic_results['roi'] = roi

    return economic_results, equipment_cost_dict, CNF_production_tonnes_per_year

############################################################################################End of Running the simulation

def tornado_plot(parameters_data, calculation_function):
    plt.rcParams['axes.unicode_minus'] = False
    lcop_baseline = calculation_function(parameters_data)[0]['lcop']
    print(f"Baseline LCOP: {lcop_baseline:.4f}\n")

    results = {}
    for parameterr, baseline_value in parameters_data.items():
        parameters_minus_50 = parameters_data.copy()

        if parameterr == 'Project lifetime':
            parameters_minus_50[parameterr] = int(round(baseline_value * 0.5))
        else:
            parameters_minus_50[parameterr] = baseline_value * 0.5
        lcop_min_50percent = calculation_function(parameters_minus_50)[0]['lcop']

        parameters_plus_50percent = parameters_data.copy()

        if parameterr == 'Project lifetime':
            parameters_plus_50percent[parameterr] = int(round(baseline_value * 1.5))
        else:
            parameters_plus_50percent[parameterr] = baseline_value * 1.5
        lcop_plus_50percent = calculation_function(parameters_plus_50percent)[0]['lcop']

        results[parameterr] = {
            'minus_50': lcop_min_50percent,
            'plus_50': lcop_plus_50percent
        }

    sorted_results = sorted(results.items(), key=lambda item: max(abs(item[1]['plus_50'] - lcop_baseline), abs(item[1]['minus_50'] - lcop_baseline)), reverse=True)

    names_of_parameters = [item[0] for item in sorted_results]
    lcop_minimum = [item[1]['minus_50'] for item in sorted_results]
    lcop_maximum = [item[1]['plus_50'] for item in sorted_results]

    ############Plot Tornado plot
    plt.style.use('seaborn-v0_8-whitegrid')
    figure, axis = plt.subplots(figsize=(12, 6))
    y_position = np.arange(len(names_of_parameters))

    for k, (minimum, maximum) in enumerate(zip(lcop_minimum, lcop_maximum)):
        axis.barh(y_position[k], width=minimum - lcop_baseline, left=lcop_baseline, color='skyblue', label='-50%' if k == 0 else "")
        axis.barh(y_position[k], width=maximum - lcop_baseline, left=lcop_baseline, color='salmon', label='+50%' if k == 0 else "")

    axis.axvline(x=lcop_baseline, color='black', linestyle='--', linewidth=1)
    axis.set_yticks(y_position)
    axis.set_yticklabels(names_of_parameters, fontsize=12)
    axis.set_xlabel('Levelized cost of purification ($/kg)', fontsize=18)
    axis.invert_yaxis()

    minimum_value = min(min(lcop_minimum), min(lcop_maximum), lcop_baseline)
    maximum_value = max(max(lcop_minimum), max(lcop_maximum), lcop_baseline)
    axis.set_xlim(minimum_value - 0.05, maximum_value + 0.05)

    axis.legend(fontsize=16, loc='lower right')
    axis.tick_params(axis='x', labelsize=14)
    axis.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

tornado_plot(parameters_dict, full_calculation)


sim.CloseAspen()




















