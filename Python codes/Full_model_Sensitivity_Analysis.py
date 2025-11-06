import numpy as np
from Demos.BackupSeek_streamheaders import stream_name
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as mtick
from scipy.interpolate import splrep, splev
from scipy.optimize import brentq

from CodeLibrary import Simulation
from tabulate import tabulate


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
T_steam_CSTR_heating_in = 200
T_steam_CSTR_heating_out = 110

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


###Prices of waste streams
acid_waste_price = 316 #USD/tonne
dust_waste_price = 20 #USD/tonne


##########################################################################Running the simulation

#cnf_selling_price = 25000 ## Between 25.000 and 113.000 USD per tonne in 2017

#Dictionary for economic parameters
parameters_data = {
    'electricity_price': 0.10,
    'water_price': 0.10,
    'operator_salary': 3850,
    'interest_rate': 10,
    'hcl_buying_price': 112.03,
    'cnf_selling_price': 25000,
    'project_lifetime': 20
}


def full_calculation(parameters):

    electricity_price = parameters['electricity_price']
    water_price = parameters['water_price']
    operator_salary = parameters['operator_salary']
    interest_rate = parameters['interest_rate']/100 #Go from percentage (e.g. 10%) to 0.10. Is the number you need.
    hcl_buying_price = parameters['hcl_buying_price']
    cnf_selling_price = parameters['cnf_selling_price']
    project_lifetime = parameters['project_lifetime']

    sim.Run()
    sim.DialogSuppression(TrueOrFalse=True)


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
    massflow_steam_required_CSTR = CSTR_duty_kW * 1000 / (Cp_steam * (T_steam_CSTR_heating_in - T_steam_CSTR_heating_out)) * 3600  # kg/hr. *1000 to J. Cp is in J/kgK

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

    # CNF production numbers
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
    CSTR_parameters = {}
    CSTR_parameters['CSTR cost'] = CSTR_cost
    CSTR_parameters['Number of reactors'] = CSTR_number
    CSTR_parameters['CSTR total combined volume'] = CSTR_volume
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
        'Air cooler cost': air_cooler_cost,
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
    economic_results['fixed capital costs'] = fixed_capital_costs

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
    economic_results['fcop'] = fcop
    economic_results['ccop'] = cash_cost_of_production
    working_capital_calculated = working_capital(isbl)
    economic_results['working capital'] = working_capital_calculated


    capital_cost_array, prod_array, cash_cost_array, revenue_array, gross_profit_array, depreciation_array, taxable_income_array, tax_paid_array, cash_flow = cash_flow_calculation(
        fixed_capital_costs,
        working_capital_calculated,
        fcop,
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


    payback_time = payback_time_calculation(revenue_array, cash_flow, fixed_capital_costs, working_capital_calculated)
    economic_results['payback time'] = payback_time


    irr = calculate_irr(cash_flow)
    economic_results['irr'] = 100*irr #*100 for decimal to percentage


    roi = roi_calculation(fixed_capital_costs, cash_flow, project_lifetime)
    economic_results['roi'] = roi


    return economic_results, equipment_cost_dict, CNF_production_tonnes_per_year, CSTR_parameters

############################################################################################End of Running the simulation
lcop_results = []
tested_flow_rates = []
tested_flow_rates_tonnesperyear = []
production_tonnes_per_year_list = []
CSTR_volume = []
CSTR_number = []
CSTR_costs = []

###########################For loop for flow rate effect on the LCOP
# flow_rates_to_test = np.linspace(1500, 20000, 38)
#
#
# for CNF_Feed_TotalFlow2 in flow_rates_to_test:
#     print(f"Calculating for CNF_Feed_TotalFlow = {CNF_Feed_TotalFlow2} kg/hr...")
#     Acid_Feed_Lflow = CNF_Feed_TotalFlow2 * AcidSolution_Multiplier  # Assuming this calculation is correct
#     Washer_water_added = CNF_Feed_TotalFlow2
#
#     massflow_solid_slurry_dryer_in = 0.7255*CNF_Feed_TotalFlow2
#     massflow_water_slurry_dryer_in = 0.25*massflow_solid_slurry_dryer_in
#     massflow_dry_air_required = massflow_water_slurry_dryer_in / (Humidity_air_dryer_out - Humidity_air_dryer_in) * Dryer_safety_factor
#     sim.STRM_Set_ComponentFlowRate("DRYAIR", massflow_dry_air_required, "AIR")
#     sim.STRM_Set_TotalFlowRate("HCLFEED", Acid_Feed_Lflow)
#     sim.STRM_CISOLID_Set_TotalFlowRate("CNFFEED", CNF_Feed_TotalFlow2)
#     sim.STRM_Set_ComponentFlowRate("WASLIQIN", Washer_water_added, "H2O")
#
#     economic_results, equipment_cost_dict, production_tonnes_per_year, CSTR_parameters = full_calculation(parameters_data)
#     lcop_results.append(economic_results['lcop'])
#     tested_flow_rates.append(CNF_Feed_TotalFlow2)
#     CNF_Feed_TotalFlow2_tonnesperyear = CNF_Feed_TotalFlow2*plant_hours_per_year/1000 #Flow in tonnes per year
#     tested_flow_rates_tonnesperyear.append(CNF_Feed_TotalFlow2_tonnesperyear)
#     production_tonnes_per_year_list.append(production_tonnes_per_year)
#     CSTR_volume.append(CSTR_parameters['CSTR total combined volume'])
#     CSTR_number.append(CSTR_parameters['Number of reactors'])
#     CSTR_costs.append(CSTR_parameters['CSTR cost'])



#################################For loop for reactor T effect on LCOP
# CSTR_Temperatures_to_test = np.linspace(30, 100, 60)
# CSTR_Temperatures_Tested = []
#
#
# for CSTR_Temp in CSTR_Temperatures_to_test:
#     print(f"Calculating for CSTR Temperature = {CSTR_Temp} degrees Celsius...")
#     sim.BLK_CISTR_Set_Temperature("CSTR", CSTR_Temp)
#     sim.BLK_HEATER_Set_Temperature("CNFCOOLE", CSTR_Temp) #Equal to CSTR T
#     economic_results, equipment_cost_dict, production_tonnes_per_year, CSTR_parameters = full_calculation(parameters_data)
#     lcop_results.append(economic_results['lcop'])
#     print('lcop is', economic_results['lcop'])
#     CSTR_Temperatures_Tested.append(CSTR_Temp)
#     production_tonnes_per_year_list.append(production_tonnes_per_year)



###############################For loop for effect of Residence time on the LCOP
# CSTR_ResidenceTimes_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# CSTR_ResidenceTimes_Tested = []
#
# for CSTR_Residencetime in CSTR_ResidenceTimes_to_test:
#     print(f"Calculating for CSTR Residence time = {CSTR_Residencetime} hours...")
#     sim.BLK_CISTR_Set_ResidenceTime("CSTR", CSTR_Residencetime)
#     economic_results, equipment_cost_dict, production_tonnes_per_year, CSTR_parameters = full_calculation(parameters_data)
#     lcop_results.append(economic_results['lcop'])
#     CSTR_ResidenceTimes_Tested.append(CSTR_Residencetime)
#     production_tonnes_per_year_list.append(production_tonnes_per_year)
#     CSTR_volume.append(CSTR_parameters['CSTR total combined volume'])
#     CSTR_number.append(CSTR_parameters['Number of reactors'])
#     CSTR_costs.append(CSTR_parameters['CSTR cost'])



##############################For loop for effect of CNF selling price on the irr, roi, pbt and npv
# # Define the range of cnf_selling_price values to test
#
# price_range = np.linspace(10000, 113000, 104)
#
# irr_list = []
# roi_list = []
# payback_time_list = []
# npv_list = []
#
# for price in price_range:
#     print(f"Calculating for CNF selling price = {price} USD/tonne...")
#     parameters_data['cnf_selling_price'] = price
#     economic_results, equipment_cost, CNF_prod, CSTR_parameters = full_calculation(parameters_data)
#     irr_list.append(economic_results['irr'])
#     roi_list.append(economic_results['roi'])
#     payback_time_list.append(economic_results['payback time'])
#     npv_list.append(economic_results['npv'])


##########################################For loop and PLOT for effect of interest rate on the NPV
#
# interest_rate_list = np.linspace(10, 500, 50)
# lifetime_list = [10, 15, 20, 25, 30]
# plt.figure(figsize=(12, 8))
#
# for lifetime in lifetime_list:
#     npv_list = []
#     print(f"Calculating for lifetime = {lifetime} years...")
#
#     for interest_rate in interest_rate_list:
#         parameters_data['interest_rate'] = interest_rate
#         parameters_data['project_lifetime'] = lifetime
#
#         economic_results, equipment_cost, CNF_prod, CSTR_parameters = full_calculation(parameters_data)
#         npv_list.append(economic_results['npv'])
#
#     npv_list_np = np.array(npv_list)
#     npv_list_millions = npv_list_np / 1000000
#     plt.plot(interest_rate_list, npv_list_millions, label=f'Lifetime: {lifetime} years')
#
# axis = plt.gca()
# plt.xlabel('Interest rate (%)', fontsize = 18)
# plt.ylabel('Net Present Value ($ millions)', fontsize = 18)
# axis.tick_params(axis='y', labelsize=14)
# axis.tick_params(axis='x', labelsize=14)
# plt.grid(True)
# plt.legend(title='Project Lifetime', fontsize = 12, title_fontsize = 14)
# plt.show()



#################Determine the break-even point
#
# price_lower_bound = 100
# price_upper_bound = 50000
#
# try:
#     break_even_price = brentq(
#         lambda price: (parameters_data.update({'cnf_selling_price': price}) or full_calculation(parameters_data))[0]['npv'],
#         price_lower_bound,
#         price_upper_bound
#     )
#     print(f"The break-even selling price is: {break_even_price:.2f} per tonne.")
# except ValueError as e:
#     print(f"Error: {e}")
#     print("The NPV does not cross zero. Check price bounds.")



################### Plot of CNF production versus LCOP and SCALING EXPONENT
#
# results_dataframe = pd.DataFrame({
#     'CNF production (tonnes/year)': production_tonnes_per_year_list,
#     'LCOP': lcop_results
# })
#
# plt.figure(figsize=(10, 6))
# axis = plt.gca()
# plt.plot(results_dataframe['CNF production (tonnes/year)'], results_dataframe['LCOP'], marker='o')
# formatter = mtick.FuncFormatter(lambda x, pos: f'{x/1000:,.0f}')
# axis.xaxis.set_major_formatter(formatter)
# plt.xlabel('CNF production (kilotonnes / year)', fontsize=18)
# plt.ylabel('Levelized Cost of Purification ($/kg)', fontsize=18)
# axis.tick_params(axis='y', labelsize=14)
# axis.tick_params(axis='x', labelsize=14)
# plt.grid(True)
# plt.show()
#
# #Scaling exp
# results_dataframe2 = pd.DataFrame({
#     'CNF production (tonnes/year)': production_tonnes_per_year_list,
#     'LCOP': lcop_results
# })
#
# P1 = results_dataframe2['CNF production (tonnes/year)'].iloc[0]
# C1 = results_dataframe2['LCOP'].iloc[0]
# P2 = results_dataframe2['CNF production (tonnes/year)'].iloc[-1]
# C2 = results_dataframe2['LCOP'].iloc[-1]
#
# try:
#     scaling_exponent = np.log(C2 / C1) / np.log(P2 / P1)
#
#     print("\n--- Scaling Exponent Calculation ---")
#     print(f"P1 (Initial Production): {P1:.2f} tonnes/year")
#     print(f"C1 (Initial LCOP): {C1:.2f} USD/kg")
#     print(f"P2 (Final Production): {P2:.2f} tonnes/year")
#     print(f"C2 (Final LCOP): {C2:.2f} USD/kg")
#     print(f"\nCalculated Scaling Exponent (E): {scaling_exponent:.4f}")
#
# except (ZeroDivisionError, ValueError) as eee:
#     print(f"Error in calculation: {eee}.")




######################Plot of CSTR temperature versus LCOP
#
# results_dataframe = pd.DataFrame({
#     'CSTR Temperature': CSTR_Temperatures_Tested,
#     'LCOP': lcop_results
# })
#
# plt.figure(figsize=(10, 6))
# plt.plot(results_dataframe['CSTR Temperature'], results_dataframe['LCOP'], marker='o', linestyle='-')
# axis = plt.gca()
# plt.xlabel(r'CSTR Temperature ($\circ$C)', fontsize=18)
# plt.ylabel('Levelized Cost of Purification ($/kg)', fontsize=18)
# axis.tick_params(axis='y', labelsize=14)
# axis.tick_params(axis='x', labelsize=14)
# plt.grid(True)
# plt.show()



######################Plot of CSTR Residence time versus LCOP
#
# results_dataframe = pd.DataFrame({
#     'CSTR Residence time': CSTR_ResidenceTimes_Tested,
#     'LCOP': lcop_results
# })
#
# plt.figure(figsize=(10, 6))
# plt.plot(results_dataframe['CSTR Residence time'], results_dataframe['LCOP'], marker='o', linestyle='-')
# axis = plt.gca()
# plt.xlabel('CSTR residence time (hours)', fontsize=18)
# plt.ylabel('Levelized Cost of Purification ($/kg)', fontsize=18)
# axis.tick_params(axis='y', labelsize=14)
# axis.tick_params(axis='x', labelsize=14)
# plt.grid(True)
# plt.show()



#############################Plots of cnf selling price versus irr, roi, pbt and npv
# payback_time_list = np.nan_to_num(payback_time_list, nan=100)
#
# price_range_smooth = np.linspace(min(price_range), max(price_range), 500)
# figure, axes = plt.subplots(2, 2, figsize=(12, 10))
# plt.subplots_adjust(hspace=0.65, wspace = 0.4) #Vertical space between plots
# x_ticks = np.arange(10000, 130000, 10000)
# x_labels = (x_ticks / 1000).astype(int)
#
# #Plot of IRR
# spline_irr = splrep(price_range, irr_list)
# irr_smooth = splev(price_range_smooth, spline_irr)
# axes[0, 0].plot(price_range_smooth, irr_smooth, '-')
# axes[0, 0].set_title('Effect of the CNF selling price on the IRR', fontsize = 20)
# axes[0, 0].set_xlabel('CNF Selling Price ($ thousands / tonne)', fontsize = 18)
# axes[0, 0].set_ylabel('Internal Rate of Return (%)', fontsize = 18)
# axes[0, 0].tick_params(axis='y', labelsize=14)
# axes[0, 0].tick_params(axis='x', labelsize=14)
# axes[0, 0].set_xlim(10000, 120000)
# axes[0, 0].grid(True)
# axes[0, 0].set_xticks(x_ticks, labels=x_labels)
#
# #Plot of ROI
# spline_roi = splrep(price_range, roi_list)
# roi_smooth = splev(price_range_smooth, spline_roi)
# axes[0, 1].plot(price_range_smooth, roi_smooth, '-', color='orange')
# axes[0, 1].set_title('Effect of the CNF selling price on the ROI', fontsize = 20)
# axes[0, 1].set_xlabel('CNF Selling Price ($ thousands / tonne)', fontsize = 18)
# axes[0, 1].set_ylabel('Return on Investment (%)', fontsize = 18)
# axes[0, 1].tick_params(axis='y', labelsize=14)
# axes[0, 1].tick_params(axis='x', labelsize=14)
# axes[0, 1].set_xlim(10000, 120000)
# axes[0, 1].grid(True)
# axes[0, 1].set_xticks(x_ticks, labels=x_labels)
#
# #Plot of PBT
# spline_ofpayb = splrep(price_range, payback_time_list)
# payback_smooth = splev(price_range_smooth, spline_ofpayb)
# axes[1, 0].plot(price_range_smooth, payback_smooth, '-', color='green')
# axes[1, 0].set_title('Effect of the CNF selling price on the PBT', fontsize = 20)
# axes[1, 0].set_xlabel('CNF Selling Price ($ thousands / tonne)', fontsize = 18)
# axes[1, 0].set_ylabel('Payback Time (years)', fontsize = 18)
# axes[1, 0].tick_params(axis='y', labelsize=14)
# axes[1, 0].tick_params(axis='x', labelsize=14)
# axes[1, 0].set_xlim(10000, 120000)
# axes[1, 0].grid(True)
# axes[1, 0].set_xticks(x_ticks, labels=x_labels)
#
# #Plot of NPV
# spline_npv = splrep(price_range, npv_list)
# npv_smooth = splev(price_range_smooth, spline_npv)
# npv_smooth_millions = npv_smooth / 1_000_000
# axes[1, 1].plot(price_range_smooth, npv_smooth_millions, '-', color='red')
# axes[1, 1].set_title('Effect of CNF selling price on the NPV', fontsize = 20)
# axes[1, 1].set_xlabel('CNF Selling Price ($ thousands / tonne)', fontsize = 18)
# axes[1, 1].set_ylabel('Net Present Value ($ millions)', fontsize = 18)
# axes[1, 1].tick_params(axis='y', labelsize=14)
# axes[1, 1].tick_params(axis='x', labelsize=14)
# axes[1, 1].set_xlim(10000, 120000)
# axes[1, 1].grid(True)
# axes[1, 1].set_xticks(x_ticks, labels=x_labels)
#
# plt.show()



######################Plots of CSTR parameters
# #Run CNF production loop or CSTR residence time loop for this
#
# #CSTR cost versus CSTR volume
# plt.figure(figsize=(10, 6))
# plt.plot(CSTR_volume, CSTR_costs, marker='o', linestyle='-')
# axis1 = plt.gca()
# plt.xlabel('Total CSTR volume ($m^3$)', fontsize = 18)
# plt.ylabel('CSTR cost ($ millions)', fontsize = 18)
# sizing_y_axis = mtick.FuncFormatter(lambda y, p: f'{y/1e6:,.0f}')
# axis1.yaxis.set_major_formatter(sizing_y_axis)
# axis1.tick_params(axis='y', labelsize=14)
# axis1.tick_params(axis='x', labelsize=14)
# plt.grid(True)
#
# #LCOP versus CSTR volume
# plt.figure(figsize=(10, 6))
# plt.plot(CSTR_volume, lcop_results, marker='o', linestyle='-')
# axis2 = plt.gca()
# plt.xlabel('Total CSTR volume ($m^3$)', fontsize = 18)
# plt.ylabel('Levelized Cost of Purification($/kg)', fontsize = 18)
# axis2.tick_params(axis='y', labelsize=14)
# axis2.tick_params(axis='x', labelsize=14)
# plt.grid(True)
#
# #CSTR number versus CSTR volume
# plt.figure(figsize=(10, 6))
# plt.plot(CSTR_volume, CSTR_number, marker='o', linestyle='-')
# axis3 = plt.gca()
# plt.xlabel('Total CSTR volume ($m^3$)', fontsize = 18)
# plt.ylabel('Number of Reactors', fontsize = 18)
# axis3.tick_params(axis='y', labelsize=14)
# axis3.tick_params(axis='x', labelsize=14)
# plt.grid(True)
# plt.show()



sim.CloseAspen()















