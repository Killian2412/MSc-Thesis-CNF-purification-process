import numpy as np
import pandas as pd
from scipy.optimize import root_scalar


#############################################Input and important comments


Cepci_Jan2010 = 532.90 #Towler
Cepci_2023 = 800.8 #https://www.training.itservices.manchester.ac.uk/public/gced/CEPCI.html?reactors/CEPCI/index.html
Cepci_Woods = 1000
Cepci_2018 = 603.10 #https://www.training.itservices.manchester.ac.uk/public/gced/CEPCI.html?reactors/CEPCI/index.html
#Cepci_Jan2004 = 400 #Ulrich figure 5.57B, page 410

#Factorial method
#installation factors
f_er = 0.5 #equipment erection
f_p = 0.6 #piping
f_m = 1 #materials, we assume carbon steel
f_i = 0.3 #instrumentation and control
f_el = 0.2 #electrical
f_c = 0.3 #civil
f_s = 0.2 #structures and buildings
f_l = 0.1 #lagging, insulation, or paint

OS = 0.4 #offsites, 40% of ISBL cost is used if no details of the site are known
Design_and_Engineering = 0.25 #design and engineering (Is 30% of ISBL + OSBL for smaller projects and 10% for larger projects)
Contingency = 0.1 #minimum of 10% of ISBL + OSBL
Location_factor = 1.19 #location factor for the Netherlands #--> Assume the Woods and Lbnl correlations are also on US Gulf Coast basis.

#hcl_price_input = 112.03 # USD per metric tonne, circa mei 2025 https://businessanalytiq.com/procurementanalytics/index/hydrochloric-acid-price-index/
#electricity_price_input = 0.10 #USD/kWh, from Panji. Electricity prices in Netherlands on 28/08/2025: URL https://euenergy.live/country.php?a2=NL
#water_price_input = 0.3 #USD/m^3, from Panji. But this depends on the price of electricity that you use!
#water_price_input = 2.37 #USD/m^3, with conversion rate https://www.rabobank.com/knowledge/d011409185-the-price-of-water
#water_price_input = 0.067 #USD/1000kg, is USD per 1000L, is 0.067 USD per m^3 https://www.pearson.de/analysis-synthesis-and-design-of-chemical-processes-9780134177489. Table 8.3
#water_price_input = 0.067 * 136.91/90.55 #USD per m^3. Based on two sources: https://www.pearson.de/analysis-synthesis-and-design-of-chemical-processes-9780134177489
# and https://www.theglobaleconomy.com/Netherlands/cpi/ for the CPI indexes in April 2008 and April 2025
#This gives a value of around 0.10 USD per m^3.
dry_air_price_input = 0
## operator_salary = 3270  # euros per month on average. Calculated using multiple sources, which are described in the techno eco and eq's document.


tax_rate_input = 0.21 #Dutch tax rate
#project_lifetime_input = 20 #is usual
#interest_rate_input = 1.10 #Interest_rate_input #Common in the industrial sector is an interest/discount rate of 8-12%. For lower risk projects, it can be 3-7% and for higher risk, it can be higher, around 15%.
    #https://ieomsociety.org/proceedings/singapore2025/362.pdf
    #https://remadeinstitute.org/wp-content/uploads/2025/03/75_Agile-techno-economic_Shi.pdf
    #https://www.osti.gov/servlets/purl/2476013
    #If IRR surpasses the interest/discount rate, the project is viable. See https://www.sciencedirect.com/science/article/pii/S2666789424000734

########################################### Fixed capital investment

#Calculate parallel units and calculate if lower_bound is satisfied
def calculate_parallel_units(s, upper_bound):
    num_units = np.ones_like(s, dtype=int) #Calculates the number of units / size of the array (which is a scalar in this case)

    if np.any(s > upper_bound): #Checks if s exceeds the upper bound
        num_units = np.ceil(s / upper_bound).astype(int) #Calculates how many units are needed (s / upper bound). Rounds up to whole number.
        adjusted_s = s / num_units #Adjust original s value to s value of each parallel piece of equipment
    else:
        adjusted_s = s #if upper bound is not exceeded, s will stay the same

    return num_units, adjusted_s

def validate_input_range(s, lower_bound, upper_bound):
    if s < lower_bound:
        raise ValueError(
            f'One or more parameter values fall below the minimum limit of the cost correlation range:'
            f' [{lower_bound}, {upper_bound}]'
        )
    return s

#Cost calculation equations
def equipment_cost(a, b, s, n, cepci1, cepci2, num_units):
    cost = a + b*s**n
    cost_cepci_correction = cost*(cepci2/cepci1)
    total_cost_equipment = cost_cepci_correction * num_units
    return total_cost_equipment

def scaling_law_woods(cost_ref, size_ref, size_equipment, n, cepci1, cepci2, num_units):
    cost_equipment = cost_ref*(size_equipment/size_ref)**n
    cost_equipment_cepci_correction = cost_equipment*(cepci2/cepci1)
    total_cost_equipment = cost_equipment_cepci_correction * num_units
    return total_cost_equipment

#Also use this correlation for the solids cooler
def rotary_dryer_woods(s):
    s_rotary_dryer = validate_input_range(s, 40, 700) #m^2, range is 40-700
    s_reference_rotary_dryer = 100 #m^2
    cost_reference_rotary_dryer = 155000 #USD
    n_rotary_dryer = 0.75
    num_units_rotary_dryer, s_rotary_dryers = calculate_parallel_units(s_rotary_dryer, 700)
    cooler_cost = scaling_law_woods(cost_reference_rotary_dryer, s_reference_rotary_dryer, s_rotary_dryers, n_rotary_dryer, Cepci_Woods, Cepci_2023, num_units_rotary_dryer)
    return cooler_cost, num_units_rotary_dryer

def rotary_dryer_towler_2022(s):
    a_rotary_dryer = 15000
    b_rotary_dryer = 10500
    s_rotary_dryer = validate_input_range(s, 7, 180) #m^2 Range is 11-180, but had to lower a bit
    n_rotary_dryer = 0.9
    num_units_rotary_dryer, s_rotary_dryers = calculate_parallel_units(s_rotary_dryer, 180)
    rotary_dryer_cost = equipment_cost(a_rotary_dryer, b_rotary_dryer, s_rotary_dryers, n_rotary_dryer, Cepci_Jan2010, Cepci_2023, num_units_rotary_dryer)
    return rotary_dryer_cost, num_units_rotary_dryer

#This is for an agitated reactor without glass lining
def cstr_towler_2022(s):
    a_cstr = 61500
    b_cstr = 32500
    s_cstr = validate_input_range(s, 0.5, 100) #volume of the reactor in m^3, 0.5-100 m^3
    n_cstr = 0.65
    num_units_cstr, s_cstrs = calculate_parallel_units(s_cstr, 100)
    cstr_cost = equipment_cost(a_cstr, b_cstr, s_cstrs, n_cstr, Cepci_Jan2010, Cepci_2023, num_units_cstr)
    return cstr_cost, num_units_cstr

def preliminary_treatment_unit_woods(s):
    s_preliminary_treatment_unit = validate_input_range(s, 1, 10000) #L/s, range of 1-10000
    s_reference_preliminary_treatment_unit = 58 #L/s
    cost_reference_preliminary_treatment_unit = 270000 #USD
    n_preliminary_treatment_unit = 0.64
    num_units_preliminary_treatment_unit, s_preliminary_treatment_units = calculate_parallel_units(s_preliminary_treatment_unit, 10000)
    preliminary_treatment_unit_cost = scaling_law_woods(cost_reference_preliminary_treatment_unit, s_reference_preliminary_treatment_unit, s_preliminary_treatment_units, n_preliminary_treatment_unit, Cepci_Woods, Cepci_2023, num_units_preliminary_treatment_unit)
    return preliminary_treatment_unit_cost, num_units_preliminary_treatment_unit

def hydrocyclone_woods(s):
    if s < 9:
        #print('s is outside range for the hydrocyclone, setting s equal to 9 L/s, original value:', s)
        s = 9
    s_hydrocyclone = validate_input_range(s, 9, 1300)
    s_reference_hydrocyclone = 50 #L/s
    cost_reference_hydrocyclone = 38000 #USD
    n_hydrocylone = 0.35
    num_units_hydrocyclone, s_hydrocyclones = calculate_parallel_units(s_hydrocyclone, 1300)
    hydrocyclone_cost = scaling_law_woods(cost_reference_hydrocyclone, s_reference_hydrocyclone, s_hydrocyclones, n_hydrocylone, Cepci_Woods, Cepci_2023, num_units_hydrocyclone)
    return hydrocyclone_cost, num_units_hydrocyclone

def filter_towler_2022(s): #Vacuum drum filter
    s_filter = validate_input_range(s, 1, 180) #original range is 10-180 m^2. Lowered range here, as using the minimum would not suffice due to larger price.
    a_filter = -73000
    b_filter = 93000
    n_filter = 0.3
    num_units_filter, s_filters = calculate_parallel_units(s_filter, 180)
    filter_cost = equipment_cost(a_filter, b_filter, s_filters, n_filter, Cepci_Jan2010, Cepci_2023, num_units_filter)
    return filter_cost, num_units_filter

def washer_mixersettler_woods(s):
    if s < 1:
        #print('s is outside range for the washer, setting s equal to 1 L/s, original value:', s)
        s = 1

    s_washer = validate_input_range(s, 1, 100) #originally 1 to 100
    s_reference_washer = 10 #For volumetric flow rate of aqueous phase to be treated = 10 L/s
    cost_reference_washer = 30000 #USD

    if s_washer < 10:
        n_washer = 0.22
    elif s_washer >= 10: #Up to 100
        n_washer = 0.60
    else:
        raise ValueError("The s for the calculation of the washer is not in range")

    num_units_washer, s_washers = calculate_parallel_units(s_washer, 100)
    washer_cost = scaling_law_woods(cost_reference_washer, s_reference_washer, s_washers, n_washer, Cepci_Woods, Cepci_2023, num_units_washer)
    return washer_cost, num_units_washer

def single_stage_centrifugal_pump_towler_2010(s):  # doi: https://doi.org/10.1016/B978-0-12-821179-3.00007-8
    a_pump = 8000
    b_pump = 240
    s_pump = validate_input_range(s, 0.2, 126) #L/s, range of 0.2 to 126 L/s
    n_pump = 0.9
    num_units_pump, s_pumps = calculate_parallel_units(s_pump, 126)
    pump_cost = equipment_cost(a_pump, b_pump, s_pumps, n_pump, Cepci_Jan2010, Cepci_2023, num_units_pump)
    return pump_cost, num_units_pump

def steam_boiler_lbnl(s): #https://industrialapplications.lbl.gov/sites/default/files/pdf-embeds/IAC%20Decarb%20Tipsheet%203.pdf
    a_boiler = 0
    b_boiler = 110280
    s_boiler = validate_input_range(s, 10, 250) #Boiler rating in MMBtu/hr, range of 10 to 250 MMBtu/hr
    n_boiler = 0.627
    num_units_boiler, s_boilers = calculate_parallel_units(s_boiler, 250)
    boiler_cost = equipment_cost(a_boiler, b_boiler, s_boilers, n_boiler, Cepci_2018, Cepci_2023, num_units_boiler)
    return boiler_cost, num_units_boiler

def air_filter_woods(s): #This is a bag filter
    if s < 2.5:
        #print('s is outside range for the air filter, setting s equal to 2.5 Nm^3/s, original value:', s)
        s = 2.5
    s_airfilter = validate_input_range(s, 2.5, 30)
    s_reference_airfilter = 10 #for inlet gas feed rate of 10 Nm^3/s
    cost_reference_airfilter = 82000 #USD
    n_airfilter = 0.71 #for range 2.5-30 Nm^3
    num_units_airfilter, s_airfilters = calculate_parallel_units(s_airfilter, 30)
    airfilter_cost = scaling_law_woods(cost_reference_airfilter, s_reference_airfilter, s_airfilters, n_airfilter, Cepci_Woods, Cepci_2023, num_units_airfilter)
    return airfilter_cost, num_units_airfilter

#
def heater_plateframe_towler_2010(s):
    a_heater = 1600
    b_heater = 210
    s_heater = validate_input_range(s, 1, 500) #s in m2 of heat exchange area (1-500)
    n_heater = 0.95
    num_units_heater, s_heaters = calculate_parallel_units(s_heater, 500)
    heater_cost = equipment_cost(a_heater, b_heater, s_heaters, n_heater, Cepci_Jan2010, Cepci_2023, num_units_heater)
    return heater_cost, num_units_heater

def slurry_heater_floating_head_shell_and_tube_towler_2010(s):
    a_slurry_heater = 32000
    b_slurry_heater = 70
    s_slurry_heater = validate_input_range(s, 10, 1000) #m^2, range of 10-1000 m^2
    n_slurry_heater = 1.2
    num_units_slurry_heater, s_slurry_heaters = calculate_parallel_units(s_slurry_heater, 1000)
    slurry_heater_cost = equipment_cost(a_slurry_heater, b_slurry_heater, s_slurry_heaters, n_slurry_heater, Cepci_Jan2010, Cepci_2023, num_units_slurry_heater)
    return slurry_heater_cost, num_units_slurry_heater

def air_cooler_floating_head_shell_and_tube_towler_2010(s):
    a_air_cooler = 32000
    b_air_cooler = 70
    s_air_cooler = validate_input_range(s, 10, 1000) #m^2, range of 10-1000 m^2
    n_air_cooler = 1.2
    num_units_air_cooler, s_air_coolers = calculate_parallel_units(s_air_cooler, 1000)
    air_cooler_cost = equipment_cost(a_air_cooler, b_air_cooler, s_air_coolers, n_air_cooler, Cepci_Jan2010, Cepci_2023, num_units_air_cooler)
    return air_cooler_cost, num_units_air_cooler

def capital_cost_calculation(cooler, cstr, separator, hydrocyclone, filter1, washer, filter2, solid_heater, dryer, air_filter, air_heater, steam_boiler, water_pump, air_cooler):
    c_total = cooler + cstr + separator + hydrocyclone + filter1 + washer + filter2 + solid_heater + dryer + air_filter + air_heater + steam_boiler + water_pump + air_cooler
    isbl = c_total*((1+f_p)*f_m + f_er + f_el + f_i + f_c +f_s + f_l)
    fixed_capital_costs = isbl*(1 + OS)*(1 + Design_and_Engineering + Contingency)*Location_factor
    return isbl, fixed_capital_costs

#############################################OPEX
#########################Variable costs of production

def opex_variable_production_cost(hcl_consumption, electricity_consumption, water_consumption, air_consumption, electricity_price, water_price, hcl_price, plant_hours_per_year, acid_waste_flow, acid_waste_price, dust_waste_flow, dust_waste_price):
    # Raw material costs
    hcl_cost = hcl_price  # USD per metric tonne
    hcl_annual_consumption = hcl_consumption*plant_hours_per_year/1000 # input of this in kg/hr, then multiply it by yearly operating hours, divide it by 1000 to get consumption in tonnes
    hcl_total_cost = hcl_annual_consumption * hcl_cost
    raw_material_cost = hcl_total_cost
    # No CNF annual cost included. Want to calculate the LCO of purification.

    # Utilities costs

    electricity_cost = electricity_price*electricity_consumption #USD, electricity consumption is in kWh/year

    water_cost = water_price*water_consumption #USD. water consumption is in m^3/year

    dry_air_price = dry_air_price_input #There is actually no price for the atmospheric air itself
    dry_air_consumption = air_consumption*24*plant_hours_per_year #kg/year
    dry_air_cost = dry_air_price*dry_air_consumption

    utilities_cost = electricity_cost + water_cost + dry_air_cost #+ natural_gas_cost + chemicals_cost

    #Consumables costs
    #HCl is included in raw material costs

    #Waste disposal costs
    acid_waste_cost = acid_waste_flow*acid_waste_price #In USD. Price is USD/tonne, flow is in tonne/year
    dust_waste_cost = dust_waste_flow*dust_waste_price #In USD. Price is USD/tonne, flow is in tonne/year

    # Packing and shipping costs
    # /

    vcop = raw_material_cost + utilities_cost + acid_waste_cost + dust_waste_cost # Total variable cost of production

    return vcop, raw_material_cost, electricity_cost, water_cost, acid_waste_cost, dust_waste_cost


##########################Fixed costs of production

def opex_fixed_production_costs(isbl, vcop, solids_cooler_number, cstr_number, separator_number, hydrocyclone_number, filter1_number, washer_number, filter2_number, slurry_heater_number, dryer_number, operator_salary):

    # Labor costs
    operator_number = 5*(3 + solids_cooler_number + cstr_number + separator_number + hydrocyclone_number + filter1_number + washer_number + filter2_number + slurry_heater_number + dryer_number) #Towler2022: for fluids+solids continuous process there are 3 shift positions + 1 for every solids handling section. Then multiply by 5 for teams.
    operator_cost_yearly = operator_number * operator_salary * 12
    supervision_and_management_cost_yearly = 0.25 * operator_cost_yearly  # supervision and management costs
    direct_salary_overhead_cost_yearly = 0.6 * (operator_cost_yearly + supervision_and_management_cost_yearly)  # direct salary overhead costs

    # Maintenance costs
    maintenance_cost_yearly = 0.05 * isbl  # 5% of ISBL per year

    # Land, rent and local property taxes
    landrent_yearly = 0.01 * isbl * (1 + OS)  # 1% of ISBL+OSBL per year
    property_taxes = 0.01 * isbl * (1 + OS)  # 1% of ISBL+OSBL per year

    # Insurance
    insurance_cost_yearly = 0.01 * isbl * (1 + OS)  # 1% of ISBL+OSBL per year

    # Allocated environmental charges to cover superfund payments
    environmental_charges_cost_yearly = 0.01 * isbl * (1 + OS)  # 1% of ISBL+OSBL per year

    # Interest payments
    # /

    # Corporate overhead charges
    general_and_administrative_cost_yearly = 0.65 * (operator_cost_yearly + supervision_and_management_cost_yearly + direct_salary_overhead_cost_yearly)  # 65% of labor cost plus supervision and overhead

    fixed_production_costs_prelim = operator_cost_yearly + supervision_and_management_cost_yearly + direct_salary_overhead_cost_yearly + maintenance_cost_yearly + landrent_yearly + property_taxes + insurance_cost_yearly + environmental_charges_cost_yearly + general_and_administrative_cost_yearly
    cash_cost_of_production = (vcop + fixed_production_costs_prelim) / (1 - 0.07)  #Division by 1 - 0.07 to account for r&d, s&m and patents&royalties

    r_and_d_cost_yearly = 0.03 * cash_cost_of_production  # 1-15% of revenues.
    selling_and_marketing_cost_yearly = 0.02 * cash_cost_of_production  # 0-5% of total cost of production.

    # License fees and royalties
    patents_royalties = 0.02 * cash_cost_of_production

    # Total fixed cost of production
    fcop = operator_cost_yearly + supervision_and_management_cost_yearly + direct_salary_overhead_cost_yearly + maintenance_cost_yearly + landrent_yearly + property_taxes + insurance_cost_yearly + r_and_d_cost_yearly + selling_and_marketing_cost_yearly + general_and_administrative_cost_yearly + environmental_charges_cost_yearly + patents_royalties

    fcop_dictionary = {}
    fcop_dictionary['Operator cost'] = operator_cost_yearly
    fcop_dictionary['Supervision and management cost'] = supervision_and_management_cost_yearly
    fcop_dictionary['Direct salary overhead cost'] = direct_salary_overhead_cost_yearly
    fcop_dictionary['Maintenance cost'] = maintenance_cost_yearly
    fcop_dictionary['Land rent'] = landrent_yearly
    fcop_dictionary['Property taxes'] = property_taxes
    fcop_dictionary['Insurance cost'] = insurance_cost_yearly
    fcop_dictionary['Environmental charges'] = environmental_charges_cost_yearly
    fcop_dictionary['General and administrative cost'] = general_and_administrative_cost_yearly
    fcop_dictionary['R&D cost'] = r_and_d_cost_yearly
    fcop_dictionary['Patents and royalties cost'] = patents_royalties
    fcop_dictionary['Selling and marketing cost'] = selling_and_marketing_cost_yearly

    return fcop, cash_cost_of_production, fcop_dictionary


############################################# Working capital (PART OF CAPEX)

def working_capital(isbl):
    fixed_capital_cost = isbl*(1 + OS)
    working_capital_calc = 0.10*fixed_capital_cost #Sinnot says % of fixed capital (ISBL plus OSBL cost). But is it then only ISBL + OSBL or also contingency, LF and D&E?
    #5% is low and is used for a simple single product process with little or no finished product storage. Typical figure for petrochemical plants is 15%. Is have now estimated 10%, to be inbetween the numbers of one product and that of typical petrochemical plant.

    return working_capital_calc


##############################################################Revenues and profits second part

def cash_flow_calculation(fixed_capital_cost, working_capital_cost, fixed_opex, variable_opex, cnf_production, cnf_selling_price, project_lifetime):
    tax_rate = tax_rate_input #Dutch tax rate.

    # Initialize arrays
    capital_cost_array = np.zeros(project_lifetime)
    prod_array = np.zeros(project_lifetime)
    revenue_array = np.zeros(project_lifetime)
    cash_cost_array = np.zeros(project_lifetime)
    gross_profit_array = np.zeros(project_lifetime)
    depreciation_array = np.zeros(project_lifetime)
    taxable_income_array = np.zeros(project_lifetime)
    tax_paid_array = np.zeros(project_lifetime)

    previous_taxable_income = 0
    depreciation_counter = 0
    depreciation_amount = fixed_capital_cost / (project_lifetime / 2) #Is just an assumption. Copied the assumption made by Panji.

    cash_flow = np.zeros(project_lifetime)

    for year in range(project_lifetime):
        if year == 0:
            prod = 0
            cash_cost = 0
            capital_cost = fixed_capital_cost * 0.3
            revenue = 0
        elif year == 1:
            prod = 0
            cash_cost = 0
            capital_cost = fixed_capital_cost * 0.6
            revenue = 0
        elif year == 2:
            prod = 0.3 * cnf_production
            cash_cost = fixed_opex + 0.3 * variable_opex
            capital_cost = fixed_capital_cost * 0.1 + working_capital_cost
            revenue = cnf_selling_price * prod
        elif year == 3:
            prod = 0.7 * cnf_production
            cash_cost = fixed_opex + 0.7 * variable_opex
            capital_cost = 0
            revenue = cnf_selling_price * prod
        else:
            prod = cnf_production
            cash_cost = fixed_opex + variable_opex
            capital_cost = 0
            revenue = cnf_selling_price * prod

        gross_profit = revenue - cash_cost  # is per year


        if gross_profit > 0 and depreciation_counter < (project_lifetime / 2):
            depreciation = depreciation_amount
            depreciation_counter += 1
        else:
            depreciation = 0

        #Tax is paid over the income of the previous year
        taxable_income = gross_profit - depreciation_amount #Taxable income = gross profit - tax allowances. Most common one is depreciation.
        tax_paid = tax_rate * previous_taxable_income if previous_taxable_income >0 else 0

        capital_cost_array[year] = capital_cost
        prod_array[year] = prod
        cash_cost_array[year] = cash_cost
        revenue_array[year] = revenue
        gross_profit_array[year] = gross_profit
        depreciation_array[year] = depreciation
        taxable_income_array[year] = taxable_income
        tax_paid_array[year] = tax_paid
        cash_flow[year] = gross_profit - tax_paid - capital_cost #If you do not subtract capital cost, will pay taxes in year 1 already

        previous_taxable_income = taxable_income

    #Below is because you get the working capital back at the end of the lifetime
    capital_cost_array[-1] -= working_capital_cost
    cash_flow[-1] += working_capital_cost

    return (capital_cost_array, prod_array, cash_cost_array, revenue_array, gross_profit_array,
            depreciation_array, taxable_income_array, tax_paid_array, cash_flow)

##############################################################Economic evaluation of the project

def npv_calculation(cash_flow, interest_rate):
    years = np.arange(1, len(cash_flow) + 1) #Why the +1? This is just to get from 1 to 20 instead of 0 to 19
    pv_array = cash_flow / ((1 + interest_rate) ** years)
    npv_array = np.cumsum(pv_array)

    return pv_array, npv_array

def create_cash_flow_table(capital_cost_array, prod_array, revenue_array, cash_cost_array, gross_profit_array, depreciation_array, taxable_income_array, tax_paid_array, cash_flow, pv_array, npv_array):

    data = {
        "Year": np.arange(len(cash_flow)) + 1,
        "Capital cost": capital_cost_array,
        "Production [unit]": prod_array,
        "Revenue": revenue_array,
        "Cash cost of prod.": cash_cost_array,
        "Gross profit": gross_profit_array,
        "Depreciation": depreciation_array,
        "Taxable income": taxable_income_array,
        "Tax paid": tax_paid_array,
        "Cash flow": cash_flow,
        "PV of cash flow": pv_array,
        "NPV": npv_array,
    }

    df = pd.DataFrame(data)

    formatted_df = df.style.format(
        {col: "${:,.2f}" for col in df.columns if col != "Year"}
    )

    output_path_html = "cash_flow_table.html"
    with open(output_path_html, "w") as f:
        f.write(formatted_df.to_html())

    print(f"Formatted table saved to {output_path_html}")

    return formatted_df


def calculate_levelized_cost(capital_cost, prod, cash_cost, interest_rate):
    disc_capex = np.zeros(len(cash_cost))
    disc_opex = np.zeros(len(cash_cost))
    disc_prod = np.zeros(len(cash_cost))

    for year in range(len(cash_cost)):
        disc_capex[year] = (capital_cost[year]) / ((1 + interest_rate) ** (year+1))
        disc_opex[year] = (cash_cost[year]) / ((1 + interest_rate) ** (year+1))
        disc_prod[year] = prod[year] / ((1 + interest_rate) ** (year+1))

    lcop_tonne = max(np.sum(disc_capex + disc_opex) / np.sum(disc_prod), 0)
    lcop_kg = lcop_tonne/1000
    return lcop_kg


def payback_time_calculation(revenue, cash_flow, fixed_capital_cost, working_capital_input):
    revenue_generating_years = cash_flow[revenue > 0]

    #Should represent the total investment cost, working capital is also included in that. See page 316 in S&T.
    total_investment = fixed_capital_cost + working_capital_input

    if len(revenue_generating_years) == 0:
        payback_time = float('nan')
    else:
        average_annual_cash_flow = np.mean(revenue_generating_years)
        payback_time = total_investment / average_annual_cash_flow if average_annual_cash_flow > 0 else float(
            'nan')

    return payback_time

# IRR is also referred to as DCFROR: discounted cash flow rate of return
def calculate_irr(cash_flow):
    """
    Calculates the Internal Rate of Return (IRR) for a given cash flow series.
    Args:
        config (dict): A configuration dictionary containing the key "cash_flow",
            which should be a list or array of cash flow values for each period.
    Returns:
        float: The IRR value as a decimal (e.g., 0.1 for 10%) if a solution is found,
            otherwise NaN if the IRR cannot be computed.
    Notes:
        - Uses the Brent's method to find the root of the NPV function.
        - The search for IRR is performed in the bracket [-10, 10].
        - Returns NaN if the root finding fails or if an exception occurs.
    """

    def npv(irr):
        return sum(cash_flow[year] / ((1 + irr) ** (year + 1)) for year in range(len(cash_flow)))

    try:
        sol = root_scalar(npv, bracket=[-100, 100], method='brentq')
        return sol.root if sol.converged else float('nan')

    except (ValueError, RuntimeError):
        return float('nan')


def roi_calculation(initial_investment, cash_flow, project_lifetime):
    cumulative_net_profit = np.cumsum(cash_flow)[-1] #Cash flow is just the net profit actually.
    roi = cumulative_net_profit/(project_lifetime * initial_investment) *100 #This is the ROI calculated as an average over the whole project

    return roi

