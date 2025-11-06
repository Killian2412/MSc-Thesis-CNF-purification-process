# CNF-purification-process
This Github page concerns the design and simulation of a process plant for the purification of CMP-produced Carbon nanofibers.

The Python codes for the techno-economic and sensitivity analyses are in the Python Codes folder. Specifically, the following Python codes are included: <br>
-Full_model_CSTR_Serial: Code for running the simulation for the improved baseline with changed reactor setup scenario <br>
-Full_model_Monte_Carlo: Code for running the Monte Carlo simulation for the baseline scenario <br>
-Full_model_Reactor_Analysis: Code fur running the sensitivity analyses of the impact of reactor and feed parameters on the reactor conversion of the baseline scenario <br>
-Full_model_Sensitivity_Analysis: Code for running the sensitivity analyses of the impact of various parameters on the economics of the baseline scenario <br>
-Full_model_Techno_Eco_Baseline: Code for running the simulation for the baseline scenario <br>
-Improved_baseline: Code for running the simulation for the improved baseline scenario, without acid recycle <br>
-baseline_withacidrecycle: Code for running the simulation for the baseline scenario with an acid recycle <br>
-improved_baseline_withacidrecycle: Code for running the simulation for the improved baseline scenario with acid recycle <br>

In the Techno Economic Library, there is a Python file for the techno-economic calculations. This file should be linked to the Python codes. Furthermore, there is a Python file which is only meant for linking with the Full_model_CSTR_Serial code.

In the Aspen Code Library is a file that contains all the code necessary to link Python with Aspen, send input to Aspen and retrieve data from Aspen. Credit is to ....

The Aspen files folder contains all the relevant Aspen files for the Python codes and the kinetics curve fitting codes: <br>
-Batch_reactor_allcompsincluded: Aspen file for the kinetics curve fitting codes <br>
-CSTR_Serial_Testing_Full_Model: Aspen file for the simulation for the improved baseline with changed reactor setup scenario <br>
-Final_Aspen_model: Aspen file for running all the Python codes relating to the techno-ecnomic and sensitivity analyses, except for the improved baseline with changed reactor setup scenario. <br>

Finally, the kinetics curve fit folder contains the Python code used for the fitting of a power law model of acid leaching of nickel to literature data:
-Aspen_k_fitting_validating: Validating the power law model of the acid leaching kinetics and comparison with the literature data.
-Aspen_k_fitting_with353Kline: Curve fitting in Python and Aspen of the acid leaching kinetics of nickel with a power law model. The k value is refitted.
-Python curve fitting: Curve fitting in Python of the acid leaching kinetics of nickel with a power law model.



Upload Aspen
