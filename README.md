Python code to align, compare and plot measurements from coincidental tracks from IceSat-2 and CryoSat-2 achieved in the framework of the Cryo2ice project.

sortnsave_cryo2ice_xings.py : is the function that allows to align the data from various CS2 processings with IS2 data (from all strong beams). The output of the function is a pikle file that contains comparable (same dimension) vectors for each required parameter provided in is2_dict.py and cs2_dict.py.

is2_dict.py and cs2_dict.py are dictionnaries that makes the translation from usage parameter names to product parameter names

common_functions.py contains all usefull functions: downloading and reading external data, extracting coordinates or parameters from multiple file types, achieving interpolations
