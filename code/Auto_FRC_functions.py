# Standard Library Imports
import codecs as cd
import copy as copy
import os
import platform
import random
import re
import shutil
import threading
import traceback
import time
from collections import defaultdict
from math import (
    acos, cos, degrees, log, log10, pi, remainder, sin,
    sqrt, tanh
)
from pathlib import Path
from zipfile import ZipFile


# Third-Party Imports
import ipywidgets as widgets
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from ipywidgets import HBox, VBox
from matplotlib.colors import (
    LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
)
from matplotlib.figure import figaspect
from numba import jit
from numpy.polynomial.polynomial import polyfit
from scipy.interpolate import griddata, interp2d
from scipy.linalg import lstsq


# Constants
mu0 = 4 * pi * 1e-7
ms0 = 491880.5
kb = 1.3806503e-23
tau = 10e-9
roottwohffield = 2 ** 0.5


from collections import defaultdict
from pathlib import Path

# extensions to select from input folder
WANTED_EXTS = {
    '.frc': 'frc',   
    '.nrm': 'nrm',
    '.arm': 'arm',
    '.sirm': 'sirm'}


# ===== Auto run code functions =====

def build_site_index(all_data_root, site_folder, exclude_dirs=None):
    """
    sorts through only the specified site folder (all_data/site_folder) and returns
      sample_basename (file ext.) 'gen': [paths], 'nrm': [...], 'arm': [...], 'sirm': [...]
    """
    site_root = Path(all_data_root) / site_folder
    if not site_root.exists() or not site_root.is_dir():
        raise FileNotFoundError(f"Site folder not found: {site_root}")

    exclude_dirs = set() if exclude_dirs is None else set(exclude_dirs)
    index = defaultdict(lambda: defaultdict(list))

    for dirpath, dirnames, filenames in os.walk(site_root):

        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for fname in filenames:
            f_lower = fname.lower()
            for ext, key in WANTED_EXTS.items():
                if f_lower.endswith(ext):
                    sample = f_lower[:-len(ext)]
                    fullpath = os.path.join(dirpath, fname)
                    index[sample][key].append(fullpath)
                    break
    return index


# chooses just one file for each data entry, the most recent file if multiple of the same name
def pick_preferred(entry, prefer_latest=True):
    """Return a dict with single path or None for each extension key."""
    chosen = {}
    for key in set(WANTED_EXTS.values()):
        paths = entry.get(key, [])
        if not paths:
            chosen[key] = None
        else:
            chosen[key] = max(paths, key=os.path.getmtime) if prefer_latest else paths[0]
    return chosen


# sample processing - like findfiles functions in original
def process_sample(X, sample_basename, entry, prefer_latest=True):
    """
    Populate X with file paths and paleo_results.dat handling, then call field_range(X).
    - entry is index[sample_basename]
    """
    sample = sample_basename
    chosen = pick_preferred(entry, prefer_latest=prefer_latest)

    X['names'] = [sample]
    X['fn'] = chosen['frc']
    X['nrm_files'] = [p for p in entry.get('nrm', [])] 
    X['arm_files'] = [p for p in entry.get('arm', [])]
    X['sirm_files'] = [p for p in entry.get('sirm', [])]

        # check all data required is found
    if not X['fn']:
        print(f"[WARN] No FORC (.frc) file for sample {sample}")
    if not X['nrm_files']:
        print(f"[WARN] No NRM file for sample {sample}")
    if not X['arm_files']:
        print(f"[WARN] No ARM file for sample {sample}")
    if not X['sirm_files']:
        print(f"[WARN] No SIRM file for sample {sample}")


    file_exists = os.path.exists('paleo_results.dat')
    sample_copy = 1
    if file_exists:
        with open('paleo_results.dat', 'r') as bigf:
            bigf.readline()
            for my_line in bigf:
                if not my_line.strip():
                    continue
                line = my_line.split('\t')
                if line[0].strip().lower() == sample.lower():
                    sample_copy += 1
                    try:
                        X['min_field'] = float(line[10])
                        X['max_field'] = float(line[11])
                        X['nat_cool'] = float(line[12])
                        X['curie_t'] = float(line[13])
                        X['afval'] = float(line[14])
                        X['SF'] = float(line[15])
                        X['reset_limit_hc'] = float(line[16])
                        X['reset_limit_hi'] = float(line[17])
                        X['sf_list_correct'] = float(line[18])
                    except (IndexError, ValueError):
                        # existing line doesn't have expected columns or is malformed
                        print(f"[WARN] Unexpected paleo_results.dat format for sample {sample} line: {my_line!r}")
                        traceback.print_exc()

        # write output file of important values for each sample
    else:
        with open('paleo_results.dat', 'w') as bigf:
            bigf.write('Sample name \t Repeat \t AF steps \t Range \t AF min \t AF max \t mean PI \t std mean \t median \t iqr median \t Min field \t Max field \t cooling time \t curie temp \t AF value \t SF \t max hc \t max hi \t SF_factor \n')

    X['sample_copy'] = sample_copy

    # create output dir for sample if missing
    outdir = f'{sample}_nrf'
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # NOTE: here I force the min/max field for plotting - original code has user choose
    print(f"Attempt {sample_copy} of running sample {sample}")

    X['min_field'] = 10
    X['max_field'] = 50   
    return X

# function to run all functions
def run_site(site_folder=None, all_data_root=None, exclude_dirs=None, prefer_latest=True):
    """
    Run the pipeline for a specific site folder (e.g. 'HC10') inside all_data.
    If site_folder is None, prompt the user for it.
    Returns a dict: sample_basename -> X dict (results)
    """
    
    # location sanity check
    if all_data_root is None:
        all_data_root = os.path.join(os.getcwd(), 'all_data')

        # allows for user input option of site folder if desired
    if site_folder is None:
        site_folder = input("Enter the site folder name inside 'all_data' (e.g., HC10): ").strip()

    # build index for this site only
    try:
        print(f"Building index for site: {site_folder}")
        index = build_site_index(all_data_root, site_folder, exclude_dirs=exclude_dirs)
    except FileNotFoundError as e:
        print(str(e))
        traceback.print_exc()
        return {}
    
    # error messages for issues
    if not index:
        print("No samples found in this site.")
        traceback.print_exc()
        return {}

    print(f"Found {len(index)} unique sample base names in {site_folder}.")

    for sample in sorted(index.keys()):
        X = {}
        V = {}
        try:
            print(f"\n--- Processing sample {sample} ---")
            # read in the individual data files and sort data
            X = process_sample(X, sample, index[sample], prefer_latest=prefer_latest)
            
            # demag and zijderfeld plots
            X = demag_data_generic_arm(X)
            X = proccess_all(X)
            X = prod_FORCs(X)
            X = plot_zplot(X)

            # FORC distribs
            X = find_plot_fwhm(X)
            X = check_fwhm(X)
            X = pick_SF(X)
            X = divide_mu0(X) 
            X = sym_norm_forcs(X)
            norm_rho_all(X) #keep as was 
            plot_sample_FORC(X['Hc'], X['Hu'], X['rho_n'], X['SF'], X['name'],X['hcmaxa'],X['himaxa'],X['himina'])
            X = user_input(X)
            
            # simulate TRM, SIRM, TRM data, and use measured data to calc paleointensity
            start = time.time()
            thread = ElapsedTimeThread()
            thread.start()
            # X, V = pf.TRM_acq(X, V)
            X, V = TRM_acq_w_arm(X, V)
            time.sleep(0.2)
    
            # something is finished so stop the thread
            thread.stop()
            thread.join()
            print() # empty print() to output a newline
            print("Finished in {:.3f} seconds".format(time.time()-start))
            
            
            while True:
                # produce SIRM and ARM data 
                Xs, Vs = calc_PI_checks_SIRM(V, X)
                fin_pal_SIRM_auto(Xs, Vs)
            
                Xa, Va = calc_PI_checks_ARM(V, X)
                fin_pal_ARM_auto(Xa, Va)
            
                # ask user if they want to accept or redo
                while True:
                    try_again = input(
                        "\nAre you happy with the estimated paleointensities?\n"
                        " Y: proceed to next sample\n"
                        " N: re-select plateaus (re-run)\n"
                        "Enter Y or N: "
                    ).strip().upper()
            
                    if try_again == 'Y':
                        # proceed to next sample -> break outer loop
                        proceed = True
                        break
                    elif try_again == 'N':
                        # re-select plateaus -> repeat outer loop
                        proceed = False
                        break
                    else:
                        print("Not a valid input. Please enter 'Y' or 'N'.")
            
                if proceed:
                    break   # exit the outer while True and continue to next sample
                # else: loop repeats and SIRM/ARM steps re-run


        except Exception as e:
            print(f"[ERROR] Sample {sample} failed: {e}")
            traceback.print_exc()


    print("\nSite run complete.")
    return 


# ====== classes ======

class ElapsedTimeThread(threading.Thread):
    """"Stoppable thread that prints the time elapsed"""
    def __init__(self):
        super(ElapsedTimeThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        thread_start = time.time()
        while not self.stopped():
            print("\rElapsed Time {:.0f} seconds".format(time.time()-thread_start), end="")
            # include a delay here so the thread doesn't uselessly thrash the CPU
            time.sleep(1)
            

    
    
# select field range code:
# ==== not used in this code while forcing the field range in earlier steps =====
    
def field_range(X):
    if (X['sample_copy'] > 1): #only if second go 
        field_1 = input('The most recent field range used for sample {} was {} \u03BCTto {} \u03BCT. If you want to re-use these variables enter K, to change them enter any other charactor'.format(X['names'][0], X['min_field'], X['max_field']))
    else:
        field_1 = 'L'    
    if (field_1 != 'K') or (X['sample_copy'] < 2):
        X['min_field'] = 10
        X['max_field'] = 50
        suggest_field = input("The standard field bounds are {} \u03BCT to {} \u03BCT, if you want to keep these enter K, else enter any other charactor:".format(X['min_field'], X['max_field']))
        #suggest_field = 'K'
        if (suggest_field != 'K'):
            while True:
                    
                    minf = (input("Pick the lower bound of field range to be tested in \u03BCT:" ))
                
                    try:
                        minf = float(minf)
                        if (minf > 0):# (minf <= maxSF)
                            #print('in bounds')
                            break
                    except ValueError:
                        print('Not a number')
                        True

            while True:
                    maxf = (input("Pick the upper bound of field range to be tested in \u03BCT:" ))
                
                    try:
                        maxf = float(maxf)
                        if (maxf > minf):
                            #print('in bounds')
                            break
                    except ValueError:
                        print('Not a number')
                        True
            print('Expected field range: {} \u03BCT to {} \u03BCT'.format(minf, maxf))
            X['max_field'] = maxf
            X['min_field'] = minf
        else:
            pass

    return(X)


# ==== original process data function  =====   
def proccess_all(X):
    """
    Perform full preprocessing of magnetic measurement data for a given sample.
    This includes parsing measurements and calibration, applying unit conversions,
    correcting for drift and slope, removing first and last point artifacts, and
    subtracting the lower branch from FORC data.

    The input dictionary `X` is updated with all relevant data arrays and metadata.

    Parameters
    ----------
    X : dict
        Dictionary containing sample file information and preprocessing options.

    Returns
    -------
    X : dict
        Updated dictionary with preprocessed measurement and calibration data, 
        including drift-corrected, slope-corrected, artifact-cleaned, and lower-branch
        subtracted data.
    """

    # Widget setup for sample preprocessing options
    style = {'description_width': 'initial'}
    fn = X['fn'] 
    sample, unit, mass = sample_details(fn)

    # Text widget for sample name
    sample_widge = widgets.Text(value=sample, description='Sample name:', style=style) 

    # HTML titles for interface
    prop_title = widgets.HTML(value='<h3>Sample preprocessing options</h3>')
    mass_title = widgets.HTML(value='To disable mass normalization use a value of -1')

    # Float widget for sample mass
    if mass == "N/A":
        mass_widge = widgets.FloatText(value=-1, description='Sample mass (g):', style=style)
    else:
        mass_widge = widgets.FloatText(value=mass, description='Sample mass (g):', style=style)
    mass_widge1 = HBox([mass_widge, mass_title])

    # Store widgets and unit in X dictionary
    X["sample"] = sample_widge
    X["mass"] = mass_widge
    X["unit"] = unit

    # Parse measurement and calibration data
    H, Hr, M, Fk, Fj, Ft, dH = parse_measurements(X["fn"])
    Hcal, Mcal, tcal = parse_calibration(X["fn"])

    # Determine measurement field limits
    Hc1, Hc2, Hb1, Hb2 = measurement_limts(X)

    # Store all parsed data in X dictionary
    X["H"] = H
    X["Hr"] = Hr
    X["M"] = M
    X["dH"] = dH
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["Ft"] = Ft
    X["Hcal"] = Hcal
    X["Mcal"] = Mcal
    X["tcal"] = tcal
    X["Hc1"] = Hc1
    X["Hc2"] = Hc2
    X["Hb1"] = Hb1
    X["Hb2"] = Hb2

    # Set slope for slope correction
    slope = 70.
    X["slope"] = slope 
    
    # Convert units from CGS to SI if necessary
    if X['unit'] == 'Cgs':
        X = CGS2SI(X)
		
    # Correct for instrument drift
    X = drift_correction(X) 
   
    # Apply linear slope correction
    X = slope_correction(X)

    # Remove first and last point artifacts from FORC measurements
    X = remove_fpa(X) 
    X = remove_lpa(X)

    # Subtract lower branch from FORC data
    X = lowerbranch_subtract(X)

    return X

    
# ====== original prod_forcs function =======
def prod_FORCs(X):
    """
    Process a FORC dataset dictionary by assigning a readable name,
    preparing calculation arrays, computing FORC distributions at multiple
    smoothing factors, removing invalid values, and rotating the FORC surface.

    Parameters
    ----------
    X : dict
        Dictionary containing metadata and parsed FORC measurement arrays.
        Must include "fn" (filepath to the FORC data).

    Returns
    -------
    dict
        Updated dictionary including computed FORC distributions, processed
        arrays, metadata, and rotated FORC results.
    """

    # Detect operating system to correctly parse file path formatting
    plta = platform.system()

    # Extract filename differently depending on OS path style
    if plta == "Windows":
        name = X['fn'].split("\\")
    else:
        name = X['fn'].split("/") 

    # Remove file extension and store cleaned name
    name2 = name[-1].split('.')
    X['name'] = name2[0]

    # Set maximum smoothing factor to compute FORC distributions
    maxSF = 5
    X['maxSF1'] = maxSF

    # Initialize arrays needed for FORC processing
    X = create_arrays(X, maxSF)

    # Replace invalid or placeholder values (e.g., NaNs)
    X = nan_values2(X)

    # Compute FORC distributions for multiple smoothing factors
    for sf in range(2, maxSF+1):
        X = calc_rho(X, sf)
        sf += 1

    # Clean final FORC array output after computation
    X = nan_values(X, maxSF)

    # Rotate computed FORC surface into standard coordinate space
    X = rotate_FORC(X)

    return(X)

    
# ==== original parse functions ======
    
def parse_header(file, string):
    """
    Search a file header for a line beginning with a specified keyword and
    extract the numeric value following it. The line may contain the value
    after an '=' sign or immediately after the keyword.

    Parameters
    ----------
    file : str
        Path to the file to be scanned.
    string : str
        Header keyword to search for at the beginning of each line.

    Returns
    -------
    float
        The numeric value extracted from the matching header line.
        Returns -1 if no matching value is found.
    """

    # Default return value if no match is found
    output = -1 
    
    # Open the file using latin9 encoding
    with cd.open(file, "r", encoding='latin9') as fp:
        
        # Iterate only over lines that start with the given string
        for line in lines_that_start_with(string, fp):
            
            # Look for an equals sign separating the key and value
            idx = line.find('=')
            
            # If '=' found, parse everything after it as a float
            if idx > -1.:
                output = float(line[idx+1:])
            else:
                # If no '=', assume value follows directly after the keyword
                idx = len(string)
                output = float(line[idx+1:])

    return output



def parse_measurements(file):
    """
    Parse raw FORC measurement data from a .frc file and extract structured FORC arrays.
    
    This function processes magnetic field and magnetization values from a measurement file,
    identifies individual FORC curves, converts units if necessary, determines average
    field spacing, and estimates time stamps for each measurement point.

    Parameters
    ----------
    file : str
        Full file path to the .frc measurement file.

    Returns
    -------
    tuple
        A tuple containing:
        - H (np.ndarray): Applied magnetic field values for all measurements.
        - Hr (np.ndarray): Reversal field values for each measurement point.
        - M (np.ndarray): Magnetization values.
        - Fk (np.ndarray): FORC index number for each point.
        - Fj (np.ndarray): Measurement index within each FORC.
        - Ft (np.ndarray): Measurement time estimates.
        - dH (float): Mean field spacing between points in the final FORC.
    """

    dum=-9999.99 
    N0=int(1E6) 

    # Pre-allocate large arrays for fast file reading (filled with NaNs)
    H0=np.zeros(N0)*np.nan 
    M0=np.zeros(N0)*np.nan 

    # First entry is a dummy marker
    H0[0]=dum 
    M0[0]=dum 

    count=0 # counter used while reading the raw file
    
    # Open measurement file and read only valid data lines
    with cd.open(file,"r",encoding='latin9') as fp: 
        for line in find_data_lines(fp):
            count=count+1
            
            # Locate first comma separating field and magnetization data
            idx = line.find(',') 
            if idx>-1: 
                # Assign field value (first column)
                H0[count]=float(line[0:idx]) 
                line=line[idx+1:] 
                
                # Magnetization value may be followed by another comma
                idx = line.find(',')
                if idx>-1: 
                    M0[count]=float(line[0:idx]) 
                else:
                    # If no more commas, remaining text is the magnetization value
                    M0[count]=float(line) 
            else:
                # Blank or malformed line → store dummy markers
                H0[count]=dum 
                M0[count]=dum

    # Determine start of real data (first non-dummy entry)
    idx_start=np.argmax(H0!=dum) 

    # Strip leading dummy values but keep one placeholder
    M0=M0[idx_start-1:-1] 
    M0=M0[~np.isnan(M0)] # Remove trailing NaNs
    H0=H0[idx_start-1:-1] 
    H0=H0[~np.isnan(H0)]

    ## Detect FORC segment boundaries based on dummy markers
    idxSAT = np.array(np.where(np.isin(H0, dum))) 
    idxSAT = np.ndarray.squeeze(idxSAT) 
    idxSTART = idxSAT[1::2]+1 # First point of each FORC
    idxEND = idxSAT[2::2]-1   # Last point of each FORC

    # Extract the first FORC to initialize output arrays
    M=M0[idxSTART[0]:idxEND[0]+1] 
    H=H0[idxSTART[0]:idxEND[0]+1] 
    Hr=np.ones(idxEND[0]+1-idxSTART[0])*H0[idxSTART[0]] 
    Fk=np.ones(idxEND[0]+1-idxSTART[0]) 
    Fj=np.arange(1,1+idxEND[0]+1-idxSTART[0])

    # Loop through remaining FORCs and append them to output arrays
    for i in range(1,idxSTART.size):
        M=np.concatenate((M,M0[idxSTART[i]:idxEND[i]+1]))
        H=np.concatenate((H,H0[idxSTART[i]:idxEND[i]+1]))
        Hr=np.concatenate((Hr,np.ones(idxEND[i]+1-idxSTART[i])*H0[idxSTART[i]]))
        Fk=np.concatenate((Fk,np.ones(idxEND[i]+1-idxSTART[i])+i))
        Fj=np.concatenate((Fj,np.arange(1,1+idxEND[i]+1-idxSTART[i])))

    # Check measurement unit and convert to SI if needed
    unit = parse_units(file) 
    if unit=='Cgs':
        H=H/1E4 
        Hr=Hr/1E4 
        M=M/1E3 

    # Compute average field spacing using final FORC
    dH = np.mean(np.diff(H[Fk==np.max(Fk)]))

    # Estimate measurement times based on FORC indexing
    Ft=measurement_times(file,Fk,Fj)

    return H, Hr, M, Fk, Fj, Ft, dH


def parse_units(file):
    """
    Parse file header and detect whether magnetic units are SI or CGS.
    Returns 'SI', 'CGS', or None if not detected.
    """

    idxSI = -1
    idxCGS = -1

    for line in file:
        si_pos = line.find("Hybrid SI")
        cgs_pos = line.find("Cgs")

        if si_pos != -1:
            idxSI = si_pos
        if cgs_pos != -1:
            idxCGS = cgs_pos

    # Decide which unit was found
    if idxSI > idxCGS:
        return "SI"
    elif idxCGS > idxSI:
        return "CGS"
    else:
        return "SI"
 

def parse_mass(file):
    """
    Extract the sample mass value from a measurement file header. The function searches
    for a line beginning with the keyword 'Mass' and retrieves the value either after
    an equals sign or directly following the keyword. If no valid mass value is found,
    'N/A' is returned.

    Parameters
    ----------
    file : str
        Path to the measurement file containing mass metadata.

    Returns
    -------
    float or str
        The numeric mass value if found, otherwise the string 'N/A'.
    """

    # Default output if mass cannot be found or is not applicable
    output = 'N/A'

    # Header keyword expected in the file
    string = 'Mass'

    # Open the file in latin9 encoding to ensure consistent parsing
    with cd.open(file, "r", encoding='latin9') as fp:
        
        # Iterate over lines that begin with the keyword
        for line in lines_that_start_with(string, fp):

            # Check for '=' separator format
            idx = line.find('=')

            if idx > -1.:  
                # Extract text after '='
                output = (line[idx+1:])
            else:
                # Extract text immediately after the keyword if '=' is absent
                idx = len(string)
                output = (line[idx+1:])

            # Clean and evaluate the extracted value
            if output.find('N/A') > -1:
                output = 'N/A'
            else:
                # Convert extracted text to numeric value
                output = float(output)

    return output

    
def measurement_times(file, Fk, Fj):
    """
    Estimate measurement time for each FORC point based on instrument header metadata.

    Parameters
    ----------
    file : str or file-like
        Input data file containing FORC measurement header fields.
    Fk : np.ndarray
        FORC index (used to determine reversal field segments).
    Fj : np.ndarray
        Minor loop index / sub-steps within each FORC.

    Returns
    -------
    Ft : np.ndarray
        Estimated measurement time at each data point.
    """

    # Determine units from file (CGS or SI)
    unit = parse_units(file)

    # Parse relevant timing and instrument parameters from header
    string = 'PauseRvrsl'
    tr0 = parse_header(file, string)  # pause at reversal (new format)

    string = 'PauseNtl'
    tr1 = parse_header(file, string)  # pause at reversal (old format)

    # Select whichever pause value is valid (largest non -1)
    tr = np.max((tr0, tr1))

    string = 'Averaging time'
    tau = parse_header(file, string)  # time spent averaging each measurement

    string = 'PauseCal'
    tcal = parse_header(file, string)  # pause at calibration field

    string = 'PauseSat'
    ts = parse_header(file, string)  # pause at saturation field

    string = 'SlewRate'
    alpha = parse_header(file, string)  # magnetic field sweep rate

    string = 'HSat'
    Hs = parse_header(file, string)  # saturation field amplitude

    string = 'Hb2'
    Hb2 = parse_header(file, string)  # high bound of reversal field range

    string = 'Hb1'
    Hb1 = parse_header(file, string)  # low bound of reversal field range

    string = 'Hc2'
    Hc2 = parse_header(file, string)  # coercivity field bound

    string = 'NForc'
    N0 = parse_header(file, string)  # number of FORCs (new format)

    string = 'NCrv'
    N1 = parse_header(file, string)  # number of FORCs (old format)

    # Select valid number of FORCs
    N = np.max((N0, N1))

    # Convert CGS → SI if necessary (Oersted → Tesla)
    if unit == 'Cgs':
        alpha = alpha / 1E4
        Hs = Hs / 1E4
        Hb2 = Hb2 / 1E4
        Hb1 = Hb1 / 1E4

    # Estimate field increment dH using FORC box bounds
    dH = (Hc2 - Hb1 + Hb2) / N

    # Compute coercivity index cutoff
    nc2 = Hc2 / dH

    # Compute fixed timing components from Elgi’s FORC time model
    Dt1 = tr + tau + tcal + ts + 2. * (Hs - Hb2 - dH) / alpha
    Dt3 = Hb2 / alpha

    # Initialize output array for measurement times
    Npts = int(Fk.size)
    Ft = np.zeros(Npts)

    # Loop through each FORC point to compute total time based on formula
    for i in range(Npts):

        # If FORC index is within reversal field range:
        #   use standard timing relationship
        if Fk[i] <= 1 + nc2:
            Ft[i] = (
                Fk[i] * Dt1 + Dt3 + Fj[i] * tau
                + dH/alpha * (Fk[i] * (Fk[i] - 1))
                + (tau - dH/alpha) * (Fk[i] - 1)**2
            )

        # Else: use modified timing formula after coercivity cutoff
        else:
            Ft[i] = (
                Fk[i] * Dt1 + Dt3 + Fj[i] * tau
                + dH/alpha * (Fk[i] * (Fk[i] - 1))
                + (tau - dH/alpha) * ((Fk[i] - 1) * (1 + nc2) - nc2)
            )

    # Return the estimated measurement time array
    return Ft


def parse_calibration(file):
    """
    Parse calibration point data from a FORC measurement file.

    This function extracts field values and magnetizations from the calibration 
    segments embedded in the raw measurement file. The file contains both measurement 
    and calibration records separated by dummy marker values. The function identifies 
    calibration entries, cleans the extracted data, converts units if needed, and 
    returns usable calibration field, magnetization, and timing arrays.

    Parameters
    ----------
    file : str or file-like
        Path or handle to the FORC measurement file.

    Returns
    -------
    Hcal : np.ndarray
        Calibration field values (in Tesla after unit correction).
    Mcal : np.ndarray
        Corresponding calibration magnetization values.
    tcal : np.ndarray
        Estimated measurement timestamps for calibration points.
    """

    dum = -9999.99  # Dummy marker indicating break between FORC measurements and calibration points

    N0 = int(1E6)  # Pre-allocated maximum possible data size (large upper bound)

    # Initialize arrays for field (H) and magnetization (M) with NaN values
    H0 = np.zeros(N0) * np.nan
    M0 = np.zeros(N0) * np.nan

    # First entries are set as dummy markers
    H0[0] = dum
    M0[0] = dum

    count = 0  # Line counter used to populate H and M arrays

    # Open measurement file using compatible encoding
    with cd.open(file, "r", encoding='latin9') as fp:
        # Iterate only through lines containing measurement data
        for line in find_data_lines(fp):
            count += 1  # Increment array index for storing parsed data

            idx = line.find(',')  # Locate first comma separating field and magnetization

            if idx > -1:
                # Extract field value from first column
                H0[count] = float(line[0:idx])

                # Remove processed content and isolate next value(s)
                line = line[idx + 1:]
                idx = line.find(',')

                # Extract magnetization depending on column format
                if idx > -1:
                    M0[count] = float(line[0:idx])
                else:
                    M0[count] = float(line)
            else:
                # Blank or malformed data line → fill with dummy separator values
                H0[count] = dum
                M0[count] = dum

    # Identify start of true measurement data (first non-dummy entry)
    idx_start = np.argmax(H0 != dum)

    # Remove leading dummy values while leaving exactly one at the beginning
    M0 = M0[idx_start - 1:-1]
    M0 = M0[~np.isnan(M0)]  # Remove trailing NaNs

    H0 = H0[idx_start - 1:-1]
    H0 = H0[~np.isnan(H0)]  # Remove trailing NaNs

    # Calibration points occur after alternating dummy markers
    idxSAT = np.array(np.where(np.isin(H0, dum)))  # Locate dummy markers
    idxSAT = np.ndarray.squeeze(idxSAT)            # Flatten index array
    idxSAT = idxSAT[0::2] + 1                      # Select every second dummy + 1 → calibration entry

    # Extract calibration field and magnetization values
    Hcal = H0[idxSAT[0:-1]]
    Mcal = M0[idxSAT[0:-1]]

    # Estimate acquisition time for each calibration record
    tcal = calibration_times(file, Hcal.size)

    # Ensure that returned values use SI units if needed
    unit = parse_units(file)
    if unit == 'Cgs':
        Hcal = Hcal / 1E4  # Convert oersted → tesla
        Mcal = Mcal / 1E3  # Convert emu → A·m²

    return Hcal, Mcal, tcal


def calibration_times(file, Npts):
    """
    Estimate timestamps for calibration measurements in a FORC dataset.

    This function extracts timing-related metadata from the file header and uses 
    instrument settings (pause times, averaging durations, slewrate, saturation 
    field values, and FORC bounding parameters) to reconstruct the approximate 
    time at which each calibration point was measured.

    The computation follows the method described by Elgi et al., and accounts 
    for both legacy and new file format parameter names.

    Parameters
    ----------
    file : str or file-like
        Path to the FORC raw data file.
    Npts : int
        Number of calibration points expected in the dataset.

    Returns
    -------
    tcal_k : np.ndarray
        Array of estimated timestamps (relative time units) for each calibration point.
    """

    unit=parse_units(file)  # Determine whether measurements are in CGS or SI system

    # Calibration timing metadata (may differ across file versions)
    string='PauseRvrsl'  
    tr0=parse_header(file,string)  # Pause duration at reversal field (new format)

    string='PauseNtl'
    tr1=parse_header(file,string)  # Pause duration at reversal field (old format)

    tr=np.max((tr0,tr1))  # Pick whichever format returned a valid value

    string='Averaging time'
    tau=parse_header(file,string)  # Instrument averaging time during acquisition

    string='PauseCal'
    tcal=parse_header(file,string)  # Pause time specifically applied during calibration points

    string='PauseSat'
    ts=parse_header(file,string)  # Pause duration at saturation field

    string='SlewRate'
    alpha=parse_header(file,string)  # Field slewrate used during ramping

    string='HSat'
    Hs=parse_header(file,string)  # Saturation field value

    string='Hb2'
    Hb2=parse_header(file,string)  # Upper Hb boundary of FORC processing box

    string='Hb1'
    Hb1=parse_header(file,string)  # Lower Hb boundary of FORC processing box

    string='Hc2'
    Hc2=parse_header(file,string)  # Upper coercivity limit (Hc1 implied to be zero)

    # Extract number of FORC curves (depends on file format)
    string='NForc'
    N0=parse_header(file,string)

    string='NCrv'
    N1=parse_header(file,string)

    N=np.max((N0,N1))  # Select correct parameter depending on format version

    # Convert to SI units if original data are CGS
    if unit=='Cgs':
        alpha=alpha/1E4  # Slewrate Oe → T
        Hs=Hs/1E4        # Saturation field Oe → T
        Hb2=Hb2/1E4      # Hb bounds Oe → T
        Hb1=Hb1/1E4

    # Estimate spacing between field steps used in FORC scanning
    dH = (Hc2-Hb1+Hb2)/N

    # Following Elgi’s published timing model
    nc2 = Hc2/dH  # Number of steps to reach coercivity limit

    # Fundamental timing expressions for FORC scanning phases
    Dt1 = tr + tau + tcal + ts + 2.*(Hs-Hb2-dH)/alpha
    Dt2 = tr + tau + (Hc2-Hb2-dH)/alpha

    Npts=int(Npts)  # Ensure number of calibration points is integer
    tcal_k=np.zeros(Npts)  # Allocate array for computed timestamps

    # Compute timestamp for each calibration point based on its sequence position
    for k in range(1, Npts+1):

        # First stage timing expression applies until coercivity cutoff is reached
        if k <= 1 + nc2:
            tcal_k[k-1] = (
                k*Dt1 - Dt2
                + dH/alpha*k**2
                + (tau - dH/alpha)*(k-1)**2
            )
        else:
            # Alternate timing model for calibration points beyond Hc boundary
            tcal_k[k-1] = (
                k*Dt1 - Dt2
                + dH/alpha*k**2
                + (tau - dH/alpha)*((k-1)*(1+nc2) - nc2)
            )

    return tcal_k


# ==== simple resued functions (copied from original) ======

def sample_details(fn):
    """
    Extract sample information from a FORC filename, including the parsed sample
    name, measurement units, and recorded mass from the file header.

    Parameters
    ----------
    fn : str
        Filepath to the FORC data file.

    Returns
    -------
    tuple
        sample : str
            The extracted sample name without path or extension.
        units : str
            The measurement unit system (e.g., 'SI' or 'CGS'), parsed from file.
        mass : float or str
            The recorded mass value from the file header, or 'N/A' if not available.
    """

    # Extract the filename from the full path
    sample = fn.split('/')[-1]

    # Remove file extension and keep only the name component
    sample = sample.split('.')

    # Ensure sample is treated as a string (not list)
    if type(sample) is list:
        sample = sample[0]

    # Retrieve units and mass values by reading header metadata
    units = parse_units(fn)
    mass = parse_mass(fn)

    # Return extracted file-based descriptors
    return sample, units, mass


def measurement_limts(X):
    """
    Read measurement boundary values (Hc and Hb limits) from the FORC file header
    defined in dictionary X and convert them to SI units if necessary.

    Parameters
    ----------
    X : dict
        A dictionary containing at minimum:
        - 'fn' : str  → path to the FORC data file
        - 'unit' : str → measurement unit system ('SI' or 'Cgs')

    Returns
    -------
    tuple
        (Hc1, Hc2, Hb1, Hb2)
        where:
        Hc1 : float → lower coercivity boundary
        Hc2 : float → upper coercivity boundary
        Hb1 : float → lower bias field boundary
        Hb2 : float → upper bias field boundary
    """

    # Extract the upper Hb value from file header
    string = 'Hb2'
    Hb2 = parse_header(X["fn"], string)

    # Extract the lower Hb limit
    string = 'Hb1'
    Hb1 = parse_header(X["fn"], string)

    # Extract the upper coercivity limit
    string = 'Hc2'
    Hc2 = parse_header(X["fn"], string)

    # Extract the lower coercivity limit
    string = 'Hc1'
    Hc1 = parse_header(X["fn"], string)

    # Convert values to SI units if file is recorded in CGS
    if X['unit'] == 'Cgs':
        Hc2 = Hc2 / 1E4
        Hc1 = Hc1 / 1E4
        Hb2 = Hb2 / 1E4
        Hb1 = Hb1 / 1E4

    return Hc1, Hc2, Hb1, Hb2


#### Unit conversion ####

def CGS2SI(X):
    
    X["H"] = X["H"]/1E4 #convert Oe into T
    X["M"] = X["M"]/1E3 #convert emu to Am2
      
    return X

#### low-level IO routines

def find_data_lines(fp):

    return [line for line in fp if ((line.startswith('+')) or (line.startswith('-')) or (line.strip()=='') or line.find(',')>-1.)]
    
def lines_that_start_with(string, fp):
 
    return [line for line in fp if line.startswith(string)]

def remove_lpa(X):
    """
    Remove last-point artifacts from FORC measurement data.

    Some FORC acquisition systems record an extra (invalid) final point at the end 
    of each FORC branch. This function identifies and removes those artifact points 
    by examining the maximum reversal index (Fj) within each FORC curve (Fk). The 
    corrected data are written back into the input dictionary.

    Parameters
    ----------
    X : dict
        Dictionary containing FORC dataset arrays. Must include keys:
        "Fj", "H", "Hr", "M", "Fk", "Ft".

    Returns
    -------
    X : dict
        Updated dictionary with artifact points removed and FORC numbering reset.
    """

    # --- Unpack relevant arrays from dictionary for convenience ---
    Fj = X["Fj"]   # Index along each FORC measurement sequence
    H = X["H"]     # Applied field values
    Hr = X["Hr"]   # Reversal fields
    M = X["M"]     # Magnetization values
    Fk = X["Fk"]   # FORC curve index (identifies each reversal curve)
    Ft = X["Ft"]   # Estimated measurement times
    
    # --- Identify FORC count and create mask to track valid points ---
    Nforc = int(np.max(Fk))  # Total number of distinct FORC curves
    W = np.ones(Fk.size)     # Initialize mask (1 = keep, 0 = remove)
    
    # Loop through each FORC curve to find and flag its final measurement point
    for i in range(Nforc):
        Fj_max=np.sum((Fk==i))                   # Count points for FORC i
        idx = ((Fk==i) & (Fj==Fj_max))           # Boolean mask for its final point
        W[idx]=0.0                               # Mark final point as invalid (remove)
    
    # Keep only points that are not flagged as artifacts
    idx = (W > 0.5)
    H=H[idx]
    Hr=Hr[idx]
    M=M[idx]
    Fk=Fk[idx]
    Fj=Fj[idx]
    Ft=Ft[idx]

    # Reset numbering of Fk in case indexing is no longer consecutive
    Fk=Fk-np.min(Fk)+1.
    
    # --- Repack cleaned arrays back into the dictionary ---
    X["Fj"] = Fj
    X["H"] = H
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft        
    
    return X

def remove_fpa(X):
    """
    Remove first-point artifacts from FORC measurement data.

    Some FORC measurement systems record an extra initial point (artifact) at the 
    start of each FORC curve. This function removes those points by identifying 
    rows where Fj == 1.0 (the artifact condition), updates the arrays accordingly, 
    and renumbers Fk and Fj to maintain consistent indexing.

    Parameters
    ----------
    X : dict
        Dictionary containing FORC dataset arrays. Must include keys:
        "Fj", "H", "Hr", "M", "Fk", "Ft".

    Returns
    -------
    X : dict
        Updated dictionary with artifact points removed and indexing corrected.
    """

    # --- Unpack values from dictionary for readability ---
    Fj = X["Fj"]   # Index along each FORC curve
    H = X["H"]     # Applied field values
    Hr = X["Hr"]   # Reversal field values
    M = X["M"]     # Magnetization values
    Fk = X["Fk"]   # FORC curve identifier
    Fj = X["Fj"]   # Repeated assignment (kept unchanged per instruction)
    Ft = X["Ft"]   # Measurement timestamp estimates
    
    # --- Identify and remove first-point artifacts (where Fj == 1.0) ---
    idx = ((Fj == 1.0))  # Boolean mask identifying artifact points
    H = H[~idx]          # Remove artifact rows from applied field array
    Hr = Hr[~idx]        # Remove matching reversal field entries
    M = M[~idx]          # Remove associated magnetization values
    Fk = Fk[~idx]        # Remove corresponding FORC identifiers
    Fj = Fj[~idx]        # Remove matching Fj rows
    Ft = Ft[~idx]        # Remove matching timestamps

    # --- Reset FORC numbering after removal to maintain sequence consistency ---
    Fk = Fk - np.min(Fk) + 1.  # Renumber Fk starting at 1
    Fj = Fj - 1.               # Shift Fj indexing to maintain continuity
    
    # --- Write cleaned values back into dictionary ---
    X["Fj"] = Fj
    X["H"] = H
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft    
    
    return X


def drift_correction(X):
    """
    Apply instrument drift correction to magnetization data.

    This function corrects long-term drift in measured magnetization values by
    interpolating calibration measurements across the measurement timeline and
    scaling the raw magnetization values accordingly.

    Parameters
    ----------
    X : dict
        Dictionary containing measurement data. Must include keys:
        "M", "Mcal", "Ft", and "tcal".
        - M : raw magnetization values
        - Mcal : calibration magnetization values
        - Ft : measurement timestamps
        - tcal : calibration timestamps

    Returns
    -------
    X : dict
        Updated dictionary with drift-corrected magnetization values stored under "M".
    """

    # --- Unpack required variables for clarity ---
    M = X["M"]         # Raw magnetization data
    Mcal = X["Mcal"]   # Calibration magnetization values
    Ft = X["Ft"]       # Measurement timestamps
    tcal = X["tcal"]   # Calibration timestamps
    
    # --- Apply drift correction using time-based calibration interpolation ---
    # The interpolation estimates drift over time, ensuring that magnetization values
    # remain consistent with calibration points.
    M = M * Mcal[0] / np.interp(Ft, tcal, Mcal, left=np.nan)

    # --- Store corrected magnetization back into dictionary ---
    X["M"] = M

    return X


def FORC_extend(X):
    """
    Extend FORC measurement curves by reflecting the first portion of each FORC branch.

    This function generates additional synthetic FORC points for each measurement
    curve by mirroring (reflecting) the first measured segment. The extension enhances
    numerical stability during later processing steps such as smoothing and derivative
    estimation.

    Parameters
    ----------
    X : dict
        Dictionary containing FORC measurement data. Must include fields:
        "H", "Hr", "M", "Fk", "Fj", and "dH".

    Returns
    -------
    X : dict
        Updated dictionary with extended measurement arrays for:
        "H", "Hr", "M", "Fk", and "Fj".
    """

    Ne = 20  # Number of points to extend backward (limit extension length)
    
    # --- Unpack data arrays for readability ---
    H = X["H"]        # Applied field values
    Hr = X["Hr"]      # Reversal fields
    M = X["M"]        # Magnetization values
    Fk = X["Fk"]      # FORC index (branch number)
    Fj = X["Fj"]      # Index along each FORC
    dH = X["dH"]      # Field step size (not modified here, but accessed)

    # --- Loop through each FORC branch and generate synthetic extension data ---
    for i in range(int(X['Fk'][-1])):  # Iterate through each full FORC sequence

        # Extract current FORC branch
        M0 = M[Fk == i+1]        # Magnetization for branch i
        H0 = H[Fk == i+1]        # Field values for branch i
        Hr0 = Hr[Fk == i+1][0]   # Reversal field (same for entire branch)
        
        # Create reflected synthetic extension by reversing tail section
        M1 = M0[0] - (np.flip(M0)[1:] - M0[0])
        H1 = H0[0] - (np.flip(H0)[1:] - H0[0])
            
        # Trim extension if number exceeds allowed limit
        if M1.size > Ne:
            H1 = H1[-Ne-1:-1]
            M1 = M1[-Ne-1:-1]
        
        # Initialize output arrays for first FORC branch
        if i == 0:
            N_new = np.concatenate((M1, M0)).size  # Total number of points after extension
            
            # Build extended series
            H_new = np.concatenate((H1, H0))
            M_new = np.concatenate((M1, M0))
            Hr_new = np.ones(N_new) * Hr0           # Reversal field repeated
            
            Fk_new = np.ones(N_new)                # Branch index (1 for first branch)
            Fj_new = np.arange(N_new) + 1 - M1.size  # Adjust measurement counter indices
        
        # Append subsequent extended branches to existing arrays
        else:
            N_new = np.concatenate((M1, M0)).size
            
            # Append synthetic + original branch data
            H_new = np.concatenate((H_new, H1, H0))
            M_new = np.concatenate((M_new, M1, M0))
            Hr_new = np.concatenate((Hr_new, np.ones(N_new) * Hr0))
            Fk_new = np.concatenate((Fk_new, np.ones(N_new) + i))  # Update branch numbering
            Fj_new = np.concatenate((Fj_new, np.arange(N_new) + 1 - M1.size))

    # --- Store updated extended arrays back into dictionary ---
    X['H'] = H_new
    X['Hr'] = Hr_new
    X['M'] = M_new
    X['Fk'] = Fk_new
    X['Fj'] = Fj_new
    
    return X


def lowerbranch_subtract(X):
    """
    Remove systematic lower branch magnetization offset from FORC measurements.

    This function estimates the lower branch trend in magnetization by locally fitting
    quadratic polynomials across the field range (using a small neighbourhood of FORCs).
    The fitted lower branch is then interpolated and subtracted from the original
    magnetization values to produce a corrected dataset. This improves symmetry and
    prepares the signal for numerical differentiation in FORC diagram construction.

    Parameters
    ----------
    X : dict
        Dictionary of FORC measurement data containing:
        "H", "Hr", "M", "Fk", "Fj", and "dH".

    Returns
    -------
    X : dict
        Updated dictionary with original arrays cleaned of NaN values and an added
        field "DM" containing lower-branch–corrected magnetization.
    """

    # --- Unpack input arrays for easier access ---
    H = X["H"]        # Applied field
    Hr = X["Hr"]      # Reversal field
    M = X["M"]        # Measured magnetization
    Fk = X["Fk"]      # FORC branch number
    Fj = X["Fj"]      # Step index within FORC
    dH = X["dH"]      # Field step size
    
    # Determine full field range
    Hmin = np.min(H)
    Hmax = np.max(H)

    # Number of FORCs to include in smoothing operation
    Nbar = 10

    # Compute number of interpolation points across full H span
    nH = int((Hmax - Hmin) / dH)

    # Create a fine interpolated field grid
    Hi = np.linspace(Hmin, Hmax, nH * 50 + 1)

    # Output array for smoothed lower branch estimate
    Mi = np.empty(Hi.size)
    
    # --- LOESS-like smoothing over local neighbourhoods ---
    for i in range(Hi.size):

        # Select data points within ±2.5 field steps
        idx = (H >= Hi[i] - 2.5 * dH) & (H <= Hi[i] + 2.5 * dH)
        Mbar = M[idx]      # Local magnetization values
        Hbar = H[idx]      # Local applied field values
        Fbar = Fk[idx]     # Local FORC identifiers

        # Identify unique FORC lines in the subset
        F0 = np.sort(np.unique(Fbar))

        # Restrict smoothing to last Nbar FORCs for stability
        if F0.size > Nbar:
            F0 = F0[-Nbar]
        else:
            F0 = np.min(F0)

        # Use only points from selected FORCs
        idx = Fbar >= F0

        # Quadratic polynomial fit for smoothing
        p = np.polyfit(Hbar[idx], Mbar[idx], 2)

        # Evaluate smoothed value at current field point
        Mi[i] = np.polyval(p, Hi[i])
    
    # The lower branch reference curve
    Hlower = Hi
    Mlower = Mi

    # Subtract interpolated lower branch trend from magnetization
    Mcorr = M - np.interp(H, Hlower, Mlower, left=np.nan, right=np.nan)

    # --- Remove NaN values introduced during interpolation ---
    Fk = Fk[~np.isnan(Mcorr)]
    Fj = Fj[~np.isnan(Mcorr)]
    H = H[~np.isnan(Mcorr)]
    Hr = Hr[~np.isnan(Mcorr)]
    M = M[~np.isnan(Mcorr)]
    Mcorr = Mcorr[~np.isnan(Mcorr)]
    
    # --- Repack updated values into dictionary ---
    X["H"] = H
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["DM"] = Mcorr  # Store drift-corrected magnetization
    
    return X


###### HELPER FUNCTIONS TO READ FROM FILE

def slope_correction(X):
  
    #unpack
    H = X["H"]
    M = X["M"]
  

    Hidx = H > (X["slope"]/100) * np.max(H)
    p = np.polyfit(H[Hidx],M[Hidx],1)
    M = M - H*p[0]
  
    #repack
    X["M"]=M
  
    return X

def create_arrays(X, maxSF):
    """
    Reshape parsed FORC measurement vectors into structured 2D arrays and initialize
    storage for FORC density (rho) values across multiple smoothing factors.

    This function groups data by FORC index (`Fk`) and populates row/column formatted
    arrays corresponding to individual reversal curves. It also allocates memory
    for later processing steps including FORC smoothing and interpolation.

    Parameters
    ----------
    X : dict
        Dictionary containing parsed FORC measurement vectors including:
        - 'H'  : applied field values
        - 'Hr' : reversal field values
        - 'M'  : magnetization values
        - 'Fk' : FORC index identifiers
    maxSF : int
        Maximum smoothing factor for which rho arrays should be allocated.

    Returns
    -------
    dict
        The input dictionary `X` updated with new structured FORC arrays and metadata.
    """

    # Convert FORC index labels to integer for counting occurrences
    Fk_int = (X['Fk'].astype(int))

    # Count number of measurements in each FORC curve
    counts = np.bincount(Fk_int)

    # The longest FORC defines maximum number of columns needed
    max_FORC_len = np.max(counts)

    # Number of FORCs is determined by the count distribution's maximum index
    no_FORC = np.argmax(counts)

    # Initialize empty 2D arrays for structured FORC representation
    H_A = np.zeros((no_FORC, max_FORC_len))
    Hr_A = np.zeros((no_FORC, max_FORC_len))
    M_A = np.zeros((no_FORC, max_FORC_len))
    Fk_A = np.zeros((no_FORC, max_FORC_len))

    # Allocate rho array for storing density results across smoothing factors
    Rho = np.zeros((maxSF+1, no_FORC, max_FORC_len))

    # Seed arrays with the first available data point
    H_A[0,0] = X['H'][0]
    Hr_A[0,0] = X['Hr'][0]
    M_A[0,0] = X['M'][0]
    Fk_A[0,0] = X['Fk'][0]

    # Row (i) and column (j) indices for filling structured arrays
    j = 0
    i = 0

    # Iterate through flattened dataset to populate matrix form
    for cnt in range(1, len(X['Fk']+1)):

        # If the FORC index hasn't changed, continue placing data along same row
        if (X['Fk'][cnt] == X['Fk'][cnt-1]):
            j += 1  # Next column position

            H_A[i][j] = X['H'][cnt]
            Hr_A[i][j] = X['Hr'][cnt]
            M_A[i][j] = X['M'][cnt]

        else:
            # New FORC encountered → move to next row
            i += 1
            j = 0 

            # Store first point for new FORC row
            H_A[i][j] = X['H'][cnt]
            Hr_A[i][j] = X['Hr'][cnt]
            M_A[i][j] = X['M'][cnt]

        cnt += 1  # advance pointer

    # Store structured arrays and metadata back into dictionary
    X['H_A'] = H_A
    X['Hr_A'] = Hr_A
    X['M_A'] = M_A
    X['rho'] = Rho

    # Add FORC layout properties for use in later processing
    X['no_FORC'] = no_FORC
    X['max_FORC_len'] = max_FORC_len

    return(X)


def nan_values2(X):
    """
    Replace zero entries in structured FORC arrays with string 'NaN' to mark
    invalid or uninitialized points. This helps prevent accidental use of
    placeholder zeros in later processing steps.

    Parameters
    ----------
    X : dict
        Dictionary containing structured FORC arrays:
        - 'H_A'  : applied field array
        - 'Hr_A' : reversal field array
        - 'M_A'  : magnetization array

    Returns
    -------
    dict
        Updated dictionary with zero values replaced by 'NaN' strings.
    """

    # Extract structured arrays
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    M_A = X['M_A']

    # Iterate over all rows and columns
    for i in range(len(H_A)):
        for j in range(len(Hr_A[0])):
            # Replace zeros with 'NaN' to indicate missing/uninitialized data
            if (H_A[i][j] == 0.0):
                H_A[i][j] = 'NaN'
                Hr_A[i][j] = 'NaN'
                M_A[i][j] = 'NaN'

    # Update dictionary with modified arrays
    X['H_A'] = H_A
    X['Hr_A'] = Hr_A
    X['M_A'] = M_A

    return(X)


def calc_rho(X, SF):  # no rotation, that comes later.
    """
    Calculate the FORC distribution (rho) for a given smoothing factor (SF)
    using local polynomial fitting over a smoothing window around each data point.

    Parameters
    ----------
    X : dict
        Dictionary containing structured FORC arrays:
        - 'H_A'  : 2D array of applied fields
        - 'Hr_A' : 2D array of reversal fields
        - 'M_A'  : 2D array of magnetization values
        - 'rho'  : 3D array preallocated for FORC distribution (smoothing factor x rows x columns)
        - 'no_FORC' : int, number of FORCs (rows)
        - 'max_FORC_len' : int, maximum length of a FORC (columns)
    SF : int
        Smoothing factor defining the size of the local fitting window.

    Returns
    -------
    dict
        Updated dictionary with calculated FORC distribution in 'rho'.
    """

    # Extract relevant arrays and dimensions
    no_FORC = X['no_FORC']
    max_FORC_len = X['max_FORC_len']
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    M_A = X['M_A']
    Rho = X['rho']

    # Loop over all FORC rows
    for i in range(no_FORC): 
        # Loop over all columns within a FORC
        for j in range(max_FORC_len): 

            cnt = 0  # counter for points within smoothing window

            # Define smoothing window boundaries
            h1 = min(i, SF)  # rows above
            h2 = min(SF, (no_FORC - i))  # rows below
            k1 = min(j, SF)  # columns to left
            k2 = min(SF, (max_FORC_len - j))  # columns to right

            # Initialize matrices for local least squares fitting
            A = np.zeros(((h2+h1+1)*(k1+k2+1), 6))
            b = np.zeros(((h2+h1+1)*(k1+k2+1)))
            A[:, :] = np.nan
            b[:] = np.nan

            # Only compute rho if point is above reversal field
            if (H_A[i][j] > Hr_A[i][j]):
                # Loop over smoothing window
                for h in range((-h1), (h2+1)):  
                    for k in range((-k1), (k2+1)):  
                        if ((j+h+k) >= 0 and (j+k+h) < (max_FORC_len) and (i+h) >= 0 and (i+h) < (no_FORC)): 
                            # Fill matrix A with polynomial terms for local fit
                            A[cnt, 0] = 1.
                            A[cnt, 1] = Hr_A[i+h][j+k+h] - Hr_A[i][j]
                            A[cnt, 2] = (Hr_A[i+h][j+k+h] - Hr_A[i][j])**2.
                            A[cnt, 3] = H_A[i+h][j+k+h] - H_A[i][j]
                            A[cnt, 4] = (H_A[i+h][j+k+h] - H_A[i][j])**2.
                            A[cnt, 5] = (Hr_A[i+h][j+k+h] - Hr_A[i][j]) * (H_A[i+h][j+k+h] - H_A[i][j])
                            b[cnt] = M_A[i+h][j+k+h]

                            cnt += 1  # increment point counter

                # Remove rows with NaNs
                A = A[~np.isnan(A).any(axis=1)]
                b = b[~np.isnan(b)]

                # Solve least squares if enough points exist
                if (len(A) >= 2):  
                    dmatrix, res, rank, s = lstsq(A, b)
                    Rho[SF][i][j] = (-1. * (dmatrix[5])) / 2.
                else:
                    Rho[SF][i][j] = 0.
            else:
                Rho[SF][i][j] = 0.

            j += 1
        i += 1

    # Repack arrays back into dictionary
    X['H_A'] = H_A
    X['Hr_A'] = Hr_A
    X['M_A'] = M_A
    X['rho'] = Rho
    X['no_FORC'] = no_FORC
    X['max_FORC_len'] = max_FORC_len

    return(X)
    
    
def nan_values(X, maxSF):
    """
    Replace zero entries in structured FORC arrays and the rho arrays with 'NaN'
    to mark missing or uninitialized data. This ensures that zeros are not
    misinterpreted as valid measurements during further processing.

    Parameters
    ----------
    X : dict
        Dictionary containing structured FORC arrays:
        - 'H_A'  : 2D array of applied field values
        - 'Hr_A' : 2D array of reversal field values
        - 'rho'  : 3D array of FORC distributions across smoothing factors
    maxSF : int
        Maximum smoothing factor used in the rho arrays.

    Returns
    -------
    dict
        Updated dictionary with zero values replaced by 'NaN' strings.
    """

    # Extract arrays from dictionary
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    Rho = X['rho']

    # Replace zeros in field arrays with 'NaN'
    for i in range(len(H_A)):
        for j in range(len(Hr_A[0])):
            if (H_A[i][j] == 0.0):
                H_A[i][j] = 'NaN'
                Hr_A[i][j] = 'NaN'

    # Replace zeros in all smoothing factor rho arrays with 'NaN'
    for k in range(maxSF+1):
        for i in range(len(H_A)):
            for j in range(len(Hr_A[0])):
                if (Rho[k][i][j] == 0.0):
                    Rho[k][i][j] = 'NaN'

    # Update dictionary with modified arrays
    X['H_A'] = H_A
    X['Hr_A'] = Hr_A
    X['rho'] = Rho

    return(X)
 
def rotate_FORC(X):
    """
    Computes the rotated FORC coordinate system (Hc, Hu) from raw measurement axes.

    This function transforms the applied field (`H_A`) and reversal field (`Hr_A`)
    into coercivity (Hc) and interaction field (Hu) coordinates using standard
    FORC coordinate rotation equations. It also stores the maximum and minimum
    values (scaled to mT) for plotting purposes.

    Parameters
    ----------
    X : dict
        Dictionary containing FORC arrays including:
        - 'H_A' : Applied field values
        - 'Hr_A' : Reversal field values

    Returns
    -------
    dict
        Updated dictionary `X` including:
        - 'Hc' : Coercivity axis
        - 'Hu' : Interaction field axis
        - 'hcmaxa' : Maximum coercivity value (mT)
        - 'himaxa' : Maximum interaction field value (mT)
        - 'himina' : Minimum interaction field value (mT)
    """

    # Extract applied and reversal field data from dictionary
    H_A = X['H_A']
    Hr_A = X['Hr_A']

    # Compute coercivity axis: midpoint difference
    Hc = (H_A - Hr_A) / 2.  # x-axis in FORC space

    # Compute interaction field axis: midpoint sum
    Hu = (H_A + Hr_A) / 2.  # y-axis in FORC space

    # Store rotated coordinates back into dictionary
    X['Hc'] = Hc
    X['Hu'] = Hu

    # Store useful extrema (converted to mT) for plotting axis limits
    X['hcmaxa'] = np.nanmax(Hc) * 1000
    X['himaxa'] = np.nanmax(Hu) * 1000
    X['himina'] = np.nanmin(Hu) * 1000

    return (X)

def half_max_test(fwHu_c, fwRho_c, ym):
    arr_L = np.where(fwRho_c == ym)[0]
    L = arr_L[0]
    half_ym = ym/2. #half max
    b = L+1

    while (b < len(fwRho_c)):

        if(fwRho_c[b] < half_ym):
            
            break
        b = b + 1
    
    top = fwRho_c[b-1] - fwRho_c[b]
    bot = fwHu_c[b-1] - fwHu_c[b]
   
    mo_test = top/bot
    r0 = fwHu_c[b] + ((half_ym - fwRho_c[b])/mo_test)
    #print(r0, mo_test, )
    u = L-1

    while (u > 0): 
       
        if (fwRho_c[u] < half_ym):
            
            break
        u = u - 1
    
    #interpolation to get half maximum for each in hu 
    m1 = (fwRho_c[u] - fwRho_c[u+1])/(fwHu_c[u] - fwHu_c[u+1])

    r1 = fwHu_c[u+1] + ((half_ym - fwRho_c[u+1])/m1)
  
    fwhm = r1 - r0
   
    return fwhm, r0, r1


def find_fwhm2(X, SF, sample_name): 
    """
    Calculate the Full Width at Half Maximum (FWHM) along a selected slice of a 
    FORC distribution for a given smoothing factor (SF). Updates the dictionary `X`
    with FWHM values, half-maximum points, and plots the cross-section of the peak.

    Parameters
    ----------
    X : dict
        Dictionary containing FORC data and results, including:
        - 'rho' : 3D array of FORC distributions (smoothing factor x rows x columns)
        - 'Hc'  : 2D array of coercivity values
        - 'Hu'  : 2D array of interaction/bias field values
        - 'fwhmlist' : list to store calculated FWHM values
        - 'fwRho_L' : list to store FWHM slices of rho
        - 'fwHu_L' : list to store FWHM slices of Hu
        - 'Hu_rs' : list to store half-maximum Hu positions
        - 'Rho_half' : list to store half-maximum values
    SF : int
        Smoothing factor to select which rho layer to analyze.
    sample_name : str
        Name of the sample (used for labeling/plotting, not used in calculations).

    Returns
    -------
    dict
        Updated dictionary `X` with FWHM measurements, half-maximum positions, 
        and plot added.
    """

    # Retrieve existing FWHM list and rho array for the selected smoothing factor
    fwhmlist = X['fwhmlist']
    Rho = X['rho'] 
    Hc = X['Hc']   
    Hu = X['Hu']

    # Find the location of the maximum value in the rho slice
    indices = np.unravel_index(np.nanargmax(Rho[SF]), Rho[SF].shape)

    # Extract a vertical cross-section at the maximum column
    fwHu = []
    fwRho = []
    fwHc = []
    for i in range(len(Rho[SF])):
        fwHu.append(Hu[i][indices[1]]) 
        fwHc.append(Hc[i][indices[1]]) 
        fwRho.append(Rho[SF][i][indices[1]])
        i += 1
        
    # Convert lists to arrays for numerical operations
    fwHu = np.array(fwHu)
    fwRho = np.array(fwRho)
    fwHc = np.array(fwHc)
    
    # Remove NaNs across all arrays at the same positions
    data = {"fwHu": fwHu, "fwHc": fwHc, "fwRho": fwRho}
    mask = ~np.any([np.isnan(data[key]) for key in data], axis=0)
    cleaned_data = {key: data[key][mask] for key in data}
    fwHc = cleaned_data['fwHc']
    fwHu = cleaned_data['fwHu']
    fwRho = cleaned_data['fwRho']
    
    # Initialize half-maximum positions
    r0 = 1
    r1 = -1
    
    # Find index of value closest to zero on the x-axis (Hu)
    loc_o = np.argmin(abs(fwHu))
    
    # Select portion of the cross-section for FWHM analysis
    fwHu_f = fwHu[:loc_o] 
    fwRho_f = fwRho[:loc_o]

    # Locate minimal point of cross-section
    loc_m = np.argmin(abs(fwRho_f))
    loc_n = (loc_o + 2 + (loc_o - loc_m))
    if loc_n < 7:
        loc_n = 7

    # Extract centered portion for FWHM calculation
    fwHu_c = fwHu[loc_m:loc_n]    
    fwRho_c = fwRho[loc_m:loc_n]
    X['fwRho_L'].append(fwRho_c)
    X['fwHu_L'].append(fwHu_c) 

    # Sort cross-section to find maximum value
    m_rho_a = np.sort(fwRho_c)

    # Loop until valid half-maximum points are found
    i = 1
    while ((r0 > 0) or (r1 < 0)):
        ym = m_rho_a[-i]  # take peak value
        try:
            # Compute FWHM and half-maximum points using helper function
            fwhm, r0, r1 = half_max_test(fwHu_c, fwRho_c, ym)
        except:
            print('Error in calculating FWHM for SF %', SF)
            pass
        
        # Stop if too noisy
        if (i > 5):
            print('SF {} is too noisy'.format(SF))
            fwhm = 'Nan'
            r0 = 'NaN'
            r1 = 'NaN'
            break
        i += 1
  
    # Append calculated FWHM to list
    fwhmlist.append(fwhm)

    # Calculate half-maximum value for plotting
    half = max(fwRho_c)/2.0
    X['Hu_rs'].append([r0, r1])
    X['Rho_half'].append([half, half])

    # Plot cross-section with half-maximum line
    plt.plot([r0, r1], [half, half], label=SF)
    plt.xlabel(r'$\\mathrm{h_{s}}$ (mT)')
    plt.ylabel('FORC weighting')
    plt.legend()
    plt.title('Plot of the cross sections of the FWHM at each smoothing factor')
    plt.show

    # Update dictionary
    X['fwhmlist'] = fwhmlist

    return(X)
  

  
def find_plot_fwhm(X):
    """
    Calculate and plot FWHM cross-sections for all smoothing factors (SF) of a sample.
    Generates two plots:
      1. Cross-sections of the FORC distribution at each SF
      2. SF vs FWHM trend with linear fit
    """
    # Ask whether to reuse previous SF
    if X['sample_copy'] > 1:
        sf_1 = input(f"Previous SF for sample {X['names'][0]} was {X['SF']}. Enter K to reuse or any other key to recalc: ")
    else:
        sf_1 = 'L'

    if sf_1 != 'K' or X['sample_copy'] < 2:
        maxSF = 5
        SFlist = list(range(2, maxSF+1))
        X['SF_list'] = SFlist

        X['fwhmlist'] = []
        X['fwRho_L'] = []
        X['fwHu_L'] = []
        X['Hu_rs'] = []
        X['Rho_half'] = []

        # Set up figure with two subplots
        plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        color_cycle = plt.cm.tab10.colors  # 10 distinct colors

        # Plot cross-sections for each SF
        for i, SF in enumerate(SFlist):
            X = find_fwhm2(X, SF, X['name'])

            # Convert hu_rs if valid
            hu_rsP = np.array(X['Hu_rs'][i])
            if hu_rsP[0] != 'NaN':
                ax1.plot(hu_rsP*1000, X['Rho_half'][i], color=color_cycle[i % 10], alpha=0.6, lw=1.5)

            # FWHM cross-section
            ax1.plot(np.array(X['fwHu_L'][i])*1000, X['fwRho_L'][i], color=color_cycle[i % 10],
                     label=f'SF={SF}', marker='s', markersize=4, linewidth=1.5)

        ax1.set_xlabel(r'$\mathrm{h_s}$ (mT)')
        ax1.set_ylabel('FORC weighting')
        ax1.set_title('FWHM Cross-Sections at Different SF')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(title='Smoothing Factor')

        # Part 2: Plot SF vs FWHM trend
        SFlist = X['SF_list']
        fwhmlist = X['fwhmlist']
        st_line_SFlist = []
        polyfwhm = []
        polySF = []

        maxSF1 = X['maxSF1']
        for i in range(maxSF1+1):
            st_line_SFlist.append(i)
            i += 1
        st_line_SFlist = np.array(st_line_SFlist)
        SFlist = np.array(SFlist)

        # Filter valid FWHM values
        # Part 2: SF vs FWHM trend with same colors as plot 1
        polySF = []
        polyfwhm = []
        
        # Filter valid FWHM values
        for i, f in enumerate(fwhmlist):
            if f not in ['NaN', 'Nan']:
                polySF.append(SFlist[i])
                polyfwhm.append(float(f))
        
        polySF = np.array(polySF)
        polyfwhm = np.array(polyfwhm)
        
        # Linear fit
        b, m = polyfit(polySF, polyfwhm, 1)
        ax2.cla()
        
        # Plot linear fit line
        st_line_SFlist = np.arange(X['maxSF1'] + 1)
        ax2.plot(st_line_SFlist, (b + m*st_line_SFlist)*1000, color='black', lw=0.5, linestyle='-')
        
        # Plot points color-coded the same as plot 1
        for i, sf in enumerate(polySF):
            color = color_cycle[int(sf-2) % 10]  # same mapping as plot1 (SF 2->0, 3->1 ...)
            ax2.scatter(sf, polyfwhm[i]*1000, color=color, s=60, label=f'SF={SF}')
        
   
        ax2.set_title('Smoothing factor (SF) vs FWHM using SF 2-5')
        ax2.set_xlabel('SF')
        ax2.set_ylabel('FWHM (mT)')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        X['b'] = b
        plt.show()


    # Store user choice for SF reuse
    X['sf_1'] = sf_1
    return X

def check_fwhm(X):
    """
    Verify the reliability of calculated FWHM values for different smoothing factors (SF)
    and allow the user to mark unreliable values. Updates the FWHM list in `X` and
    optionally replots the FWHM.

    Parameters
    ----------
    X : dict
        Dictionary containing:
        - 'sf_1' : user choice whether to reuse previous SF values ('K' to reuse)
        - 'SF_list' : list of smoothing factors used
        - 'fwhmlist' : list of FWHM values corresponding to each SF
        - 'maxSF1' : maximum SF allowed
        - Other keys used by `plot_fwhm`

    Returns
    -------
    dict
        Updated dictionary `X` with potentially corrected FWHM values and plots.
    """

    # Only prompt user if previous SF was not reused
    if (X['sf_1'] != 'K'):
        SFlist = X['SF_list']
        answer = None
        answer2 = None
        fwhmlist = X['fwhmlist']
        maxSF1 = X['maxSF1']

        # Loop until user confirms all FWHM values are reliable
        while answer not in ("Y", "N"):
            answer = input("Are all of the FWHM reliable? Enter Y or N: ")

            # If some FWHM values are unreliable
            if (answer == "N"):
                sf_int = 0
                # Ask user to select which SF is unreliable
                while (sf_int == 0):
                    try:
                        sf_pick = int(input("Which SF is unrealiable and needs to be removed?:"))
                        sf_int = 1
                        if not (2 <= sf_pick <= maxSF1):
                            sf_int = 0
                            print('Not an interger between 2 and 5. Please input an interger between 2 and 5.')
                    except ValueError:
                        print('Not an interger. Please input an interger between 2 and 5.')

                # Check if there are any additional unreliable FWHM values
                while answer2 not in ("Y", "N"):
                    answer2 = input("Are any other FWHM unreliable? Enter yes or no: ")
            
                    if (answer2 == "Y"):
                        sf_int2 = 0
                        while (sf_int2 == 0):
                            try:
                                sf_pick2 = int(input("Which other SF is unrealiable and needs to be removed?:"))
                                sf_int2 = 1
                                if not (2 <= sf_pick2 <= maxSF1):
                                    sf_int2 = 0
                                    print('Not an interger between 2 and 5. Please input an interger between 2 and 5.')
                            except ValueError:
                                print('Not an interger. Please input an interger between 2 and 5.')

                    elif (answer2 == "N"):
                        print(answer2)
                    
                    elif (isinstance(answer2, str)):
                        print("Please enter Y or N.")

                # Mark the unreliable FWHM values
                fwhmlist[sf_pick-2] = 'che' 
                if (answer2 == "Y"):
                    fwhmlist[sf_pick2-2] = 'che' 

                X['fwhmlist'] = fwhmlist

                # Replot the updated FWHM values
                X = plot_fwhm(X) 
                
            # If all FWHM values are reliable, just plot
            elif answer == "Y":
                #X = plot_fwhm(X)
                continue
            
            elif (isinstance(answer, str)):
                print("Please enter yes or no.")

        # Convert FWHM list to NumPy array for further processing
        fwhmlist = np.array(fwhmlist)
        X['fwhmlist'] = fwhmlist

    return(X)

def plot_fwhm(X):
    """
    Plot FWHM versus smoothing factor (SF) and replace unreliable FWHM values
    using a linear fit.

    This function:
      - Filters out unreliable or missing FWHM values marked as 'che' or 'Nan'
      - Performs a linear fit of the reliable FWHM values against SF
      - Plots the FWHM vs SF with the fit line
      - Replaces any unreliable FWHM values with predicted values from the fit

    Parameters
    ----------
    X : dict
        Dictionary containing:
        - 'SF_list': list of smoothing factors used
        - 'fwhmlist': list of FWHM values corresponding to each SF
        - 'maxSF1': maximum SF value
        - 'Hu': FORC Hu array (used here for reference, not modified)

    Returns
    -------
    dict
        Updated dictionary `X` with corrected FWHM values and plot.
    """

    # Extract lists from dictionary
    SFlist = X['SF_list']
    fwhmlist = X['fwhmlist']
    st_line_SFlist = []
    polyfwhm = []
    polySF = []

    maxSF1 = X['maxSF1']

    # Create a line of SF indices for plotting the linear fit
    for i in range(maxSF1+1):
        st_line_SFlist.append(i)
        i += 1

    st_line_SFlist = np.array(st_line_SFlist)
    SFlist = np.array(SFlist)

    # Filter out unreliable FWHM values ('che' or 'Nan') for fitting
    for i in range(len(SFlist)):
        if (fwhmlist[i] != 'che') and (fwhmlist[i] != 'Nan'):
            polyfwhm.append(float(fwhmlist[i]))
            polySF.append(float(SFlist[i]))
    
    poly_fwhm_plot = np.array(polyfwhm)

    # Perform linear fit of reliable FWHM values vs SF
    b, m = polyfit(polySF, poly_fwhm_plot, 1)
    X['b'] = b

    # Plot scatter and linear fit line
    plt.scatter(polySF, poly_fwhm_plot*1000)
    bplot, mplot = polyfit(polySF, (poly_fwhm_plot*1000), 1)
    plt.scatter(polySF, poly_fwhm_plot*1000)
    plt.plot(st_line_SFlist, bplot + mplot * st_line_SFlist, '-')
    plt.xlabel('SF')
    plt.ylabel('FWHM (mT)')
    plt.title('Smoothing factor (SF) vs FWHM using SF 2-5')
    plt.show 

    Hu = X['Hu']

    # Replace unreliable FWHM values with values predicted by the linear fit
    for i in range(len(SFlist)):
        if (fwhmlist[i] == 'Nan') or (fwhmlist[i] == 'che'):
            fwhmlist[i] = float(m*SFlist[i] + b)

    X['fwhmlist'] = fwhmlist

    return(X)


def norm_rho_all(X):
    Rho = X['rho']
    Rho_n = np.copy(Rho)
    a, x, y = np.shape(Rho)
    X['max_Rho'] = np.zeros((a))


    k=0
    for k in range(2,a):
        i = 0
        j = 0
        max_Rho = np.nanmax(Rho_n[k])

        for i in range(x):
            for j in range(y):
                Rho_n[k][i][j] = Rho[k][i][j]/max_Rho
        X['rho_n'] = Rho_n
        X['max_Rho'][k] = max_Rho
        k+=1

    return(X)    

  
def plot_sample_FORC(x, y, z, SF, sample_name, hcmaxa, himaxa, himina):
    """
    Plot a neat FORC (First-Order Reversal Curve) diagram for a given sample and smoothing factor.
    
    This function:
    - Converts Hc and Hu axes to mT for plotting
    - Plots filled contours for Rho using FORCinel colormap
    - Adds contour lines for clarity
    - Adds a dashed x=y reference line
    - Sets axis labels, limits, title, and colorbar
    - Saves the plot as a PDF
    
    Parameters
    ----------
    x : array-like
        Hc values (centered coercivity) for the FORC diagram.
    y : array-like
        Hu values (interaction field) for the FORC diagram.
    z : dict or list of 2D arrays
        Rho arrays corresponding to different smoothing factors (SF).
    SF : int
        Smoothing factor to select from z.
    sample_name : str
        Name of the sample for file naming and title.
    hcmaxa : float
        Maximum Hc value for x-axis.
    himaxa : float
        Maximum Hu value for y-axis (often 0 or positive).
    himina : float
        Minimum Hu value for y-axis (often negative).
    """

    # Select Rho data for the requested smoothing factor and copy to avoid modifying original
    z_sf = z[SF]
    zn = np.copy(z_sf)

    # Convert axes to mT
    xp = x * 1000
    yp = y * 1000

    # Define contour levels for overlay lines
    con = np.linspace(0.1, 1, 9)

    # Get FORCinel colormap and min/max values for normalization
    cmap, vmin, vmax = FORCinel_colormap(zn)
    print(f"Colormap vmin={vmin}, vmax={vmax}")

    # Set up figure with consistent sans-serif fonts
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filled contour plot of Rho
    cf = ax.contourf(xp, yp, zn, 100, cmap=cmap, vmin=vmin, vmax=vmax)

    # Overlay contour lines for clarity
    ax.contour(xp, yp, zn, con, colors='k', linewidths=0.6)

    # Set axis labels
    ax.set_xlabel(r'$\mathrm{h_{c}}$ (mT)', fontsize=14)
    ax.set_ylabel(r'$\mathrm{h_{s}}$ (mT)', fontsize=14)

    # Set axis limits
    ax.set_xlim(0, hcmaxa)
    ax.set_ylim(himina, himaxa)

    # plot diag
    ax.plot([0, hcmaxa], [0, himina], 'k--', linewidth=1, label='$H_a$') 
    ax.plot([0, hcmaxa], [0, hcmaxa], 'k-', linewidth=1, label='$H_b$') 
 
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add colorbar for Rho
    cbar = fig.colorbar(cf, ax=ax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Normalized Rho', fontsize=12)

    # Set plot title
    ax.set_title(f'{sample_name} FORC diagram, SF = {SF}', fontsize=16)

    # Add grid with dashed lines and slight transparency
    ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout for spacing
    plt.tight_layout()
    plt.legend()

    # Construct filename and ensure folder exists
    filename = os.path.join(f'{sample_name}',
                            f'FORC_diagram_sample_{sample_name}_SF{SF}.pdf')
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the figure as PDF and show
    plt.savefig(filename)
    plt.show()
    plt.pause(0.1)

    return


def divide_mu0(X):
    mu0 = mu0=4*pi*1e-7
    X['Hc_mu'] = X['Hc']/mu0

    X['Hu_mu'] = X['Hu']/mu0
    
    return(X)



def FORCinel_colormap(Z):
    """
    Generate a FORCinel-style colormap for visualizing FORC diagrams.

    The function sets up a LinearSegmentedColormap for positive and negative Rho values,
    optionally extending the negative range if needed. It also defines color anchors 
    to enhance contrast near zero.

    Parameters
    ----------
    Z : array-like
        The 2D FORC rho data array used to determine colormap limits.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        The FORCinel colormap for plotting.
    vmin : float
        Minimum value for color scaling.
    vmax : float
        Maximum value for color scaling.
    """

    # Setup initial colormap with predefined red, green, blue segments
    cdict = {
        'red': (
            (0.0, 127/255, 127/255),
            (0.1, 255/255, 255/255),
            (0.1597, 255/255, 255/255),
            (0.3, 255/255, 255/255),  # Wider white region (extended red)
            (0.5, 102/255, 102/255),
            (0.563, 204/255, 204/255),
            (0.6975, 204/255, 204/255),
            (0.8319, 153/255, 153/255),
            (0.9748, 76/255, 76/255),
            (1.0, 76/255, 76/255),
        ),

        'green': (
            (0.0, 127/255, 127/255),
            (0.1, 255/255, 255/255),
            (0.1597, 255/255, 255/255),
            (0.3, 255/255, 255/255),  # Wider white region (extended green)
            (0.5, 178/255, 178/255),
            (0.563, 204/255, 204/255),
            (0.6975, 76/255, 76/255),
            (0.8319, 102/255, 102/255),
            (0.9748, 25/255, 25/255),
            (1.0, 25/255, 25/255),
        ),

        'blue': (
            (0.0, 255/255, 255/255),
            (0.1, 255/255, 255/255),
            (0.1597, 255/255, 255/255),
            (0.3, 255/255, 255/255),  # Wider white region (extended blue)
            (0.5, 102/255, 102/255),
            (0.563, 76/255, 76/255),
            (0.6975, 76/255, 76/255),
            (0.8319, 153/255, 153/255),
            (0.9748, 76/255, 76/255),
            (1.0, 76/255, 76/255),
        ),
    }

    # Determine color scaling limits
    if np.abs(np.min(Z)) <= np.nanmax(Z):  # Negative extension not required
        vmin = -np.nanmax(Z)
        vmax = np.nanmax(Z)
    else:  # Negative extension required
        vmin = np.nanmin(Z)
        vmax = np.nanmax(Z)

    # Override with fixed limits for FORCinel style
    vmax = 1.0
    vmin = -0.5

    # Define anchor positions for color mapping (normalized 0-1)
    anchors = np.zeros(10)
    anchors[1] = (0.0005 * vmax - vmin) / (vmax - vmin)
    anchors[2] = (0.005 * vmax - vmin) / (vmax - vmin)
    anchors[3] = (0.025 * vmax - vmin) / (vmax - vmin)
    anchors[4] = (0.19 * vmax - vmin) / (vmax - vmin)
    anchors[5] = (0.48 * vmax - vmin) / (vmax - vmin)
    anchors[6] = (0.64 * vmax - vmin) / (vmax - vmin)
    anchors[7] = (0.80 * vmax - vmin) / (vmax - vmin)
    anchors[8] = (0.97 * vmax - vmin) / (vmax - vmin)
    anchors[9] = 1.0

    anchors = np.clip(anchors, 0, 1)  # Ensure anchors are within [0, 1]

    # Apply anchors to the RGB segments
    Rlst = list(cdict['red'])
    Glst = list(cdict['green'])
    Blst = list(cdict['blue'])

    for i in range(9):
        Rlst[i] = tuple((anchors[i], Rlst[i][1], Rlst[i][2]))
        Glst[i] = tuple((anchors[i], Glst[i][1], Glst[i][2]))
        Blst[i] = tuple((anchors[i], Blst[i][1], Blst[i][2]))

    cdict['red'] = tuple(Rlst)
    cdict['green'] = tuple(Glst)
    cdict['blue'] = tuple(Blst)

    # Create the LinearSegmentedColormap from the updated dictionary
    cmap = matplotlib.colors.LinearSegmentedColormap('forc_cmap', cdict)

    return cmap, vmin, vmax
    
def inter_FORC(X, SF):
    """
    Interpolate FORC rho data onto a regular grid for visualization.

    This function takes the FORC data for a given smoothing factor (SF),
    flattens the Hu, Hc, and Rho arrays, removes any NaN values, and then
    interpolates the data onto a uniform 2D grid using cubic interpolation.

    Parameters
    ----------
    X : dict
        Dictionary containing FORC data, including 'Hu_mu', 'Hc_mu', and 'rho'.
    SF : int
        The smoothing factor (index) of the Rho array to interpolate.

    Returns
    -------
    X : dict
        Updated dictionary containing the interpolated grid arrays:
        - 'xi1': x-coordinates of the target grid
        - 'yi1': y-coordinates of the target grid
        - 'zi_{SF}': interpolated Rho values on the target grid for the given SF
    """

    # Flatten the smoothed Hu, Hc, and Rho arrays for interpolation
    Hu_f = X['Hu_mu'].flatten() 
    Hc_f = X['Hc_mu'].flatten()
    Rho_f = X['rho'][SF].flatten() 

    # Combine arrays into a dictionary for simultaneous NaN removal
    data = {"Hu_f": Hu_f, "Hc_f": Hc_f, "Rho_f": Rho_f}

    # Create a mask where none of the arrays contain NaN
    mask = ~np.any([np.isnan(data[key]) for key in data], axis=0)

    # Apply the mask to remove NaN values from all arrays
    cleaned_data = {key: data[key][mask] for key in data}
    Hc_f = cleaned_data['Hc_f']
    Hu_f = cleaned_data['Hu_f']
    Rho_f = cleaned_data['Rho_f']

    # Define grid steps for interpolation
    step_xi = np.nanmax(X['Hc_mu']) / 181.
    step_yi = (np.nanmax(X['Hu_mu']) - np.nanmin(X['Hu_mu'])) / 146.

    # Define the target grid
    xi = np.arange(0, np.nanmax(X['Hc_mu']), step_xi)
    yi = np.arange(np.nanmin(X['Hu_mu']), np.nanmax(X['Hu_mu']), step_yi)
    xi1, yi1 = np.meshgrid(xi, yi)     

    # Interpolate the Rho data onto the regular grid using cubic interpolation
    zi = griddata((Hc_f, Hu_f), Rho_f, (xi1, yi1), method='cubic') 

    # Save the interpolated grids back into the dictionary
    X['xi1'] = xi1
    X['yi1'] = yi1
    X['zi_{}'.format(SF)] = zi
    
    return X

def inter_rho(xi_s_f, yi_s_f, zi_s_f, hys, i):
    """
    Interpolate the Rho value at a specific hysteron location.

    Given a 2D FORC grid (xi_s_f, yi_s_f, zi_s_f) and a hysteron position
    (hys[i,0], hys[i,1]), this function interpolates the rho value at that
    position using the four surrounding grid points. If any surrounding
    points contain NaN, a default value of -0.001 is assigned.

    Parameters
    ----------
    xi_s_f : ndarray
        2D array of x-coordinates (Hc) for the FORC grid.
    yi_s_f : ndarray
        2D array of y-coordinates (Hs) for the FORC grid.
    zi_s_f : ndarray
        2D array of rho values corresponding to the FORC grid.
    hys : ndarray
        Array of hysteron data where hys[i,0] is Hc and hys[i,1] is Hs.
        The interpolated value is stored in hys[i,3].
    i : int
        Index of the hysteron to interpolate.

    Returns
    -------
    hys : ndarray
        Updated array of hysterons with interpolated Rho value at hys[i,3].
    """

    # Extract the first row (Hc values) from the grid
    xi1_row = xi_s_f[0,:] 

    # Find indices of the surrounding Hc values for interpolation
    up_hc = xi1_row[xi1_row > hys[i,0]].min()   
    lo_hc = xi1_row[xi1_row < hys[i,0]].max() 
    up_hc_idx = list(xi1_row).index(up_hc) 
    lo_hc_idx = list(xi1_row).index(lo_hc)
    
    # Extract the first column (Hs values) from the grid
    yi1_col = yi_s_f[:,0] 

    # Find indices of the surrounding Hs values for interpolation
    up_hi = yi1_col[yi1_col > hys[i,1]].min()  
    lo_hi = yi1_col[yi1_col < hys[i,1]].max() 
    up_hi_idx = list(yi1_col).index(up_hi) 
    lo_hi_idx = list(yi1_col).index(lo_hi)

    # Extract the four surrounding grid points for interpolation
    x_arr = np.array([xi_s_f[lo_hi_idx,lo_hc_idx], xi_s_f[up_hi_idx, lo_hc_idx],
                      xi_s_f[up_hi_idx, up_hc_idx], xi_s_f[lo_hi_idx, up_hc_idx]])
    y_arr = np.array([yi_s_f[lo_hi_idx,lo_hc_idx], yi_s_f[up_hi_idx, lo_hc_idx],
                      yi_s_f[up_hi_idx, up_hc_idx], yi_s_f[lo_hi_idx, up_hc_idx]])
    z_arr = np.array([zi_s_f[lo_hi_idx,lo_hc_idx], zi_s_f[up_hi_idx, lo_hc_idx],
                      zi_s_f[up_hi_idx, up_hc_idx], zi_s_f[lo_hi_idx, up_hc_idx]])

    # Check for NaN values in the surrounding points
    xarr_has_nan = np.isnan(np.sum(x_arr))
    yarr_has_nan = np.isnan(np.sum(y_arr))
    zarr_has_nan = np.isnan(np.sum(z_arr))    

    # If no NaNs, perform linear 2D interpolation; otherwise assign default value
    if (xarr_has_nan != True) and (yarr_has_nan != True) and (zarr_has_nan != True):
        f = interp2d(x_arr, y_arr, z_arr, kind='linear')
        hys[i,3] = f(hys[i,0], hys[i,1]) 
    else:
        hys[i,3] = -0.001

    return hys
    
    
def sym_FORC(X, SF):
    """
    Symmetrize a FORC diagram about the Hs=0 axis.

    This function takes a FORC grid (xi1, yi1, zi) and enforces
    symmetry along the Hs axis by averaging points equidistant 
    above and below Hs=0. The symmetrized grid is then cut to
    a relevant region and stored back in X.

    Parameters
    ----------
    X : dict
        Dictionary containing FORC data. Must include keys 'xi1', 'yi1', 
        and 'zi_{SF}' where SF is the smoothing factor.
    SF : int
        Smoothing factor corresponding to the FORC grid to symmetrize.

    Returns
    -------
    X : dict
        Updated dictionary containing symmetrized FORC grid:
        'xis', 'yis', 'zis_{SF}'.
    """

    # Extract grids from X
    xi1 = X['xi1']
    yi1 = X['yi1']
    zi = X['zi_{}'.format(SF)] 

    # Absolute value of Hs to find symmetry axis
    yi_axis = np.copy(yi1)
    yi_axis = abs(yi_axis) - 0  # effectively yi_axis = |yi1|

    # Find index of the Hs value closest to 0
    indices = np.unravel_index(np.nanargmin(yi_axis), yi_axis.shape)

    # Make copies of grids to symmetrize
    xi_s = np.copy(xi1)
    yi_s = np.copy(yi1)
    zi_s = np.copy(zi)
  
    # Symmetrize around the Hs=0 axis
    x = 1
    while x < (len(xi1) - indices[0]): 
        for j in range(len(xi_s[0])):
            # Take the values above and below the axis
            find_mean = np.array([zi_s[indices[0]+x][j], zi_s[indices[0]-x][j]])
            # Remove NaN values
            find_mean = find_mean[~np.isnan(find_mean)]
            if (len(find_mean) > 0):
                # Replace both points with their mean
                zi_s[indices[0]+x][j] = np.mean(find_mean)
                zi_s[indices[0]-x][j] = zi_s[indices[0]+x][j]

        x += 1        

    # Define bounds for cutting the grid to a relevant region
    lower_x_bound = indices[0] - (len(xi_s) - indices[0])
    upper_x_bound = len(xi_s)

    # Cut the grids
    xi_s_cut = xi_s[lower_x_bound:upper_x_bound,:]
    yi_s_cut = yi_s[lower_x_bound:upper_x_bound,:]
    zi_s_cut = zi_s[lower_x_bound:upper_x_bound,:]

    # Store symmetrized and cut grids back in X
    X['xis'] = xi_s_cut
    X['yis'] = yi_s_cut
    X['zis_{}'.format(SF)] = zi_s_cut
    
    return(X)

    
def sym_norm_forcs(X):
    """
    Symmetrize and normalize all FORC diagrams in X.

    This function loops over all smoothing factors (SF) and performs:
    1. Interpolation of the FORC data onto a regular grid.
    2. Symmetrization of the FORC diagram along the Hs axis.
    3. Normalization of the FORC distribution.

    The function updates X with symmetrized and normalized FORC data,
    and computes corrected SF list values based on FWHM if applicable.

    Parameters
    ----------
    X : dict
        Dictionary containing FORC data, smoothing factors, and FWHM information.

    Returns
    -------
    X : dict
        Updated dictionary with symmetrized and normalized FORC data, including:
        - xi1, yi1, zi_# for interpolated FORCs
        - xis, yis, zis_# for symmetrized FORCs
        - normalized FORCs via norm_z2
        - sf_list_correct for FWHM-based correction
    """

    # Only proceed if we are not reusing previous SF data
    if (X['sf_1'] != 'K'):
        SFlist = X['SF_list']

        # Loop over each SF (e.g., 2, 3, 4, 5)
        for i in range(len(SFlist)):
            # Interpolate FORC onto a regular grid
            X = inter_FORC(X, SFlist[i])   
            # Symmetrize the FORC along the Hs axis
            X = sym_FORC(X, SFlist[i])

        # Normalize the symmetrized FORCs
        for i in range(len(SFlist)):
            X = norm_z2(X, SFlist[i])

        # Correct smoothing factor list using FWHM
        fwhmlist = X['fwhmlist']  # correct FWHM values 
        X['sf_list_correct'] = (X['b']/fwhmlist)  # original calculation

    else:
        # If reusing previous SF, only process that single SF
        X = inter_FORC(X, X['SF'])    
        X = sym_FORC(X, X['SF'])   
        X = norm_z2(X, X['SF']) 

    return(X)
   
def norm_z2(X, SF):
    """
    Normalize the symmetrized FORC distribution for a given smoothing factor.

    This function scales the FORC distribution `zis_SF` so that its maximum
    value is 1, keeping the relative structure of the FORC intact.

    Parameters
    ----------
    X : dict
        Dictionary containing symmetrized FORC data in 'zis_SF'.
    SF : int
        The smoothing factor index specifying which FORC to normalize.

    Returns
    -------
    X : dict
        Updated dictionary with normalized symmetrized FORC data in 'zis_SF'.
    """

    # Extract the symmetrized FORC for the given SF
    z_pre_norm = X['zis_{}'.format(SF)]

    # Find the maximum and minimum values in the FORC (minval_z is unused)
    maxval_z = np.nanmax(z_pre_norm)
    minval_z = np.nanmin(z_pre_norm)
    
    # Normalize the FORC by its maximum value
    z_norm = (z_pre_norm) / (maxval_z)

    # Store the normalized FORC back in X
    X['zis_{}'.format(SF)] = z_norm
    return(X)


def user_input(X): 
    """
    Interactively allows the user to review or update the FORC diagram limits.

    This function is intended to be called after the initial FORC plot. It prompts
    the user to either reuse previous maximum hc and minimum hi values or input
    new limits for the sample. It then updates the X dictionary with the chosen
    limits and re-plots the FORC diagram.

    Parameters
    ----------
    X : dict
        Dictionary containing the sample data, including FORC grid and previous limits.

    Returns
    -------
    X : dict
        Updated dictionary with new FORC limits stored in 'hcmaxa', 'himina', 
        'reset_limit_hc', and 'reset_limit_hi'.
    """

    # Check if this is a subsequent analysis for the same sample
    if (X['sample_copy'] > 1): 
        # Prompt user to reuse previous FORC limits or change them
        lim_1 = input(
            'The most recent maximum hc and maximum hi used for sample {} were {} mT and {} mT respectively. '
            'If you want to re-use these variables enter K, to change them enter any other charactor'.format(
                X['names'][0], X['reset_limit_hc'], X['reset_limit_hi'])
        )
    else:
        lim_1 = 'L'

    # If user chooses to change the limits or it's the first analysis
    if (lim_1 != 'K') or (X['sample_copy'] < 2):    

        quest_res = 0.
        while (quest_res == 0):
            # Ask whether user wants to modify hc and hi
            quest = input(
                'Do you want to change the maximum hc and/or hi value from max hc = {} and min hi = {} from the FORC file? (Y or N)'.format(
                    int(X['hcmaxa']), int(X['himina'])
                )
            )

            if (quest == 'Y'):
                quest_res = 1.

                # Prompt for maximum hc value
                hc_l = 0
                while (hc_l == 0):                
                    try:
                        max_input_hc = input("Set maximum hc (mT) using the above FORC diagram, if unchanged enter 0:" )
                        max_input_hc = float(max_input_hc)
                        if (max_input_hc == 0):
                            max_input_hc = X['hcmaxa']                            
                        hc_l = 1
                    except ValueError:
                        print('Not a number. Please input a number.')
                        hc_l = 0

                # Prompt for minimum hi value
                hi_l = 0
                while (hi_l == 0):
                    try:
                        max_input_hi = input("Set minimum hs (mT) using the above FORC diagram, if unchanged enter 0:" )
                        max_input_hi = float(max_input_hi)
                        if (max_input_hi == 0):
                            max_input_hi = X['himina']              
                        hi_l = 1
                    except ValueError:
                        print('Not a number. Please input a number.')
                        hi_l = 0

                # Update the dictionary with new limits
                X['hcmaxa'] = float(max_input_hc)
                X['himina'] = float(max_input_hi)
                print('FORC limits: Max hc = {}, max hi = {}'.format(X['hcmaxa'], X['himina']))          

                # Re-plot the FORC diagram with the new limits
                plot_sample_FORC(X['Hc'], X['Hu'], X['rho_n'], X['SF'], X['name'],X['hcmaxa'],X['himaxa'],X['himina'])
                        
            if (quest == 'N'):
                quest_res = 1.
                max_input_hc = X['hcmaxa']    
                max_input_hi = X['himina']   

    # Store the chosen or default limits for future reference
    X['reset_limit_hc'] = float(max_input_hc)
    X['reset_limit_hi'] = float(max_input_hi)

    return X


# ==== demagnetising data functions =====


def demag_data_generic_arm(X):
    """
    Reads and processes AF (alternating field) demagnetization data for NRM, ARM, and SIRM.

    This function reads the demagnetization data files provided in X['nrm_files'], 
    X['arm_files'], and X['sirm_files']. It extracts the total, decayed (dec), and 
    incremental (inc) values for each AF step, converts them to NumPy arrays, and 
    stores them in the X dictionary. It also checks that the AF steps are consistent 
    across all measurements.

    Parameters
    ----------
    X : dict
        Dictionary containing paths to NRM, ARM, and SIRM files:
        - X['nrm_files']: list of NRM file paths
        - X['arm_files']: list of ARM file paths
        - X['sirm_files']: list of SIRM file paths

    Returns
    -------
    X : dict
        Updated dictionary with demagnetization data stored as NumPy arrays, including:
        - af_step, af_nrm, af_arm, af_irm
        - af_nrm_dec, af_arm_dec, af_irm_dec
        - af_nrm_inc, af_arm_inc, af_irm_inc
        - cntfield: number of AF steps
    """

    # Initialize lists to store data for each measurement type
    af_step = []

    af_nrm_n = []
    af_arm_n = []
    af_sirm_n = []
    af_step_sirm = []
    af_step_arm = []
    af_nrm_dec = []
    af_nrm_inc = []
    af_arm_dec = []
    af_arm_inc = []
    af_sirm_dec = []
    af_sirm_inc = []

    # Read NRM file
    af_nrm_data = open(X['nrm_files'][0], "r") 
    for myline1 in af_nrm_data:
        line = myline1.split()  # split line into columns by whitespace
        af_step.append(float(line[1])) 
        af_nrm_n.append(float(line[3]))
        af_nrm_dec.append(float(line[4]))
        af_nrm_inc.append(float(line[5]))
    af_nrm_data.close()

    # Read ARM file
    af_arm_data = open(X['arm_files'][0], "r") 
    for myline1 in af_arm_data:
        line = myline1.split()
        af_step_arm.append(float(line[1])) 
        af_arm_n.append(float(line[3]))
        af_arm_dec.append(float(line[4]))
        af_arm_inc.append(float(line[5]))
    af_arm_data.close()
    
    # Read SIRM file
    af_irm_data = open(X['sirm_files'][0], "r") 
    for myline in af_irm_data:
        line = myline.split()
        af_step_sirm.append(float(line[1])) 
        af_sirm_n.append(float(line[3]))
        af_sirm_dec.append(float(line[4]))
        af_sirm_inc.append(float(line[5]))
    af_irm_data.close()

    # Check if AF steps are consistent across all datasets
    if (af_step == af_step_sirm) and (af_step == af_step_arm):
        pass
    else:
        print('AF demagnetisation steps are different for NRM and SIRM. Re-import data.')

    # Total number of AF steps
    cntfield = len(af_step)

    # Convert lists to NumPy arrays for easier numerical operations
    af_step = np.array(af_step)
    af_nrm_n = np.array(af_nrm_n)
    af_arm_n = np.array(af_arm_n)
    af_sirm_n = np.array(af_sirm_n)
    af_nrm_dec = np.array(af_nrm_dec)
    af_arm_dec = np.array(af_arm_dec)
    af_sirm_dec = np.array(af_sirm_dec)
    af_nrm_inc = np.array(af_nrm_inc)
    af_arm_inc = np.array(af_arm_inc)
    af_sirm_inc = np.array(af_sirm_inc)

    # Adjust SIRM length to match NRM if necessary
    af_sirm_n = af_sirm_n[:len(af_nrm_n)]

    # Optional normalization factor (currently just stored)
    afnorm = af_nrm_n[0] / af_sirm_n[0]

    # Store processed data in the X dictionary
    X['af_step'] = af_step
    X['af_nrm'] = af_nrm_n
    X['af_arm'] = af_arm_n
    X['af_irm'] = af_sirm_n
    X['af_nrm_dec'] = af_nrm_dec
    X['af_arm_dec'] = af_arm_dec
    X['af_irm_dec'] = af_sirm_dec
    X['af_nrm_inc'] = af_nrm_inc
    X['af_arm_inc'] = af_arm_inc
    X['af_irm_inc'] = af_sirm_inc
    X['cntfield'] = cntfield

    return X


def hys_angles():
    """
    Generates random angles for hysteron orientation in spherical coordinates.

    Returns
    -------
    phi : float
        Random polar angle (0 to pi/2) for the dynamic hysteron.
    phistatic : float
        Random polar angle (0 to pi/2) for the static hysteron.
    thetastatic : float
        Random azimuthal angle (0 to 2*pi) for the static hysteron.
    """
    
    # Generate a random number between 0 and 1 for dynamic hysteron
    angle = random.random()
    # Convert uniform random number to polar angle (0 to pi)
    phi = acos(2*angle - 1)
    # Ensure angle is within 0 to pi/2
    if(phi > (pi/2.)): 
        phi = pi - phi

    # Generate a random number for static hysteron
    angle2 = random.random()
    # Convert to polar angle
    phistatic = acos(2*angle2 - 1)
    # Ensure angle is within 0 to pi/2
    if(phistatic > (pi/2.)):
        phistatic = pi - phistatic
    
    # Generate azimuthal angle for static hysteron (0 to 2*pi)
    angle3 = random.random()
    thetastatic = 2 * pi * angle3

    # Return the three angles
    return phi, phistatic, thetastatic


def calc_hk_arrays(hys, num, V): 
    """
    Calculate the anisotropy field (Hk) for a set of hysterons based on 
    coercivity, orientation, and thermal activation.

    Parameters
    ----------
    hys : np.ndarray
        Array of hysteron properties. Column 0 is coercivity (Hc), column 5 is phi.
    num : int or float
        Number of hysterons to process.
    V : dict
        Dictionary containing simulation parameters, must include 'tm' (measurement time).

    Returns
    -------
    hys : np.ndarray
        Input array with updated Hk values stored in column 9.
    """

    # Measurement time from input dictionary
    tm = V['tm']

    # Coercivity of hysterons
    hc = np.copy(hys[:,0]) 
    tempt = 300.  # Temperature in Kelvin (used in formula)

    # Initial attempt at HF, scaling coercivity with empirical exponent
    hf = ((hc/(sqrt(2)))**(0.54))*(10**(-0.52)) 
    # hf = ((hc)**(0.54))*(10**(-0.52))   # alternative commented version

    # Polar angle of each hysteron
    phi = np.copy(hys[:,5])
    
    # Orientation factor based on polar angle
    phitemp = (((np.sin(phi))**(2./3.)) + ((np.cos(phi))**(2./3.)))**(-3./2.) 

    # Empirical phi-dependent scaling factor
    gphitemp = (0.86 + (1.14 * phitemp)) 
  
    # Adjusted coercivity for calculation
    hatmp = hc / (sqrt(2)) 

    # Thermal activation term
    ht = hf * (log(tm / tau)) 
    
    # Initial Hk estimate
    hktmp = hatmp + ht + (2*hatmp*ht + ht**2)**0.5 
    hktmpmax = hktmp
    
    # Store initial Hk values
    hktmpstore = hktmp
    i = 0

    # Iterate over hysterons to refine Hk
    for i in range(int(num)):
        factor = 0.5
        searchme = 1000000.0     
        hstep = (hktmp[i] - hatmp[i]) / 5.  # Initial step size

        # Refine Hk until convergence (difference < 5)
        while (abs(searchme) > 5):
            searchme = hktmp[i] - hatmp[i] - hktmp[i] * ((2*ht[i]*phitemp[i]/hktmp[i])**(1/gphitemp[i]))
            hktmpstore[i] = hktmp[i]

            # Adjust Hk based on sign of difference
            if (searchme > 0):
                hktmp[i] = hktmp[i] - hstep
            else:
                hktmp[i] = hktmp[i] + hstep
                hstep = hstep * factor  # Reduce step size for finer convergence

    # Final Hk corrected for orientation
    hkphi = hktmpstore 
    hk = hkphi[:int(num)] / phitemp[:int(num)]

    # Store computed Hk in column 9 of hysteron array
    hys[:int(num), 9] = hk 

    return hys

    
def pop_hys(num_hys, X, V, SF): 
    """
    Populate an array of hysterons with random Hc, Hi, orientation angles, 
    and calculate their Hk values based on input FORC data and correction factors.

    Parameters
    ----------
    num_hys : int
        Total number of hysterons to generate.
    X : dict
        Dictionary containing FORC and sample data, including:
        - 'xis', 'yis', 'zis_<SF>': symmetrized FORC arrays
        - 'sf_list_correct': correction factors for FWHM
        - 'reset_limit_hc', 'reset_limit_hi': user-defined limits
        - 'Hc2', 'Hb2': maximum Hc and Hi values
    V : dict
        Dictionary of simulation parameters, required for Hk calculation.
    SF : int
        Smoothing factor index.

    Returns
    -------
    hys : np.ndarray
        Array of hysterons with populated properties and Hk values.
    num_pop : int
        Number of hysterons generated for half the population (mirrored to create full set).
    """

    # Correction factor for this SF
    corr = X['sf_list_correct'][SF - 2]

    # Initialize hysteron array: 11 columns for properties
    hys = np.zeros((num_hys, 11)) 
    
    # Number of hysterons for half the distribution (mirrored later)
    num_pop = num_hys / 2
    
    # Symmetrized FORC data for interpolation
    xi_s_cut = X['xis']
    yi_s_cut = X['yis']
    zi_s_cut = X['zis_{}'.format(SF)] 

    # Limits for Hc and Hi based on user input and FORC maximums
    xlim_res = X['reset_limit_hc'] / 1000.
    ylim_res = X['reset_limit_hi'] / 1000.
    maxHc = min(xlim_res, X['Hc2'])
    maxHi = min(ylim_res, X['Hb2'])

    # Max/min Hc and Hi for random generation (FORC-based)
    maxHc = np.nanmax(xi_s_cut)
    maxHi = min(abs(np.nanmax(yi_s_cut)), abs(np.nanmin(yi_s_cut)))
    minHc = np.nanmin(xi_s_cut) 
    minHi = np.nanmin(yi_s_cut)
    
    i = 0
    # Populate half of the hysterons randomly
    while (i < int(num_pop)):
        z1 = random.random()
        z2 = random.random()
        z3 = random.random()
  
        # Random Hc and Hi within the range
        hys[i,0] = (z2 * maxHc) 
        hys[i,1] = (z3 * maxHi) 

        # Interpolate rho value from FORC data
        hys = inter_rho(xi_s_cut, yi_s_cut, zi_s_cut, hys, i) 

        # Apply correction factor to Hi
        hys[i,1] = hys[i,1] * corr

        # Assign random orientation angles for this hysteron
        hys[i,5], hys[i,6], hys[i,7] = hys_angles() 
        
        # Ensure physically valid hysteron: Hi <= Hc and 0 <= rho <= 1
        if ((hys[i,1]) <= (hys[i,0])) and (hys[i,3] >= 0) and (hys[i,3] <= 1): 
            i += 1 

    # Calculate Hk for the populated hysterons
    hys = calc_hk_arrays(hys, int(num_pop), V) 

    # Set magnetization scale to 1
    hys[:,4] = 1

    # Auxiliary property: scaled polar angle
    hys[:,8] = hys[:,5] * hys[:,4] 

    num_pop = int(num_pop)

    # Mirror half-population to generate full hysteron set
    j = 0
    for j in range(num_pop):
        hys[(j + num_pop), :] = hys[j, :]
        hys[j + num_pop, 1] = -hys[j, 1]  # flip Hi for mirrored half
        j += 1

    return hys, num_pop

@jit(nopython = True)
def block_loc(var_1, hys, blockg, boltz, armstore):
    """
    Compute blocking states for each hysteron at a given temperature and field.
    """

    # ---- Unpack variables ----
    num_hyss = int(var_1[0])
    beta = float(var_1[1])
    blockper = float(var_1[2])
    temp = float(var_1[3])
    aconst = float(var_1[4])
    curie_t = float(var_1[5])
    rate = float(var_1[6])
    tempmin = float(var_1[7])
    field = float(var_1[8])
    tm = float(var_1[9])

    # ---- Constants ----
    tau = 10e-9
    roottwohffield = np.sqrt(2)

    # ---- Initialise arrays ----
    hfstore = np.zeros(num_hyss)
    histore = np.zeros(num_hyss)
    hcstore = np.zeros(num_hyss)
    blocktemp = np.zeros(num_hyss)

    # ---- Main loop ----
    for i in range(num_hyss):
        phitemp = ((np.sin(hys[i, 5]) ** (2. / 3.)) + (np.cos(hys[i, 5]) ** (2. / 3.))) ** (-3. / 2.)
        hc = np.sqrt(2) * hys[i, 9] * beta
        hcstore[i] = hc / np.sqrt(2)

        hi = hys[i, 1] * beta * blockper
        histore[i] = hi / np.sqrt(2)

        g = 0.86 + 1.14 * phitemp
        hf = (hys[i, 9] ** 0.54) * (10 ** -0.52) * temp / (300 * beta)
        hfstore[i] = hf

        # ---- Relaxation time calculation ----
        if rate == 1:
            r = (1. / aconst) * (temp - tempmin)
            tm = (temp / r) * (1 - (temp / (curie_t + 273))) / np.log(
                (2 * temp) / (tau * r) * ((1. - (temp / (curie_t + 273))))
            )
            if tm == 0.0:
                tm = 60.0

        ht = roottwohffield * hf * np.log(tm / tau)
        bracket = 1 - (2 * ht * phitemp / hc) ** (1 / g)
        hiflipplus = hc * bracket + field * roottwohffield
        hiflipminus = -hc * bracket + field * roottwohffield

        # ---- Determine blocking state ----
        if hc >= (2 * ht * phitemp):
            if (hi > hiflipminus) and (hi < hiflipplus):
                if (blockg[i] == 0) or (blockg[i] in (2, -2)):
                    if hi >= (field * roottwohffield):
                        blocktemp[i] = -1
                    else:
                        blocktemp[i] = 1

                elif blockg[i] == -1:
                    blocktemp[i] = -1

                elif blockg[i] == 1:
                    if (armstore == 1) and (hi > field):
                        #print('armstore',armstore,hi,field)
                        blocktemp[i] = -1
                        boltz[i] = -1
                    else:
                        blocktemp[i] = 1
                else:
                    print(blockg[i], blocktemp[i])
                    print('----', i)

            elif hi >= hiflipplus:
                blocktemp[i] = -2
            else:
                blocktemp[i] = 2

        else:
            if (hi < hiflipminus) and (hi > hiflipplus):
                blocktemp[i] = 0
                boltz[i] = 0.0
            else:
                if hi >= hiflipminus:
                    blocktemp[i] = -2
                else:
                    blocktemp[i] = 2

    return hfstore, histore, boltz, blocktemp


@jit(nopython=True)
def block_val(hys, histore, hfstore, blocktemp, beta, num_hyss, boltz, blockg, field, afswitch, afstore):
    """
    Compute the blocked magnetization and blocking fraction for a set of hysterons.

    Parameters
    ----------
    hys : np.ndarray
        Array of hysteron properties.
    histore : np.ndarray
        Array of hysteron coercivities along the applied field.
    hfstore : np.ndarray
        Array of hyperfine or local field parameters for hysterons.
    blocktemp : np.ndarray
        Current blocking state of each hysteron.
    beta : float
        Scaling factor for magnetization.
    num_hyss : int
        Number of hysterons.
    boltz : np.ndarray
        Array of Boltzmann factors for thermal activation.
    blockg : np.ndarray
        Array storing updated blocking states.
    field : float
        Current applied magnetic field.
    afswitch : int
        Flag to include AF (alternating field) effects: 1 = yes, 0 = no.
    afstore : float
        Alternating field amplitude for AF demagnetization.

    Returns
    -------
    blockper : float
        Fraction of hysterons contributing to blocked magnetization.
    totalm : float
        Total blocked magnetization.
    boltz : np.ndarray
        Updated Boltzmann factors after computation.
    blockg : np.ndarray
        Updated blocking states after computation.
    """

    i = 0
    totalm = 0.0
    blockper = 0   

    for i in range(num_hyss):
        x = blocktemp[i]
        blockg[i] = x  # update blocking state
        absblockg = abs(blockg[i])
        
        # If hysteron is fully blocked
        if (absblockg == 1): 
            # Only update if Boltzmann factor is essentially zero
            if (boltz[i] < 1e-8) and (boltz[i] > -1e-8):
                boltz[i] = tanh((field - histore[i]) / hfstore[i])
                # Reset Boltzmann factor if AF switch is on
                if (afswitch == 1):
                    boltz[i] = 0.

        # Assign moment based on block state
        if (blockg[i] == -2):
            moment = -0  # effectively zero
        elif (blockg[i] == 2):
            moment = 0
        else:
            moment = blockg[i] 

        # Apply AF demagnetization effects if switch is on
        if (afswitch == 1): 
            hi = histore[i] * sqrt(2)
            hc = sqrt(2) * (hys[i,9] * beta) * (((sin(hys[i,6])**(2./3.)) + (cos(hys[i,6])**(2./3.)))**(-3./2.))
    
            af = afstore / (1000 * mu0)  # convert AF amplitude to proper units
       
            if (hi >= 0) and (hi > (hc - af)):
                moment = 0
                boltz[i] = 0
                blockg[i] = -2
            
            if (hi <= 0) and (hi < (-hc + af)):
                moment = 0
                blockg[i] = 2
                boltz[i] = 0

        # Update total magnetization contribution
        totalm += abs(moment) * abs(cos(hys[i,5])) * hys[i,3] * beta * boltz[i]

        # Update blocked fraction (normalized later)
        blockper += abs(moment) * 1.0 * hys[i,3] 
        i += 1

    # Normalize blocked fraction by total hysteron weight
    blockper /= (1.0 * np.sum(hys[:,3]))

    return blockper, totalm, boltz, blockg


def blockfind(temp, field, afswitch, V, X):
    """
    Update the blocking states and magnetization for a given temperature and applied field.

    This function is typically called within TRM/ARM acquisition or AF demagnetization loops.
    It computes local effective fields, updates Boltzmann factors, and calculates the
    blocked fraction and total magnetization using the block_val function.

    Parameters
    ----------
    temp : float
        Current temperature during simulation/measurement.
    field : float
        Applied magnetic field.
    afswitch : int
        Flag to include AF (alternating field) effects: 1 = yes, 0 = no.
    V : dict
        Dictionary containing hysteron properties and current simulation state.
    X : dict
        Dictionary containing material/sample parameters (e.g., curie temperature).

    Returns
    -------
    V : dict
        Updated simulation state with new blocking fractions, Boltzmann factors, and total magnetization.
    """

    # Extract hysteron and simulation parameters
    hys = V['hys']
    num_hyss = V['num_hyss']
    hcstore = V['hcstore']
    histore = V['histore']
    beta = V['beta']
    rate = V['rate']
    aconst = V['aconst']
    tempmin = V['tempmin']
    tm = V['tm']

    # Initialize local field array
    hfstore = np.zeros(num_hyss)
    # Current simulation state
    blockper = V['blockper']
    blocktemp = V['blocktemp']
    boltz = V['boltz']
    blockg = V['blockg']
    curie_t = X['curie_t']

    totalm = V['totalm']
    afstore = V['afstore']
    armstore = V['armstore']

    # Aggregate relevant variables into an array for block_loc computation
    var_1 = np.array((num_hyss, beta, blockper, temp, aconst, curie_t, rate, tempmin, field, tm))

    # Compute local effective fields and update Boltzmann factors
    hfstore, histore, boltz, blocktemp = block_loc(var_1, hys, blockg, boltz, armstore)  
    
    # Update blocked fraction, total magnetization, and Boltzmann factors
    blockper, totalm, boltz, blockg = block_val(hys, histore, hfstore, blocktemp, beta, num_hyss, boltz, blockg, field, afswitch, afstore)
    
    # Store updated state back into simulation dictionary
    V['blockper'] = blockper
    V['blocktemp'] = blocktemp
    V['boltz'] = boltz
    V['blockg'] = blockg
    V['totalm'] = totalm
    V['tm'] = tm

    return V


# ====== Calculate paleomagnetism =======
    
def calc_pal(X, V):
    """
    Calculate palaeointensity (PAL) and SIRM-related metrics from AF demagnetization data.

    This function performs:
    - Normalization of AF NRM to SIRM.
    - Weighted and unweighted averaging of AF measurements.
    - Detection of flat sections in AF demagnetization curves.
    - Calculation of variances, dispersion, and standard deviations.
    - Writing results to output files for analysis and plotting.

    Parameters
    ----------
    X : dict
        Dictionary containing sample data, AF steps, NRM, SIRM, and AF ratios.
    V : dict
        Dictionary containing AF measurement fields, applied fields, and simulation state.

    Returns
    -------
    X : dict
        Updated sample dictionary (mostly unchanged in this function).
    V : dict
        Updated state dictionary with additional diagnostic metrics: shortdivlist, shortminuslist, ys.
    """

    # Extract relevant variables from input dictionaries
    cntfield = X['cntfield']
    ifield = V['ifield']
    afmag = V['afmag']
    sirm = V['sirm']  # SIRM measurements
    af_nrm_n = X['af_nrm']
    af_sirm_n = X['af_irm']
    name = X['name']
    aconst = V['aconst']
    fields = V['fields']
    af_step = X['af_step']
    tempmin = V['tempmin']
    tempmax = V['tempmax']

    # Initialize variables for averaging, variance, flat detection, and other diagnostics
    afnegsirm = 0.0
    averageme = 0.0
    averageunweight = 0.0
    averagemeadj = 0.0
    averagecount = 0
    flatcount = np.zeros((10))
    noflatsub = 0
    averageflat = np.zeros((10))
    trmsirmratio = 0.0
    sigmtot = 0.0
    selrms = 0.0
    sumegli = 0

    afpick = X['af_pick']  # index to start selecting AF data

    # Temporary storage arrays for calculations
    sit = ifield
    std = np.zeros((100))
    ys = np.zeros((100))
    used = np.zeros((50))
    flatused = np.zeros((50))
    ystore = np.zeros((100))
    sigmstore = np.zeros((100))
    flatstore = np.zeros((10,100))
    flatdi = np.zeros((10))
    sigmflat = np.zeros((10))
    flatmedianme = np.zeros((10))
    sigmtotflat = np.zeros((10))
    cntfieldaf = cntfield
    actfield = 1  # scaling field, placeholder
    dispersion = 0
    vtwo = 0
    fortyone = 0

    # Lists for tracking short-term deviations
    shortdivlist = []
    shortminuslist = []
    ilist = []
    af_step_list = []

    # Initialize flat section and AF ratio counters
    flat = 0
    afratio = 0

    # Loop over each AF demagnetization step
    for i in range(cntfieldaf):

        # Prepare lists and sums for linear regression
        xlist = []
        ylist = []
        sumx = 0.
        sumy = 0.
        sumxy = 0.
        sumxx = 0.

        # Loop over fields for current step
        for j in range(sit):
            y = fields[j]*mu0*1e6  # scale field
            x = afmag[j,i]/sirm[j,i]  # normalize AF magnetization to SIRM
            xlist.append(x)
            ylist.append(y)

            if sirm[j,i] == 0:  # handle zero SIRM
                for i in range(afpick, averagecount):
                    dispersion += (((ystore[i] - averageme/sigmtot))**2)/sigmstore[i]
                    vtwo += (1/sigmstore[i])**2
                    fortyone = 1

            sumx += x
            sumxx += x**2
            sumxy += x*y
            sumy += y

        # Linear regression to determine slope (mfit) and intercept (cfit)
        mfit = (((sit+000.0)*sumxy) - sumx*sumy)/(((sit+000.0)*sumxx) - sumx*sumx)
        cfit = (sumy*sumxx - sumx*sumxy)/((sit+000.0)*sumxx - sumx*sumx)

        # Prepare test points and residuals
        xtest = np.linspace(min(xlist), max(xlist), 10)
        ytest = mfit*xtest + cfit
        sumdi = 0.
        xlist2 = []
        ylist2 = []
        dilist = []

        for j in range(sit):
            y = fields[j]*mu0*1e6
            x = afmag[j,i]/sirm[j,i]
            di = y - (mfit*x + cfit)
            dilist.append(di)
            sumdi += di**2
            xlist.append(x)
            ylist.append(y)

        # Calculate variance for weighted averaging
        sumdi /= (sit-2)
        sigm = sit*sumdi/(sit*sumxx - sumx*sumx)

        # Compute AF ratio and related metrics
        x = af_nrm_n[i]/af_sirm_n[i]
        y = x*mfit + cfit
        xlist2.append(x)
        ylist2.append(y)

        ys[i] = y
        std[i] = sqrt(sigm)*x
        used[i] = 0
        flatused[i] = 0

        # Calculate short-term deviations and store
        shortdiv = abs(1 - ((sirm[j-1,i]/np.mean(sirm[j-1,0:3]))/(af_sirm_n[i]/np.mean(af_sirm_n[0:3]))))*100
        shortminus = abs((sirm[j-1,i]/np.mean(sirm[j-1,0:3]) - af_sirm_n[i]/np.mean(af_sirm_n[0:3])))*100
        shortdivm = abs(1 - ((sirm[j-2,i]/np.mean(sirm[j-2,0:3]))/(af_sirm_n[i-1]/np.mean(af_sirm_n[0:3]))))*100
        shortminusm = abs((sirm[j-2,i]/np.mean(sirm[j-2,0:3]) - af_sirm_n[i-1]/np.mean(af_sirm_n[0:3])))*100
        shortdivlist.append(shortdiv)
        shortminuslist.append(shortminus)
        af_step_list.append(af_step[i])

        if (i >= afpick) and (sumx != 0.0): 
            # Only consider points at or after the user-picked AF step (afpick)
            # and ensure sumx (denominator used earlier) is not zero to avoid divide-by-zero issues.

            sigm = sigm*(x**2)
            # Adjust the current variance-like quantity 'sigm' by multiplying by x^2.
            # 'sigm' is being built up from residual/variance contributions; this line rescales it.

            afratio = afmag[2,i]/afmag[2,0]
            # Compute the ratio of AF-demagnetised magnitude at step i to the initial (index 0).
            # Used to check that the AF-demagnetisation series is meaningful at this step.

            if (i >= 1): #greater than 1 as 0 counts here
                afratiom = (afmag[2, i-1]/afmag[2,0]) #also do ratio for prev point - poss compares afratio and afratiom
                # Compute the AF ratio for the previous index for potential comparison with current.

            # Compute SIRM diagnostic metrics comparing simulated and measured SIRM behaviour
            shortdiv = abs(1 - ((sirm[j-1,i] / np.mean(sirm[j-1,0:3])) / (af_sirm_n[i] / np.mean(af_sirm_n[0:3])))) * 100
            shortminus = abs(((sirm[j-1,i] / np.mean(sirm[j-1,0:3])) - (af_sirm_n[i] / np.mean(af_sirm_n[0:3])))) * 100
            # Also compute the same diagnostics for the previous AF step (used to check continuity)
            shortdivm = abs(1 - ((sirm[j-2,i] / np.mean(sirm[j-2,0:3])) / (af_sirm_n[i-1] / np.mean(af_sirm_n[0:3])))) * 100
            shortminusm = abs(((sirm[j-2,i] / np.mean(sirm[j-2,0:3])) - (af_sirm_n[i-1] / np.mean(af_sirm_n[0:3])))) * 100

            sigm = abs(1 - sigm)
            # Convert the previously scaled 'sigm' into a bounded measure (distance from 1).
            # This becomes a variance-like weight used later (1/sigm).

            # Record indices and (potentially) the AF step for diagnostic lists
            ilist.append(i)

            # Track short-range diagnostics only for a sensible AF-range (arbitrary cap at 30)
            if (i > 0) and (i < 30):

                # Require that the normalised NRM at this AF step and AF ratio are above thresholds
                if (af_nrm_n[i] / af_nrm_n[0] > 0.01) and (afratio > 0.01):

                    if (y > 0.0):
                        # Only proceed if the fitted/measured 'y' (field estimate) is positive

                        if (shortdiv < 100):
                            # disallow wildly different S_ratio values; 100% is a very permissive cut
                            if (shortminus < 20):
                                # require small absolute difference (S_diff) as further quality control

                                # accumulate percent difference of TRM/SIRM between measured and simulated
                                selrms = selrms + abs((af_sirm_n[i] / af_sirm_n[0]) - (sirm[j-1,i] / sirm[j-1,0]))

                                # accumulate inverse-variance weight (sigm should be >0 here)
                                sigmtot = sigmtot + (1 / sigm)

                                # sum the ratio of field used to the activation field (used in weighted average)
                                sumegli = sumegli + (y / actfield)

                                # accumulate weighted sum of predicted fields (weight = 1/sigm)
                                averageme = averageme + (y) / sigm

                                # unweighted sum for reference
                                averageunweight = averageunweight + y

                                # count accepted points
                                averagecount = averagecount + 1

                                # store the raw y value for later median/dispersion calculations
                                ystore[averagecount] = y

                                # store the per-point sigma-like weight for later dispersion calculations
                                sigmstore[averagecount] = sigm

                                # mark this AF index as used in selection
                                used[i] = 1

                                # accumulate true TRM/SIRM ratio for diagnostics
                                trmsirmratio = trmsirmratio + x

                                if (i > 1):
                                    # Check flatness relative to previous point
                                    flatdiff = abs(y - ys[i-1]) / max(y, ys[i-1])

                                    # Accept as part of a "flat" region if change is small, or both small (<3)
                                    if (flatdiff < 0.2) or (y < 3.0 and ys[i-1] < 3.0):

                                        if (noflatsub == 0):
                                            # Start a new flat region if not already in a sub-flat
                                            flat = flat + 1

                                            # Validate previous point before adding it to flat group:
                                            if (i-1 < afpick) or (shortdivm > 100) or (shortminusm > 20) or (ys[i-1] < 0.0) or (af_nrm_n[i-1] / af_nrm_n[0] < 0.01) or (afratiom < 0.01):
                                                # Reject previous point if it fails any of these checks
                                                pass
                                            else:
                                                # Add previous point to the current flat group
                                                flatcount[flat] = flatcount[flat] + 1
                                                flatstore[flat][int(flatcount[flat])] = ys[i-1]
                                                averageflat[flat] = averageflat[flat] + ys[i-1]
                                                sigmflat[flat] = sigmflat[flat] + (1 / sigm)
                                                flatused[i-1] = flat
                                                flatdi[flat] = flatdi[flat] + (y - ys[i-1]) / max(y, ys[i-1])

                                        # Mark that a sub-flat has been started/continued
                                        noflatsub = 1

                                        # Add the current point to the same flat group
                                        flatcount[flat] = flatcount[flat] + 1
                                        flatstore[flat][int(flatcount[flat])] = y
                                        averageflat[flat] = averageflat[flat] + y
                                        sigmflat[flat] = sigmflat[flat] + (1 / sigm)
                                        flatdi[flat] = flatdi[flat] + (y - ys[i-1]) / max(y, ys[i-1])
                                        flatused[i] = flat
                                    else:
                                        # current point does not continue a flat region
                                        noflatsub = 0
                                else:
                                    # no previous point to compare to -> cannot form flatsub
                                    noflatsub = 0

                            else:
                                # shortminus too large: reject as flat candidate
                                noflatsub = 0

                        else:
                            # shortdiv too large: reject as flat candidate
                            noflatsub = 0

                    else:
                        # Non-positive y: not usable
                        noflatsub = 0

                else:
                    # AF NRM or AF ratio too small: reject
                    noflatsub = 0

            else:
                # index outside considered AF range: reject
                noflatsub = 0

        else: 
            # i < afpick or sumx == 0 (the outer if failed): reset noflatsub
            noflatsub = 0

    # if any points accepted, compute a percent-based RMS measure
    if (averagecount != 0):
        selrms = 100 * (selrms) / (1.0 * averagecount)

    # prepare temporary array to compute median of accepted y values
    temptwo = np.empty((100))
    temptwo[:] = np.nan

    for i in range(averagecount):
        temptwo[i] = ystore[i]

    # remove NaNs from stored values to compute median correctly
    temptwo = temptwo[~np.isnan(temptwo)]
    medianme = np.median(temptwo)

    # process each flat region to calculate a "flat median"
    for jf in range(flat):

        flatc = 0

        for i in range(int(flatcount[jf])): # iterate points in this flat group

            if (i >= 1): # if there are previous entries to compare with
                # simple monotonicity checks to set a flatc counter
                if (flatstore[jf,i] > 1.0 * flatstore[jf,i-1]):
                    flatc = flatc + 1

                if (flatstore[jf,i] < 1.0 * flatstore[jf,i-1]):
                    flatc = flatc - 1

        # if all comparisons point in the same monotonic direction and enough points exist
        if (abs(flatc) == (flatcount[jf] - 1)) and (flatcount[jf] > 3):

            if (abs(flatstore[jf,0] - flatstore[jf,flatcount[jf]]) > 10):
                # if the flat group spans a large range it's invalidated
                flatcount[jf] = 0
                break
        else:
            # flat group does not meet monotonic test; continue without special action
            pass

        if (flatcount[jf] <= 1):
            # not enough points to form a flat group -> discard and stop checking further groups
            flatcount[jf] = 0
            break

        # compute flat median for this group (placeholder logic - uses array indexing pattern from original)
        flatmedianme[jf] = np.median(temptwo[:, jf])

    # initialize dispersion sums if not in special '41' mode
    if (fortyone == 0):
        dispersion = 0.0
        vtwo = 0.0

    # compute dispersion and vtwo using accepted points from afpick to averagecount
    for i in range(afpick, averagecount):
        dispersion = dispersion + (((ystore[i] - averageme / sigmtot))**2) / sigmstore[i]
        vtwo = vtwo + (1 / sigmstore[i])**2

    # For each flat group, compute a standard-deviation-like metric
    for i in range(flat):
        sigmtotflat[i] = 0
        for k in range(int(flatcount[flat])):
            sigmtotflat[i] = sigmtotflat[i] + ((flatstore[i,k] - (averageflat[i] / flatcount[i]))**2) / flatcount[i]

        sigmtotflat[i] = sqrt(sigmtotflat[i])
        # end of flat normal

    # Compute RMS and other statistics, write results to output files
    sirmrms = 0
    print("All demag and palaeointensity data is saved in afdemag-all-{0}.dat".format(name))
    print("All demag and SIRM data is saved in afdemag-sirm-{0}.dat".format(name))
    fall = open("afdemag-all-{0}.dat".format(name), "w")
    fsirm = open("afdemag-sirm-{0}.dat".format(name), "w")

    # Write headers
    fall.write('afield\tstep\tstepPI\tstd\tselect\tflatno\tshortminus\tSIRMAFS%\tSIRMAFM%\tAFNRM/SIRM-M%\tAFNRM/SIRM-S%\tAFNRM/NRM-M%\tAFNRM/NRM-S%shortdiv%')
    fsirm.write('afield\tmeasured\tsimulated')

    # Populate output files
    for i in range(cntfieldaf):
        sirmrms += abs((af_sirm_n[i]/af_sirm_n[0]) - sirm[j-1, i]/sirm[j-1,0])
        fall.write('\n')
        fall.write(str(af_step[i]) + '\t' + str(i) + '\t' + str(ys[i]) + '\t' + str(std[i]) + '\t' + str(used[i]) + '\t' + str(flatused[i]) + '\t' + str((((sirm[j-1,i]/sirm[j-1,0])-(af_sirm_n[i]/af_sirm_n[0])))*100) + '\t' + str(sirm[j-1,i]/sirm[j-1,0]*100) + '\t' + str((af_sirm_n[i]/af_sirm_n[0])*100) + '\t' + str((af_nrm_n[i]/af_sirm_n[i])*100) + '\t' + str((afmag[j-1,i]/sirm[j-1,i])*100) + '\t' + str((af_nrm_n[i]/af_nrm_n[0])*100) + '\t' + str((afmag[j-1,i]/afmag[j-1,0])*100) + '\t' + str(abs(1-((sirm[j-1,i]/sirm[j-1,0])/(af_sirm_n[i]/af_sirm_n[0])))*100))
        fsirm.write('\n')
        fsirm.write(str(af_step[i]) + '\t' + str(af_sirm_n[i]/af_sirm_n[0]) + '\t' + str(sirm[j-1, i]/sirm[j-1,0]))

    fsirm.close()
    fall.write('\n')
    sirmrms /= cntfieldaf
    fall.write('SIRM MEAN DIFFERENCE % =\t' + str(100*sirmrms))

    # Compute weighted/unweighted averages, median, and dispersion
    if (averagecount !=0) and (averagecount !=1) and (sigmtot !=0):
        sampleunbias = dispersion*(sigmtot/((sigmtot**2)-vtwo))
        dispersion = dispersion/(averagecount-1)

    if (averagecount == 1):
        dispersion = 1

    if (sigmtot == 0):
        sigmtot = 1
        print('sigm tot = 0 weighted average not possible')

    if (averagecount == 0):
        averagecount = 1
        print('avercount = 0 weighted average not possible')

    aconst = -aconst*log(0.01*(tempmin)/(tempmax-tempmin))/3600
    print('Output results to averageout_{0}.dat'.format(name))
    fave = open('averageout_{}.dat'.format(name), 'w')
    if (averageme > 0):
        if (jfs != 0):
            fave.write(str(averageme/(sigmtot)) + '\t' + str(sqrt(dispersion/sigmtot)) + '\t' + str(averagecount) + '\t' + str(averageunweight/averagecount) + '\t' + str(medianme) + '\t' + str(flatcount[jfs]) + '\t' +  str(averageflat[jfs]/(1.0*flatcount[jfs])) + '\t '+  str(sigmtotflat[jfs]) + '\t' + str(flatmedianme[jfs]) + '\t' + str(aconst) + '\t' + '147' + '\t' + str(100*af_nrm_n[0]/af_sirm_n[0]) + '\t' + str(sirmrms) + '\t' + str(selrms))
        else:
            fave.write(str(averageme/sigmtot) + '\t' + str(sqrt(dispersion/sigmtot)) + '\t' + str(averagecount) + '\t' + str(averageunweight/averagecount) + '\t' + str(medianme) + '\t' + '0.0' + '\t' + '0.0' + '\t' + '0.0' + '\t' + str(aconst) + '\t' + '147' + str(100*(af_nrm_n[0]/af_sirm_n[0])) + '\t' + str(sirmrms) + '\t' + str(selrms))
    else:
        fave.write('no points selected' + '\t' + str(sirmrms))

    fave.close()
    V['shortdivlist'] = shortdivlist
    V['shortminuslist'] = shortminuslist
    V['ys'] = ys

    return X, V
       

def plot_pal(V, X):
    """
    Plot palaeointensity (TRM PI) estimates versus AF demagnetization steps.

    This function:
    - Extracts AF demagnetization steps and computed TRM intensities.
    - Plots the TRM palaeointensity curve with markers.
    - Highlights the AF pick step with a vertical line.
    - Saves the figure as a PDF.
    - Prints the mapping between AF step indices and values.

    Parameters
    ----------
    V : dict
        Dictionary containing calculated TRM palaeointensity ('ys').
    X : dict
        Dictionary containing AF demagnetization steps ('af_step'), AF pick index ('af_pick'), and sample name ('name').

    Returns
    -------
    None
    """

    # Extract TRM palaeointensity values and AF demagnetization steps
    ys = V['ys']
    af_step = X['af_step']
    name = X['name']
    afpick = X['af_pick']

    # Determine figure size with equal aspect ratio
    w, h = figaspect(1)  # from https://stackoverflow.com/questions/48190628/aspect-ratio-of-a-plot
    fig, ax = plt.subplots(figsize=(w, h))

    # Plot TRM palaeointensity curve in blue with circular markers
    plt.plot(af_step, ys[:len(af_step)], 'b')
    plt.plot(af_step, ys[:len(af_step)], marker='o', color='b')

    # Plot vertical line at AF pick step
    plt.plot([af_step[afpick], af_step[afpick]], [0, np.max(ys[:len(af_step)])], color='green')

    # Set plot labels and title
    plt.xlabel('AF peak (mT)')
    plt.ylabel('paleointensity (\u03BCT)')
    plt.title('TRM PI est {}'.format(name))

    # Save the figure as a PDF
    plt.savefig('ESA_MF-PI_ESTS_colour_{}.pdf'.format(name))
    plt.show

    # Create an array of indices for AF steps
    ind = []
    for i in range(len(af_step)):
        ind.append(i)
        i += 1

    ind = np.array(ind)

    # Print mapping of AF step indices to AF step values
    print("AF step index = AF step")
    for n, v in zip(ind, af_step):
        print("{} = {}".format(n, v))

        
def plot_zplot(X):
    """
    Generate normalized Zijderveld plots for a sample, allowing user interaction to select AF demagnetization steps.

    Parameters
    ----------
    X : dict
        Dictionary containing sample data, including:
        - 'af_nrm', 'af_nrm_inc', 'af_nrm_dec', 'af_step', 'name', 'afval', 'curie_t', 'sample_copy', 'cntfield', 'names'.

    Returns
    -------
    X : dict
        Updated dictionary including user-selected AF pick step ('af_pick').
    """
    
    # -------------------- Set defaults --------------------
    X['nat_cool'] = 720  # Default natural cooling time in hours
    c_l = 0

    # -------------------- Handle Curie temperature --------------------
    if X['sample_copy'] > 1:
        cur_1 = input(
            f"The most recent Curie temperature used for sample {X['names'][0]} was {X['curie_t']}°C. "
            "Enter K to reuse it, or any other character to input a new value: "
        )
    else:
        cur_1 = 'L'
    
    if cur_1 != 'K' or X['sample_copy'] < 2:
        while c_l == 0:
            try:
                curie_t = float(input("Input Curie temperature in °C: "))
                c_l = 1
            except ValueError:
                print('Not a number. Please input a number.')
        X['curie_t'] = curie_t

    # -------------------- Handle AF step reuse --------------------
    if X['sample_copy'] > 1:
        afval_1 = input(
            f"The most recent AF step used for sample {X['names'][0]} was {X['afval']} mT. "
            "Enter K to reuse it, or any other character to show the Zplot: "
        )
    else:
        afval_1 = 'L'

    # -------------------- Normalize AF NRM and convert to Cartesian --------------------
    if afval_1 != 'K' or X['sample_copy'] < 2:
        norm_int = X['af_nrm'] / X['af_nrm'][0]
        n_nrm = norm_int * np.cos(X['af_nrm_dec']*pi/180) * np.cos(X['af_nrm_inc']*pi/180)
        e_nrm = norm_int * np.sin(X['af_nrm_dec']*pi/180) * np.cos(X['af_nrm_inc']*pi/180)
        u_nrm = -norm_int * np.sin(X['af_nrm_inc']*pi/180)
        h_nrm = np.sqrt(n_nrm**2 + e_nrm**2)
        af_step = np.copy(X['af_step'])

        # -------------------- Plotting --------------------
        plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Horizontal projection (E,N vs N,U)
        ax1.plot(e_nrm, n_nrm, 's-', color='blue', markersize=5, linewidth=1, label='E, N')
        ax1.plot(n_nrm, u_nrm, 's-', color='red', markersize=5, linewidth=1, label='N, U')
        for i, step in enumerate(af_step):
            ax1.annotate(step, (e_nrm[i], n_nrm[i]), xytext=(0,3), textcoords='offset points', ha='left', fontsize=9)
            ax1.annotate(step, (n_nrm[i], u_nrm[i]), xytext=(0,3), textcoords='offset points', ha='left', fontsize=9)
        ax1.set_xlabel('E / N')
        ax1.set_ylabel('N / U')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_aspect('auto')
        ax1.legend()
        
        # Force axes through zero
        ax1.spines['left'].set_position('zero')
        ax1.spines['bottom'].set_position('zero')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Optional: move ticks outside
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_ticks_position('left')
        ax1.legend()
        
        
        # Vertical projection (E,U vs H,U)
        ax2.plot(e_nrm, u_nrm, 's-', color='green', markersize=5, linewidth=1, label='E, U')
        ax2.plot(h_nrm, u_nrm, 's-', color='orange', markersize=5, linewidth=1, label='H, U')
        for i, step in enumerate(af_step):
            ax2.annotate(step, (e_nrm[i], u_nrm[i]), xytext=(0,3), textcoords='offset points', ha='left', fontsize=9)
            ax2.annotate(step, (h_nrm[i], u_nrm[i]), xytext=(0,3), textcoords='offset points', ha='left', fontsize=9)
        ax2.set_xlabel('E / H')
        ax2.set_ylabel('U')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_aspect('auto')
        ax2.legend()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Force axes through zero
        ax2.spines['left'].set_position('zero')
        ax2.spines['bottom'].set_position('zero')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')

        plt.suptitle(f'Normalized Zijderveld plots for sample {X["name"]}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save plot as PDF
        filename = os.path.join(f'{X["name"]}_nrf', f'zijderveld_plot_{X["name"]}.pdf')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.show()

        # -------------------- User selects AF step --------------------
        cntfield = X['cntfield']
        while True:
            afval = input("Pick the AF demag step where the primary component is identified: ")
            try:
                afpick = np.where(af_step == float(afval))[0][0]
                if 0 <= afpick <= cntfield:
                    break
            except ValueError:
                print('Not a number')
        X['af_pick'] = int(afpick)
        X['afval'] = afval
        print('Selected AF step:', afval)

    else:
        afpick = np.where(X['af_step'] == float(X['afval']))[0][0]
        X['af_pick'] = int(afpick)
        print('Selected AF step:', X['afval'])

    return X
      
    

def pick_SF(X):
    """
    Prompt the user to select the lowest reliable SF (Selection Factor) from a FWHM graph.

    This function:
    - Checks if the SF has already been selected and stored in X['sf_1'].
    - If not, prompts the user to reuse a previous SF value or enter a new one.
    - Ensures the entered SF is an integer between 2 and 5.
    - Updates the X dictionary with the selected SF value.

    Parameters
    ----------
    X : dict
        Dictionary containing sample data, including:
        - 'sf_1', 'sample_copy', 'names', 'SF'.

    Returns
    -------
    X : dict
        Updated dictionary with user-selected SF value ('SF').
    """

    # Check if SF has already been confirmed
    if (X['sf_1'] != 'K'):
        sf_2 = 0.  # Flag to track valid SF input
        
        # If this is the second sample attempt, ask user if they want to reuse previous SF
        if (X['sample_copy'] > 1):
            sf_1 = input('The most recent SF used for sample {} was {}. If you want to re-use these variables enter K, to change them enter any other charactor'.format(X['names'][0], X['SF']))
        else:
            sf_1 = 'L'
        
        # If user does not reuse previous SF, prompt for a new SF
        if (sf_1 != 'K') or (X['sample_copy'] < 2):   
            while (sf_2 == 0):
                try:
                    SF = (input("Input the lowest reliable SF from the FWHM graph above:" )) 
                    X['SF'] = int(SF)  # Store user input in X dictionary
                    
                    # Validate SF is within the allowed range
                    if (int(SF) <= 5) and (int(SF) >= 2):
                        sf_2 = 1.  # Valid input received
                    else:
                        print('Not between 2 and 5, input an interger between 2 and 5.')
                except ValueError:
                    print('Not an interger. Please input an interger.')
    
    return(X)


def orthogonal_distance(params, x, y):
    m, b = params
    return (y - (m * x + b)) / np.sqrt(1 + m ** 2)

def orthogonal_regression(x, y):
    initial_guess = [1.0, 1.0]
    params, _ = opt.leastsq(orthogonal_distance, initial_guess, args=(x, y))
    #residuals = orthogonal_distance(params, x, y)
    return params #, residuals

    
    
# ====== new functions with auto-picking options =======

def TRM_acq_w_arm(X, V):
    """
    Simulate Thermoremanent Magnetization (TRM) acquisition with ARM (Anhysteretic Remanent Magnetization) addition.

    This function:
    - Uses the sample parameters in `X` and the simulation state dictionary `V`.
    - Prompts the user for an ARM bias field if not already provided.
    - Loops over a range of applied fields to simulate TRM acquisition.
    - Tracks temperature-dependent magnetization, AF demagnetization, SIRM, and ARM acquisition.
    - Stores results in the `V` dictionary including TRM, AF demag, SIRM, and ARM arrays.

    Parameters
    ----------
    X : dict
        Input dictionary containing sample-specific parameters such as:
        - SF (selection factor), curie_t, nat_cool, min_field, max_field, af_step, etc.
    V : dict
        Dictionary to store simulation variables and results, including:
        - trm_a, temp_a, afmag, sirm, arm, block variables, etc.

    Returns
    -------
    X : dict
        Updated dictionary (mostly unchanged, may contain updated fields).
    V : dict
        Updated simulation results.
    """

    # Ask user for ARM bias field if not already provided
    if 'arm_bias_uT' not in X:
        X['arm_bias_uT'] = float(input("Enter ARM bias field in microtesla (µT), e.g. 50: "))

    arm_bias_uT = X['arm_bias_uT']

    # Retrieve key sample parameters
    SF = X['SF']
    curie_t = X['curie_t']
          
    # Physical constants
    mu0 = 4*pi*1e-7
    kb = 1.3806503e-23
    tau = 10e-9
    roottwohffield = 2**(0.5)  

    # Initialize AF flags
    afone = 1
    afzero = 0

    # Number of hysterons
    num_hyss = 500000
    V['num_hyss'] = num_hyss

    # Temperature limits (Kelvin)
    tempmax = float(X['curie_t'] + 273)
    tempmin=300
    V['tempmin'] = tempmin
    V['tempmax'] = tempmax

    tempstep=1
    V['armstore'] = 0 # switch to one for ARM calculation

    # Initialize arrays for fields, AF steps, and magnetization tracking
    V['cntfield'] = len(X['af_step'])
    cntfield = V['cntfield']
    V['sirm'] = np.zeros((5, cntfield))  
    V['afmag'] = np.zeros((5, cntfield)) 
    V['af_step'] = np.copy(X['af_step'])
    V['trm_a'] = np.zeros((5, 5000))
    V['temp_a'] = np.zeros((5, 5000))
    fields = np.zeros(4)    

    # Define initial field range
    field = (X['min_field']*1e-6)/mu0
    fieldmax = (X['max_field']*1e-6)/mu0
    fieldstep = (fieldmax-field)/3.
    ct = 0

    # Loop over applied fields
    while (field < (fieldmax)):
        # Initialize per-field arrays and counters
        blockper = 0.0
        V['blockper'] = blockper
        hcstore = np.zeros(num_hyss)
        V['hcstore'] = hcstore
        histore = np.zeros(num_hyss)
        V['histore'] = histore
        V['totalm'] = 0
        V['afstore'] = 0.  

        i=0 
        temp=300
        tempt=300
        tm = 0.2 
        V['tm'] = tm

        # Generate hysteron population
        hys1, num_pop = pop_hys(num_hyss, X, V, SF)   
        ifield = ct
        ac = X['nat_cool']

        # Initialize block variables
        blockg = np.zeros(num_hyss) 
        boltz = np.zeros(num_hyss)
        blocktemp = np.zeros(num_hyss) 
        V['blocktemp'] = blocktemp
        V['boltz'] = boltz
        V['blockg'] = blockg
        V['totalm'] = 0

        # Setup temperature stepping
        tempstep1=(tempmax-tempmin)/500.0
        tm = 0.2 
        V['tm'] = tm

        temp = tempmax
        rate = 1 
        V['rate'] = rate
        afswitch = afzero

        V['hys'] = copy.deepcopy(hys1)
        tm = 0
        V['tm'] = tm
        track = 0
        temp = curie_t + 273 - 1

        # TRM acquisition loop over temperature
        while (temp > tempmin):
            aconst=(-ac*60.0*60.0)/(log(0.01*(tempmin)/(1000.0+273.0+-tempmin))) 
            V['aconst'] = aconst
            beta = (1-(temp-273)/curie_t)**0.43
            V['beta'] = beta
            V = blockfind(temp, field, afzero, V, X) 

            V['trm_a'][ifield][track] = V['totalm']
            V['temp_a'][ifield][track] = temp
            track +=1
            temp = temp - tempstep1

        # Final zero-field step at ambient temperature
        rate = 0
        V['rate'] = rate
        tm = 60
        V['tm'] = tm
        temp = tempmin + 0.1 
        beta = (1-(temp-273)/578.0)**0.43
        V['beta'] = beta
        fieldzero = 0.0
        V = blockfind(temp, fieldzero, afzero, V, X) 
        
        V['trm_a'][ifield][track] = V['totalm'] 
        V['temp_a'][ifield][track] = temp
        track+=1

        # AF demagnetization of TRM at 273 K
        V['afmag'][ct, 0] = V['totalm'] 
        for i in range(cntfield):
            V['afstore'] = V['af_step'][i]
            blockfind(temp, fieldzero, afone, V, X) 
            V['afmag'][ct, i] = V['totalm'] 

        # SIRM acquisition at 300 K
        blockg = np.ones(num_hyss)
        boltz = np.ones(num_hyss)
        blocktemp = np.ones(num_hyss) 
        V['blocktemp'] = blocktemp
        V['boltz'] = boltz
        V['blockg'] = blockg
        V['afstore'] = 0. 
        blockfind(temp, fieldzero, afone, V, X)
        V['sirm'][ifield,0] = V['totalm'] 
        for i in range(cntfield): 
            afstore = V['af_step'][i]
            V['afstore'] = afstore
            blockfind(temp, fieldzero, afone, V, X)
            V['sirm'][ifield,i] = V['totalm'] 

        # ARM acquisition
        blockg = np.ones(num_hyss)        
        boltz = np.ones(num_hyss)         
        blocktemp = np.ones(num_hyss)    
        V['blockg'] = blockg
        V['boltz'] = boltz
        V['blocktemp'] = blocktemp
        V['armstore'] = 1
        blockper = 1.0
        V['blockper'] = blockper

        bias_field = (arm_bias_uT * 1e-6) / mu0  
        if 'arm' not in V:
            V['arm'] = np.zeros_like(V['sirm'])

        # Acquire initial ARM with bias
        V['afstore'] = 0.0 
        blockfind(temp, bias_field, afone, V, X)      
        V['armstore'] = 0
        blockfind(temp, fieldzero, afzero, V, X)            
        V['arm'][ifield, 0] = V['totalm']

        # Loop over AF demag steps for ARM
        for i in range(cntfield):
            V['afstore'] = V['af_step'][i]
            blockfind(temp, fieldzero, afone, V, X)
            V['arm'][ifield, i] = V['totalm']  

        # Update field tracking
        fields[ct] = field
        field = field + fieldstep
        V['aconst'] = aconst
        V['fields'] = fields
        ifield = ifield +1 
        V['aconst'] = aconst
        V['ifield'] = ifield
       
        ct +=1

    return X, V

   
    
def find_and_report_plateau_candidates_regression(af_step, ys, window=3, slope_tol=None):
    """
    identifies plateau regions via moving-window linear regression.
    returns the best plateau (lowest |slope|) and all candidate windows.
    
    """
    af = np.asarray(af_step, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(af)
    if n < window:
        print("Not enough AF steps for regression window.")
        return None, []

    cand = []
    for i in range(n - window + 1):
        xi = af[i:i + window]
        yi = ys[i:i + window]

        if np.any(np.isnan(xi)) or np.any(np.isnan(yi)):
            continue
        if np.allclose(xi, xi[0]):
            continue  # zero spacing

        try:
            p, cov = np.polyfit(xi, yi, 1, cov=True)
            slope, intercept = p
            slope_se = np.sqrt(cov[0, 0]) if cov.shape == (2, 2) else np.nan
        except Exception:
            continue

        y_fit = np.polyval(p, xi)
        resid_std = float(np.std(yi - y_fit))
        norm_slope = abs(slope) / (np.nanmean(np.abs(yi)) + 1e-12)

        cand.append({
            'low_idx': i,
            'up_idx': i + window - 1,
            'af_range': (xi[0], xi[-1]),
            'slope': slope,
            'slope_se': slope_se,
            'resid_std': resid_std,
            'norm_slope': norm_slope
        })

    if not cand:
        print("No valid regression windows found.")
        return None, []

    cand.sort(key=lambda c: (abs(c['slope']), c['resid_std'], c['norm_slope']))
    best = cand[0]
    lo, hi = best['af_range']

    print(f"Best plateau: AF {lo:.2f}–{hi:.2f} mT | slope={best['slope']:.3g} ± {best['slope_se']:.3g}, "
          f"resid_std={best['resid_std']:.3g}, norm_slope={best['norm_slope']:.3g}")

    if slope_tol is not None and abs(best['slope']) > slope_tol:
        print(f"[WARN] Plateau slope {best['slope']:.3g} exceeds threshold ({slope_tol}).")

    return (best['low_idx'], best['up_idx']), cand


    
def fin_pal_SIRM_auto(X, V):
    """
    Final plotting and summary function for SIRM-based paleointensity analysis.

    This function:
    - Reads AF demagnetization spectra and paleointensity results from dictionaries X and V.
    - Attempts automatic plateau selection; falls back to manual selection if slope exceeds threshold.
    - Calculates basic statistics (mean, median, standard deviation, IQR) of selected plateau.
    - Produces a 3-panel figure showing: 
        (a) SIRM AF spectra (simulated vs measured),
        (b) SIRM checks (S_ratio, S_diff),
        (c) Paleointensity estimates with plateau highlighted.
    - Saves the figure as PDF and writes paleointensity values and plateau statistics to a text file.
    - Stores plateau metadata in V for downstream use.

    Parameters
    ----------
    X : dict
        Dictionary containing sample-specific info, AF steps, and selected AF step.
    V : dict
        Dictionary containing simulation results and measurements (ys, std, sirm, etc.)

    Returns
    -------
    None
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # read inputs
    ys = np.asarray(V['ys'])
    std = np.asarray(V['std'])
    af_step = np.asarray(X['af_step'])
    name = X['name']
    cntfield = X['cntfield']
    demagstep = X['af_step'] 

    # print AF steps for user info
    print('AF Demag spectra for the NRM (use to pick the AF steps)')
    for i, a in enumerate(demagstep):
        print('AF field step', i, a)
        
    af_step = np.asarray(af_step)
    ys_arr = np.asarray(ys)

    # attempt automatic plateau selection
    slope_tol = 0.02
    plateau_pair, candidates = find_and_report_plateau_candidates_regression(af_step, ys_arr, window=3, slope_tol=None)
    slope = abs(candidates[0]['norm_slope'])                                                           
    
    # check slope against threshold, use manual selection if too steep
    if slope > slope_tol:
        print("Automatic plateau greater than threshold; falling back to manual selection.")
        while True:
            low_b_val = input("Pick the AF step for the LOWER bound of the plateau region: ")
            try:
                low_b1 = np.where(af_step == float(low_b_val))
                low_b = int(low_b1[0])
                if 0 <= low_b <= cntfield:
                    break
            except Exception:
                print('Not a valid AF step')

        while True:
            up_b_val = input("Pick the AF step for the UPPER bound of the plateau region: ")
            try:
                up_b1 = np.where(af_step == float(up_b_val))
                up_b = int(up_b1[0])
                if (up_b >= low_b) and (up_b <= cntfield):
                    break
                else:
                    print('Out of bounds, must be above the lower bound')
            except Exception:
                print('Not a valid AF step')
    else:
        # automatic plateau selection
        low_b, up_b = plateau_pair
        print(f"Proceeding with automatically chosen plateau: AF steps {af_step[low_b]:.2f} to {af_step[up_b]:.2f} mT2")
        # stats for automatic selection
        selected_mean = np.mean(ys[low_b:up_b+1])
        mean_dev = np.std(ys[low_b:up_b+1])
        selected_med = np.median(ys[low_b:up_b+1])
        q3, q1 = np.percentile(ys[low_b:up_b+1], [75, 25])
        iqr = q3 - q1
        print(f"median: {selected_med:.2f} μT")
        print(f"mean: {selected_mean:.2f} ± {mean_dev:.2f} μT")

    # calculate plateau stats
    selected_vals = ys[low_b:up_b+1]
    selected_mean = np.mean(selected_vals)
    mean_dev = np.std(selected_vals)
    selected_med = np.median(selected_vals)
    q3, q1 = np.percentile(selected_vals, [75, 25])
    iqr = q3 - q1
    print(f"median: {selected_med:.2f} μT")
    print(f"mean: {selected_mean:.2f} ± {mean_dev:.2f} μT")

    # begin plots
    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # (a) SIRM AF spectra: simulated vs measured
    ax1.plot(af_step, V.get('sirm_plot', ys), marker='o', color='r', label='simulated')
    ax1.plot(af_step, V.get('af_sirm_n_n', ys), marker='o', color='b', label='measured')
    ax1.set_ylim(0, 1.3)
    ax1.set_xlim(0, np.nanmax(af_step))
    ax1.set_ylabel('normalised SIRM')
    ax1.set_xlabel('AF peak (mT)')
    ax1.legend(loc='upper right')
    ax1.set_title('SIRM AF spectra')
    ax1.grid()

    # (b) SIRM checks (S_ratio, S_diff)
    twenty = np.full_like(af_step, 20, dtype=float)
    hundred = np.full_like(af_step, 100, dtype=float)
    ax2.plot(af_step, twenty, 'b')
    ax2.plot(af_step, hundred, 'r')
    ax2.plot([af_step[0], af_step[-1]], [-20, -20], 'b')
    ax2.plot([af_step[0], af_step[-1]], [-100, -100], 'r')
    ax2.plot(af_step, V.get('shortdiv', np.zeros_like(af_step)), marker='o', color='r', label='S$_{ratio}$')
    ax2.plot(af_step, V.get('shortminus', np.zeros_like(af_step)), marker='o', color='b', label='S$_{diff}$')
    ax2.set_title('SIRM checks')
    ax2.set_xlabel('AF peak (mT)')
    ax2.set_ylabel('S$_{diff}$ or S$_{ratio}$ (%)')
    ax2.set_xlim(0, np.nanmax(af_step))
    ax2.set_ylim(-130, 130)
    ax2.legend(loc='upper right')
    ax2.grid()

    # (c) Paleointensity estimates: plot PI estimates and highlight plateau
    ax3.plot(af_step, ys[:len(af_step)], 'b', marker='o', label='All')
    if std is not None:
        ax3.errorbar(af_step, ys[:len(af_step)], yerr=std[:len(af_step)], fmt='o', ecolor='b', capsize=4, elinewidth=1.5)

    # mark plateau in red
    ax3.plot(af_step[low_b:up_b+1], ys[low_b:up_b+1], marker='o', color='r', label='Plateau')
    if std is not None:
        ax3.errorbar(af_step[low_b:up_b+1], ys[low_b:up_b+1], yerr=std[low_b:up_b+1],
                     fmt='o', ecolor='r', capsize=4, elinewidth=1.5)

    # vertical line marking af_pick
    afpick = X.get('af_pick', 0)
    ax3.plot([af_step[afpick], af_step[afpick]], [0, np.max(ys) * 1.1], color='green')

    # axes / labels
    ax3.set_xlim(0, np.nanmax(af_step))
    ax3.set_ylim(0, (np.max(ys) + 0.1 * np.max(ys)))
    ax3.grid()
    ax3.set_title('PI estimates')
    ax3.set_xlabel('AF peak (mT)')
    ax3.set_ylabel('paleointensity (μT)')

    # print stats on the figure (placed below x-axis)
    ax3.text(max(af_step) / 2, -(0.3 * np.max(ys)), f'median: {selected_med:.2f} μT')
    ax3.text(max(af_step) / 2, -(0.4 * np.max(ys)), f'mean: {selected_mean:.2f} ± {mean_dev:.2f} μT')

    plt.suptitle(f'PI SIRM results for sample {name}')

    # save figure and values
    outdir = os.path.join(f'{name}_nrf')
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f'PI_SIRM_{name}.pdf')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.pause(1)

    # write PI values to text file
    vals_filename = os.path.join(outdir, f'PI_SIRM_values_{name}.txt')
    with open(vals_filename, 'w') as paleo_data:
        for i in range(len(af_step)):
            paleo_data.write(f"{name}\t{af_step[i]}\t{ys[i]}\n")
            
        paleo_data.write("\n Plateau selection averages")
        paleo_data.write(f'median: {selected_med:.2f} μT\n')
        paleo_data.write(f'mean: {selected_mean:.2f} ± {mean_dev:.2f} μT\n')

    # store selection metadata in V for downstream use
    V['sirm_selected_plateau'] = (int(low_b), int(up_b))
    V['sirm_selected_mean'] = float(selected_mean)
    V['sirm_selected_std'] = float(mean_dev)
    V['sirm_selected_median'] = float(selected_med)
    V['sirm_selected_iqr'] = float(iqr)

    return


def fin_pal_ARM_auto(X, V):
    """
    Final plotting and summary function for ARM-based paleointensity analysis.

    This function:
    - Reads AF demagnetization spectra and ARM paleointensity results from dictionaries X and V.
    - Attempts automatic plateau selection; falls back to manual selection if slope exceeds threshold.
    - Calculates basic statistics (mean, median, standard deviation, IQR) of the selected plateau.
    - Produces a 3-panel figure showing:
        (a) Normalized ARM AF spectra (simulated vs measured),
        (b) ARM checks (S_ratio, S_diff),
        (c) ARM paleointensity estimates with plateau highlighted.
    - Saves the figure as PDF and writes paleointensity values and plateau statistics to a text file.
    - Stores plateau metadata in V for downstream use.

    Parameters
    ----------
    X : dict
        Dictionary containing sample-specific info, AF steps, and selected AF step.
    V : dict
        Dictionary containing simulation results and measurements (arm_ys, std, arm_plot, etc.)

    Returns
    -------
    None
    """

    # extract data from input dictionaries
    ys = V['arm_ys']  # ARM paleointensity values
    std = V['std']    # uncertainties
    af_step = X['af_step']  # AF demagnetization steps
    name = X['name']       # sample name
    cntfield = X['cntfield']  
    demagstep = X['af_step']  

    # print AF steps for user reference
    print('AF Demag spectra for the ARM (use to pick the AF steps)')
    for i in range(len(demagstep)):
        print('AF field step', i, demagstep[i])

    # convert to arrays for calculations
    af_step = np.asarray(af_step)
    ys_arr = np.asarray(ys)

    # attempt automatic plateau selection
    slope_tol = 0.02
    plateau_pair, candidates = find_and_report_plateau_candidates_regression(af_step, ys_arr, window=3, slope_tol=None)
    slope = abs(candidates[0]['norm_slope'])

    # check slope against threshold; fallback to manual selection if too steep
    if slope > slope_tol:
        print("Automatic plateau greater than threshold; falling back to manual selection.")
        while True:
            low_b_val = input("Pick the AF step for the LOWER bound of the plateau region: ")
            try:
                low_b1 = np.where(af_step == float(low_b_val))
                low_b = int(low_b1[0])
                if (low_b >= 0) and (low_b <= cntfield):
                    break
            except:
                print('Not a valid AF step')

        while True:
            up_b_val = input("Pick the AF step for the UPPER bound of the plateau region: ")
            try:
                up_b1 = np.where(af_step == float(up_b_val))
                up_b = int(up_b1[0])
                if (up_b >= low_b) and (up_b <= cntfield):
                    break
                else:
                    print('Out of bounds, must be above the lower bound')
            except:
                print('Not a valid AF step')
    else:
        # automatic plateau selection
        low_b, up_b = plateau_pair
        print(f"Proceeding with automatically chosen plateau: AF steps {af_step[low_b]:.2f} to {af_step[up_b]:.2f} mT")
        
        # compute basic statistics for selected plateau
        selected_mean = np.mean(ys[low_b:up_b+1])
        mean_dev = np.std(ys[low_b:up_b+1])
        selected_med = np.median(ys[low_b:up_b+1])
        q3, q1 = np.percentile(ys[low_b:up_b+1], [75, 25])
        iqr = q3 - q1
        print(f"median: {selected_med:.2f} μT")
        print(f"mean: {selected_mean:.2f} ± {mean_dev:.2f} μT")

    # begin plots
    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # (a) Normalized ARM AF spectra: simulated vs measured
    arm_plot = V.get('arm_plot', ys)
    ax1.plot(af_step, arm_plot, marker='o', color='r', label='simulated')
    ax1.plot(af_step, V.get('af_arm_n_n', ys), marker='o', color='b', label='measured')
    ax1.set_ylim(0, 1.3)
    ax1.set_xlim(0, np.nanmax(af_step))
    ax1.set_ylabel('Normalized ARM')
    ax1.set_xlabel('AF peak (mT)')
    ax1.legend(loc='upper right')
    ax1.set_title('ARM AF spectra')
    ax1.grid()

    # (b) ARM checks: S_ratio and S_diff
    ax2.plot(af_step, V.get('arm_shortdiv', np.zeros_like(af_step)), marker='o', color='r', label='S$_{ratio}$')
    ax2.plot(af_step, V.get('arm_shortminus', np.zeros_like(af_step)), marker='o', color='b', label='S$_{diff}$')
    ax2.set_xlabel('AF peak (mT)')
    ax2.set_ylabel('S$_{diff}$ or S$_{ratio}$ (%)')
    ax2.set_xlim(0, np.nanmax(af_step))
    ax2.set_ylim(-130, 130)
    ax2.set_title('ARM checks')
    ax2.legend(loc='upper right')
    ax2.grid()

    # (c) ARM paleointensity estimates
    ax3.plot(af_step, ys[:len(af_step)], 'b', label='All')
    if std is not None:  # add error bars if std exists
        ax3.errorbar(af_step, ys[:len(af_step)], yerr=std[:len(af_step)], fmt='o', ecolor='b', capsize=4, elinewidth=1.5)
    else:
        ax3.plot(af_step, ys[:len(af_step)], 'o', color='b')

    # highlight selected plateau region
    ax3.plot(af_step[low_b:up_b+1], ys[low_b:up_b+1], 'o', color='r', label='Plateau')
    if std is not None:
        ax3.errorbar(af_step[low_b:up_b+1], ys[low_b:up_b+1], yerr=std[low_b:up_b+1], fmt='o', ecolor='r', capsize=4, elinewidth=1.5)

    # axes labels and limits
    ax3.set_xlim(0, np.nanmax(af_step))
    ax3.set_ylim(0, np.max(ys) * 1.2)
    ax3.grid()
    ax3.set_title('ARM paleointensity estimates')
    ax3.set_xlabel('AF peak (mT)')
    ax3.set_ylabel('Paleointensity (μT)')

    # show statistics on figure
    ax3.text(max(af_step)/2, -(0.3*np.max(ys)), r'median: %.2f μT' % selected_med)
    ax3.text(max(af_step)/2, -(0.4*np.max(ys)), r'mean: %.2f ± %.2f μT' % (selected_mean, mean_dev))

    plt.suptitle(f'ARM-based PI results for sample {name}')
    plt.show()

    # save figure
    outdir = os.path.join(f'{name}_nrf')
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f'PI_ARM_{name}.pdf')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    plt.pause(1)

    # write PI values to text file
    vals_filename = os.path.join(outdir, f'PI_ARM_values_{name}.txt')
    with open(vals_filename, 'w') as paleo_data:
        for i in range(len(af_step)):
            paleo_data.write(f"{name}\t{af_step[i]}\t{ys[i]}\n")
        paleo_data.write("\n Plateau selection averages")
        paleo_data.write(f'median: {selected_med:.2f} μT\n')
        paleo_data.write(f'mean: {selected_mean:.2f} ± {mean_dev:.2f} μT\n')

    # save plateau metadata in V
    V['arm_selected_plateau'] = (int(low_b), int(up_b))
    V['arm_selected_mean'] = float(selected_mean)
    V['arm_selected_std'] = float(mean_dev)
    V['arm_selected_median'] = float(selected_med)
    V['arm_selected_iqr'] = float(iqr)

    return



def calc_PI_checks_SIRM(V, X):
    """
    Calculate SIRM-based paleointensity (PI) checks and generate related plots.

    This function:
    - Normalizes simulated and measured SIRM demagnetization curves.
    - Computes simulated NRM/SIRM ratios for each AF step.
    - Performs linear regression to estimate paleointensity for each AF step.
    - Calculates 95% confidence intervals for the PI estimates.
    - Computes S_ratio and S_diff checks for SIRM quality control.
    - Generates a 3-panel figure:
        1. Normalized SIRM demagnetization spectra (simulated vs measured)
        2. SIRM checks (S_ratio, S_diff)
        3. Paleointensity estimates for each AF step
    - Stores computed PI, error, and SIRM check arrays in V.

    Parameters
    ----------
    V : dict
        Dictionary containing simulation results (sirm, afmag, fields, etc.)
    X : dict
        Dictionary containing sample-specific info (cntfield, name, af_step, af_pick, af_irm, af_nrm)

    Returns
    -------
    X, V : dict
        Updated input dictionaries with PI results and SIRM checks.
    """

    # Extract simulated SIRM data
    sirm = V['sirm']
    
    cntfield = X['cntfield']
    name = X['name']
    ifield = V['ifield']
    demagstep = X['af_step']

    # Prepare AF steps for plotting (avoid zero for log or division issues)
    demagstep2 = demagstep
    demagstep2[0] = 0.0001
    demagstep2 = demagstep2[demagstep2 != 0]

    # Normalize simulated SIRM curves to mean of first three fields
    sirm2 = sirm
    sirmn = np.copy(sirm2)
    for i in range(V['ifield']):
        for j in range(cntfield):
            sirmn[i][j] = sirm2[i][j] / np.mean(sirm2[i][0:3])

    # Normalize measured SIRM
    af_sirm_n = X['af_irm']
    norm = np.mean(af_sirm_n[0:3])
    af_sirm_n_n = np.copy(af_sirm_n)
    for i in range(len(af_sirm_n)):
        af_sirm_n_n[i] = af_sirm_n[i] / norm
    V['af_sirm_n_n'] = af_sirm_n_n

    # Prepare SIRM plot data
    sirm_p = sirmn[:ifield, :cntfield]
    V['sirm_plot'] = sirm_p[2]

    # Initialize figure
    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1: Normalized SIRM demagnetization spectra
    ax1.plot(demagstep2, af_sirm_n_n, marker='o', color='b', label='measured')
    ax1.plot(demagstep2, sirm_p[2], marker='o', color='r', label='simulated')
    ax1.set_ylim(0, 1.3)
    ax1.set_xlim(0, np.nanmax(demagstep2))
    ax1.set_ylabel('SIRM (normalized)')
    ax1.set_xlabel('AF peak (mT)')
    ax1.legend(loc='upper right')
    ax1.set_title('SIRM demagnetization spectra for sample {}'.format(name))
    ax1.grid()

    # Extract additional data for calculations
    afmag = V['afmag']
    af_nrm_n = X['af_nrm']
    af_sirm_n = X['af_irm']
    fields = V['fields']
    af_step = X['af_step']
    afpick = X['af_pick']


    sit = ifield

    # Initialize arrays for PI and SIRM checks
    ys = np.zeros((100))
    std = np.zeros((100))
    cntfieldaf = cntfield
    af_step_list = []
    shortdiv = np.zeros((cntfield))
    shortminus = np.zeros((cntfield))
    res = []
    sumdi = 0.0
    flat = 0
    afratio = 0

    # Loop over each AF demagnetization step
    for i in range(cntfieldaf):
        xlist = []
        ylist = []
        sumx = 0.
        sumy = 0.
        sumxy = 0.
        sumxx = 0.
        af_sirm = afmag[:sit, i] / sirm[:sit, i]
        field_t = fields[:sit] * 1e-6 * 4 * np.pi * 1e-7 * 1e6  # convert to μT
        af_sirm_new = np.zeros(int(sit+1))
        field_t_new = np.zeros(int(sit+1))
        af_sirm_new[1:] = af_sirm
        field_t_new[1:] = field_t

        # Fit linear regression of simulated NRM/SIRM vs field
        for j in range(sit):
            y = fields[j] * 4 * np.pi * 1e-7 * 1e6
            x = afmag[j, i] / sirm[j, i]
            xlist.append(x)
            ylist.append(y)
            sumx += x
            sumxx += x**2
            sumxy += x * y
            sumy += y

        mfit = (((sit+0.0) * sumxy) - sumx * sumy) / (((sit+0.0) * sumxx) - sumx * sumx)
        cfit = (sumy * sumxx - sumx * sumxy) / ((sit+0.0) * sumxx - sumx * sumx)

        # Compute residuals and 95% confidence interval
        sumdi = 0.
        dilist = []
        for j in range(sit):
            y = fields[j] * 4 * np.pi * 1e-7 * 1e6
            x = afmag[j, i] / sirm[j, i]
            di = y - (mfit * x + cfit)
            dilist.append(di)
            sumdi += di**2
        sumdi /= (sit - 2)
        sigm = sit * sumdi / (sit * sumxx - sumx * sumx)
        xa = af_nrm_n[i] / af_sirm_n[i]
        ys[i] = xa * mfit + cfit

        # Linear fit with numpy polyfit for comparison
        p, cov = np.polyfit(af_sirm_new, field_t_new, 1, cov=True)
        mfit, cfit = p
        sigma_m, sigma_b = np.sqrt(np.diag(cov))
        y_pred = mfit * af_sirm_new + cfit
        residual = field_t_new - y_pred
        rss = np.sum(residual**2)
        sigma_residual = np.sqrt(rss / (len(af_sirm_new) - 2))
        
        std[i] = sqrt(sigm)*xa
        ya = xa*mfit +cfit
        ymax = xa*(mfit+std[i]) +cfit
        ymin = xa*(mfit-std[i]) +cfit
        std[i] = sqrt(sigm)
        V['std'] = std
    
        sigma_y = np.sqrt(sigma_m**2 * xa**2 + sigma_b**2 + sigma_residual**2)
        std[i] = sigma_y * 1.96  # 95% confidence interval
        res.append(cov)

        # Compute SIRM quality checks
        j = sit - 1
        shortdiv[i] = abs(1 - ((sirm[j-1, i] / np.mean(sirm[j-1, 0:3])) / (af_sirm_n[i] / np.mean(af_sirm_n[0:3])))) * 100
        shortminus[i] = abs((sirm[j-1, i] / np.mean(sirm[j-1, 0:3])) - (af_sirm_n[i] / np.mean(af_sirm_n[0:3]))) * 100
        af_step_list.append(af_step[i])

    # Panel 2: SIRM checks
    twenty = [20] * len(af_step_list)
    hundred = [100] * len(af_step_list)
    ax2.plot(af_step_list, twenty, 'b')
    ax2.plot(af_step_list, hundred, 'r')
    ax2.set_ylim(-130, 130)
    ax2.plot([af_step_list[0], af_step_list[-1]], [-20, -20], 'b')
    ax2.plot([af_step_list[0], af_step_list[-1]], [-100, -100], 'r')
    ax2.plot(af_step_list, shortdiv, marker='o', color='r', label='S$_{ratio}$')
    ax2.plot(af_step_list, shortminus, marker='o', color='b', label='S$_{diff}$')
    ax2.set_title('SIRM checks for sample {}'.format(name))
    ax2.set_xlabel('AF peak (mT)')
    ax2.set_ylabel('S$_{diff}$ or S$_{ratio}$ (%)')
    ax2.set_xlim(0, np.nanmax(af_step_list))
    ax2.legend(loc='upper right')
    ax2.grid()

    # Panel 3: Paleointensity estimates
    ax3.plot(af_step_list, ys[:len(af_step_list)], 'b', label='All')
    ax3.plot(af_step_list, ys[:len(af_step_list)], marker='o', color='b')
    ax3.errorbar(af_step, ys[:len(af_step)], yerr=std[:len(af_step)], fmt='o', ecolor='b', capsize=4, elinewidth=1.5)
    ax3.set_xlim(0, np.nanmax(af_step_list)*1.1)
    ax3.set_ylim(0, np.max(ys)*1.1)
    ax3.plot([af_step_list[afpick], af_step_list[afpick]], [0, np.max(ys[:len(af_step)])*1.1], color='green')
    ax3.set_xlabel('AF peak (mT)')
    ax3.set_ylabel('paleointensity (\u03BCT)')
    ax3.grid()
    ax3.set_title('PI for each AF step for sample {}'.format(name))

    plt.show()

    # Store results in V
    V['shortdiv'] = shortdiv
    V['shortminus'] = shortminus
    V['ys'] = ys  

    return (X, V)

def calc_PI_checks_ARM(V, X):
    """
    Calculate ARM-based paleointensity (PI) checks and generate plots.

    This function:
    - Normalizes simulated and measured ARM demagnetization curves.
    - Computes ARM-derived PI estimates for each AF step using linear regression.
    - Calculates 95% confidence intervals for each estimate.
    - Computes S_ratio and S_diff for quality checks of ARM demagnetization.
    - Produces a 3-panel figure:
        1. Normalized ARM demagnetization spectra (simulated vs measured)
        2. ARM quality checks (S_ratio, S_diff)
        3. ARM-based paleointensity estimates
    - Stores computed PI, error, and ARM check arrays in V for downstream analysis.

    Parameters
    ----------
    V : dict
        Dictionary containing simulation results (arm, afmag, fields, etc.)
    X : dict
        Dictionary containing sample-specific info (cntfield, name, af_step, af_arm, af_nrm, af_pick)

    Returns
    -------
    X, V : dict
        Updated input dictionaries with ARM results and checks.
    """

    # Extract simulated ARM data
    arm = V['arm']  
    cntfield = X['cntfield']
    name = X['name']
    ifield = V['ifield']
    demagstep = X['af_step']

    # Prepare AF steps for plotting (avoid zero for log or division issues)
    demagstep2 = np.copy(demagstep)
    demagstep2[0] = 0.0001
    demagstep2 = demagstep2[demagstep2 != 0]

    # Normalize simulated ARM curves to mean of first three points
    arm2 = np.copy(arm)
    armn = np.copy(arm2)
    for i in range(V['ifield']):
        for j in range(cntfield):
            armn[i][j] = arm2[i][j] / np.mean(arm2[i][0:3])

    # Normalize measured ARM demagnetization
    af_arm_n = X['af_arm']
    norm = np.mean(af_arm_n[0:3])
    af_arm_n_n = np.copy(af_arm_n)
    for i in range(len(af_arm_n)):
        af_arm_n_n[i] = af_arm_n[i] / norm
    V['af_arm_n_n'] = af_arm_n_n

    # Prepare ARM plot data
    arm_p = armn[:ifield, :cntfield]
    V['arm_plot'] = arm_p[2]

    # Initialize figure
    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1: ARM AF spectra
    ax1.plot(demagstep2, af_arm_n_n, marker='o', color='b', label='measured')
    ax1.plot(demagstep2, arm_p[2], marker='o', color='r', label='simulated')
    ax1.set_ylim(0, 1.3)
    ax1.set_xlim(0, np.nanmax(demagstep2))
    ax1.set_ylabel('ARM (normalized)')
    ax1.set_xlabel('AF peak (mT)')
    ax1.legend(loc='upper right')
    ax1.set_title(f'ARM demagnetization spectra for sample {name}')
    ax1.grid()

    # Extract additional data for calculations
    afmag = V['afmag']
    af_nrm_n = X['af_nrm']
    af_sirm_n = X['af_irm']
    fields = V['fields']
    af_step = X['af_step']
    afpick = X['af_pick']

    sit = ifield
    cntfieldaf = cntfield
    shortdiv = np.zeros(cntfield)
    shortminus = np.zeros(cntfield)
    ys = np.zeros(100)
    std = np.zeros(100)

    # Loop over each AF step to compute ARM-based paleointensity and quality checks
    for i in range(cntfieldaf):
        # Normalize simulated ARM for this step
        af_arm_nonorm = afmag[:sit, i] / arm[:sit, i]
        norm3 = np.mean(af_arm_nonorm[:3])
        af_arm = af_arm_nonorm / norm3

        # Fit linear regression: ARM vs applied field
        field_t = fields[:sit] * mu0 * 1E6
        p, cov = np.polyfit(af_arm, field_t, 1, cov=True)
        mfit, cfit = p
        sigma_m, sigma_b = np.sqrt(np.diag(cov))
        y_pred = mfit * af_arm + cfit
        residual = field_t - y_pred
        rss = np.sum(residual**2)
        sigma_residual = np.sqrt(rss / (len(af_arm) - 2))

        # Scale measured NRM to regression
        xa = af_nrm_n[i] / af_arm_n[i] 
        ys[i] = xa * mfit + cfit
        sigma_y = np.sqrt(sigma_m**2 * xa**2 + sigma_b**2 + sigma_residual**2)
        std[i] = sigma_y * 1.96  # 95% confidence interval

        # Compute ARM quality checks: S_ratio and S_diff
        j = sit - 1
        shortdiv[i] = abs(1 - ((armn[j - 1, i] / np.mean(armn[j - 1, 0:3])) /
                               (af_arm_n[i] / np.mean(af_arm_n[0:3])))) * 100
        shortminus[i] = abs(((armn[j - 1, i] / np.mean(armn[j - 1, 0:3])) -
                             (af_arm_n[i] / np.mean(af_arm_n[0:3])))) * 100

    af_step_list = af_step

    # Panel 2: ARM S_ratio / S_diff checks
    ax2.plot(af_step_list, [20]*len(af_step_list), 'b')
    ax2.plot(af_step_list, [100]*len(af_step_list), 'r')
    ax2.plot([af_step_list[0], af_step_list[-1]], [-20, -20], 'b')
    ax2.plot([af_step_list[0], af_step_list[-1]], [-100, -100], 'r')
    ax2.plot(af_step_list, shortdiv, marker='o', color='r', label='S$_{ratio}$')
    ax2.plot(af_step_list, shortminus, marker='o', color='b', label='S$_{diff}$')
    ax2.set_title(f'ARM checks for sample {name}')
    ax2.set_xlabel('AF peak (mT)')
    ax2.set_ylabel('S$_{diff}$ or S$_{ratio}$ (%)')
    ax2.set_xlim(0, np.nanmax(af_step_list))
    ax2.set_ylim(-130, 130)
    ax2.legend(loc='upper right')
    ax2.grid()

    # Panel 3: ARM-based paleointensity estimates
    ax3.plot(af_step_list, ys[:len(af_step_list)], 'b', marker='o')
    ax3.errorbar(af_step_list, ys[:len(af_step_list)], yerr=std[:len(af_step_list)],
                 fmt='o', ecolor='b', capsize=4, elinewidth=1.5)
    ax3.plot([af_step_list[afpick], af_step_list[afpick]],
             [0, np.max(ys[:len(af_step_list)]) * 1.1], color='green')
    ax3.set_xlim(0, np.nanmax(af_step_list) * 1.1)
    ax3.set_ylim(0, np.max(ys) * 1.1)
    ax3.set_xlabel('AF peak (mT)')
    ax3.set_ylabel('ARM-derived field (µT)')
    ax3.set_title(f'ARM-based PI for sample {name}')
    ax3.grid()

    plt.show()

    # Save results to V
    V['arm_shortdiv'] = shortdiv
    V['arm_shortminus'] = shortminus
    V['arm_ys'] = ys


    return X, V
