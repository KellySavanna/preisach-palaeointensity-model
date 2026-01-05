## Using the Preisach method to estimate paleointensities

This repo tracks code developed for my MSci project.
The project aims to develop paleointensity estimation methods which use FORC and SORC-type data to simulate expected demagnetisation, and compare to measured demagnetisation to obtain an intensity estimate.

The code allows you to obtain the estimates from FORC and SORC-type (remFORC), as well as the plotted materials. The code is user-led and allows for parameters to be changed as desired. The repo includes sample data for both the auto-run notebooks and the manual notebooks.

### Project Abstract

Accurate palaeointensity estimates are essential for reconstructing the past geomagnetic field, but traditional Thellier‑type methods use repeated heating steps and suffer from thermal alteration, resulting in high rejection rates. Alternative isothermal methods avoid alteration by measuring palaeointensity at ambient temperatures, however, they remain challenging from a theoretical standpoint. In this study, I evaluate an isothermal palaeointensity estimation technique, based on the Preisach model, which inputs first‑order reversal curve data, simulates remanent acquisition, and inverts for a best-estimate ancient field. I test two potential improvements against both the original Preisach method and a development of the earliest isothermal proxy method, REM'. The first test replaces FORC inputs (in‑field) with remFORC inputs (zero-field), as remFORCs emphasise remanent components and suppress reversible components. The second test explores ARM as an alternative normaliser, as ARM is suggested to more accurately replicate NRM than the current SIRM normalisation techniques. I apply the original and new methods to recent basalts with known (`true') fields, comparing the palaeointensity estimates, demagnetisation curves and AF‑step behaviours. The remFORC and FORC inputs produced similar results, suggesting that Preisach simulations are minimally impacted by transient interactions. ARM normalisation systematically underestimated the palaeointensity estimate; SIRM performed better. However, ARM produced clearer plateaus suggesting better agreement between ARM and NRM spectra, and more accurately simulated data. Further testing of additional modifications are required to improve ARM success, providing the systemic underestimation be a methodological problem rather than a theoretical violation due to properties of the remanence carriers. The Preisach model modifications tested here show promise to produce improved palaeointensity estimates, particularly the use of ARM as a normaliser, but significant refinements and larger test datasets are required before it can be considered a robust method for natural samples.

### Using the code

The code is provided in the form of jupyter notebooks and .py files, to be run by conda/anaconda navigator.

### Installing environment

1. Clone the repo
```bash
git clone https://github.com/KellySavanna/preisach-paleointensity-model.git
```
2. Create environment from included yml file
```bash
conda env create -f environment.yml
```
3. Activate environment
```bash
conda activate palaeo-env
```

### Running notebooks

Open the jupyter notebook, follow along and run cells to see how it works. Start with a manual notebook (interactive):
```bash
jupyter notebook Manual_FRC_notebook.ipynb
```

   
### Important Files 

1. **[Manual_FRC_notebook.ipynb](Manual_FRC_notebook.ipynb)** 
 Notebook for getting a single sample's paleointensity estimate using FORC data for the simulation. User prompted notebook which walks through each function called.
2. **[Manual_NRF_notebook.ipynb](Manual_NRF_notebook.ipynb)** 
  Notebook for getting a single sample's paleointensity estimate using remFORC (SORC-type) data for the simulation. User prompted as above.
3. **[Auto_FRC_notebook.ipynb](Auto_FRC_notebook.ipynb)** 
  Auto-running notebook, which cycles through a whole file of samples, using FORC data. Some user prompting.
4. **[Auto_NRF_notebook.ipynb](Auto_NRF_notebook.ipynb)**
  As for 3., but for the remFORC data.



Code is based on / adapted from work conducted in the Natural Magnetism Group (NMG) at Imperial College London.

Credits: Adrian Muxworthy, Evelyn Baker and previous members of NMG. Large language models were used for debugging, and to add comments and docstrings in some instances. Should you find any errors, please don't hesitate to drop a PR.
