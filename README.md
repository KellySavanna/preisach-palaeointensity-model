## Using the Preisach method to estimate paleointensities

This repo tracks code developed for my MSci project.
The project aims to develop paleointensity estimation methods which use FORC and SORC-type data to simulate expected demagnetisation, and compare to measured demagnetisation to obtain an intensity estimate.

The code allows you to obtain the estimates from FORC and SORC-type (remFORC), as well as the plotted materials. The code is user-led and allows for parameters to be changed as desired. The repo includes sample data for both the auto-run notebooks and the manual notebooks.

### Project Abstract

Accurate palaeointensity estimates are essential for reconstructing the past geomagnetic field, but traditional Thellier‑type methods use repeated heating steps and suffer from thermal alteration, resulting in high rejection rates. Alternative isothermal methods avoid alteration by measuring palaeointensity at ambient temperatures, however, they remain challenging from a theoretical standpoint. In this study, I evaluate an isothermal palaeointensity estimation technique, based on the Preisach model, which inputs first‑order reversal curve (FORC) data, simulates NRM and SIRM acquisition, and inverts for a best-proxy ancient field. I test two potential improvements to this approach, against both the original Preisach method and a version of the earliest isothermal proxy method, REM'. The first improvement replaces FORCs (in‑field measurement) with  remFORCs (zero-field measurement), the latter of which emphasises the remanent component and suppresses reversible components. The second improvement tests ARM as an alternative normalisation technique, as it is hypothesised to better mimic NRM acquisition compared to current SIRM methods. I apply the original and new methods to recent lava flows with known ('true') fields, comparing the palaeointensity estimates, measured-simulated curves and AF‑step behaviours of the NRM/normaliser.  The remFORC input often produced similar results to the FORC input across samples, suggesting that if improvements exist from reduced transience in the remFORC, it was offset by the coarser resolution. ARM normalisation systematically underestimated the palaeointensity estimate; SIRM performed better. However, the method produced clearer plateaus suggesting better agreement between ARM and NRM spectra, and better matching of measured-simulated data. Overall, ARM produced much lower standard deviations than SIRM, both for the individual samples and for site wide averages, implying potential for improved estimates should the systemic underestimation be a methodological problem rather than a theoretical violation due to sample properties. The Preisach model modifications tested here show promise to produce improved palaeointensity estimates, particularly the use of ARM as a normaliser, but significant refinements and larger test datasets are required before it can be considered a robust method for natural samples.

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



Code is based on / adapted from work conducted in the Natural Magnetism Group at Imperial College London.

Credits: Adrian Muxworthy, Evelyn Baker
