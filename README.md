## Using the Preisach method to estimate paleointensities

This repo tracks code developed for my MSci project.
The project aims to develop paleointensity estimation methods which use FORC and SORC-type data to simulate expected demagnetisation, and compare to measured demagnetisation to obtain an intensity estimate.

The code allows you to obtain the estimates from FORC and SORC-type (remFORC), as well as the plotted materials. The code is user-led and allows for parameters to be changed as desired. The repo includes sample data for both the auto-run notebooks and the manual notebooks.

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
conda activate paleo-env
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
