# TIMxGAL

## Overview

**TIMxGAL** is a Python package and analysis suite for forecasting and analyzing the cross-correlation of line-intensity mapping (LIM) surveys (such as the Terahertz Intensity Mapper, TIM) with galaxy surveys (such as Euclid). The codebase is designed to model, simulate, and interpret the power spectra, noise, and signal-to-noise ratios (SNR) for various survey configurations, with a focus on the CII line. It provides tools for theoretical modeling, instrument response, survey geometry, and statistical forecasting.

---

## Folder Structure

```
TIMxGAL/
│
├── xCorr_example.ipynb         # Main analysis notebook: cross-correlation forecasts, plots, SNR, etc.
├── utils2.py                   # Core utilities: window functions, k-space calculations, transfer functions, etc.
├── props2.py                   # Survey and instrument property definitions for TIM and galaxy surveys
├── props_SpaceTIM.py           # Alternative or extended survey property definitions
├── EuclidProps.py              # Euclid survey properties and configurations
├── E_lines.py                  # Emission line properties (e.g., CII)
├── deprecated/
│   └── obj.py                  # Deprecated/legacy objects (e.g., AttrDict, cosmology)
├── model_Params_from_simim/    # Model parameter files from SimIM and other sources
├── README.md                   # This file
└── ... (other supporting files)
```

---

## Key Components

### Notebooks

- **xCorr_example.ipynb**  
  The main Jupyter notebook for running cross-correlation forecasts, visualizing results, and comparing models.  
  - Imports survey properties, utilities, and emission line data.
  - Calculates power spectra, transfer functions, and SNR for different survey bins.
  - Visualizes results with Matplotlib.

### Core Modules

- **utils2.py**  
  Contains core computational utilities:
  - k-space grid and mode calculations
  - Window and transfer functions (including Gaussian and erf-based finite-volume effects)
  - Power spectrum manipulation and binning
  - Smoothing and regularization helpers

- **props2.py, props_SpaceTIM.py, EuclidProps.py**  
  Define survey and instrument properties for TIM, SpaceTIM, and Euclid, including:
  - Redshift bins, angular and spectral resolutions
  - Noise properties, voxel sizes, and survey geometry

- **E_lines.py**  
  Provides emission line properties (e.g., rest frequencies, conversion factors) for CII and other lines.

- **deprecated/obj.py**  
  Contains legacy or utility objects, such as `AttrDict` and a global cosmology instance.

---

## Usage

1. **Install dependencies**  
   Make sure you have the following Python packages installed:
   - `numpy`
   - `scipy`
   - `matplotlib`
   - `astropy`
   - (optional) `jupyter` for running notebooks

2. **Run the main notebook**  
   Open `xCorr_example.ipynb` in Jupyter or VS Code and execute the cells.  
   This will:
   - Load survey and instrument properties
   - Compute k-space grids and transfer functions
   - Calculate and plot power spectra, noise, and SNR
   - Compare theoretical models and instrument designs

3. **Modify survey parameters**  
   Edit the relevant `props*.py` files to change survey geometry, noise, or instrument settings.

4. **Extend or customize**  
   - Add new emission lines in `E_lines.py`
   - Implement new window functions or analysis routines in `utils2.py`
   - Add new survey configurations in `props2.py` or `props_SpaceTIM.py`

---

## Features

- **Flexible survey modeling:** Easily switch between different survey geometries, resolutions, and noise models.
- **Realistic window functions:** Includes both Gaussian (resolution) and erf (finite-volume) windowing.
- **Cross-correlation analysis:** Forecasts SNR for LIM x galaxy survey cross-correlations.
- **Model comparison:** Supports loading and comparing multiple theoretical models and simulation outputs.
- **Visualization:** Built-in plotting for power spectra, SNR, and survey properties.

---

## License

BSD 3-Clause License  
Copyright (c) 2025 Justin Bracks

---

## Acknowledgments

This codebase was developed for the analysis and forecasting of the Terahertz Intensity Mapper (TIM) and its synergy with the Euclid deep fields.  
If you use this code, please cite the relevant TIM and Euclid publications. 
Significant contributions were made to this code base by Shubh Agrawal and Ryan Keenan. 

---

## Contact

For questions or contributions, please contact the authors (jbracks@astro.ucla.edu) or open an issue on the project repository.