# High‑Level NEM PyPSA Model

A concise, five‑node PyPSA model of the Australian National Electricity Market (NEM), designed for learners and stakeholders exploring energy transition scenarios.

---

## Table of Contents

- [High‑Level NEM PyPSA Model](#highlevel-nem-pypsa-model)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Directory Structure](#directory-structure)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [Customization \& Scenarios](#customization--scenarios)
  - [View Quarto rendered HTML](#view-quarto-rendered-html)

---

## Overview

The **High‑Level NEM** model provides a baseline representation of the NEM’s five regions (one bus per region) at hourly resolution, using actual 2024 generation and demand data. It enables:

- Analysis of current dispatch and grid topology
- Exploration of high‑renewable scenarios with storage
- Visualisation of network topology, generation mix, dispatch and curtailment


<p align="center">
  <img src="images/scenario-dispatch.png" alt="Scenario dispatch" width="700">
</p>

![Scenario dispatch](images/scenario-dispatch.png)

**Intended Audience:**

- Students and professionals learning power system modelling
- Energy analysts and policy researchers interested in the Australian grid
- Anyone curious about the electricity transition and renewable integration

---

## Features

- **5‑bus network** representing NSW, QLD, SA, TAS, VIC
- **Hourly time series** for loads and generator availability
- **Baseline vs. Future Scenarios**: easily swap in higher VRE/storage capacities
- **Optimisation** via linear programming (HiGHS/Gurobi)
- **Interactive & static plots** of dispatch, imports/exports, and curtailment
- **Exportable** to NetCDF for external tools

---

## Directory Structure

```plaintext
.
├─ High-level_NEM.qmd                  # Main Quarto analysis/report
├─ data/                               # Input datasets (already cleaned)
│  ├─ inputs/                          # Assumption tables
│  └─ nemweb/clean/                    # NEMWeb extracts
├─ scripts/                            # Data prep & rendering helpers
├─ results/
│  └─ scenarios/                       # Scenario netCDF + summaries
└─ docs/                               # Built site & assets for html report

```

## Prerequisites

- Python 3.8+
- [PyPSA](https://pypsa.readthedocs.io/en/latest/)
- pypsa, pandas, numpy, matplotlib, cartopy, datetime, great_tables
- (Optional) Gurobi solver (requires license)

---



3. **Input Data**
   

   - Components:  `generators.csv`, `buses.csv`, `loads.csv`, `lines.csv`, `links.csv`, `storage_units.csv`
   - Time series: `loads_t.p_set.csv`, `generators_t.p_max_pu.csv`

---

## Usage

1. **Run the Quarto document**

   Open High-level_NEM.qmd in Quarto and run cells, or render to HTML by running `quarto_to_html_render.py`, or:



2. **Walk through sections**

   - **Data Import & Preprocessing**
   - **Network Construction** (buses, loads, generators, storage, lines/links)
   - **Optimisation (Solve the network)**
   - **Visualisation (senario analysis)** (`plot_dispatch` examples for static & interactive plots (plotly) and the great_tables package for variable renewable curtailment analysis)


---

## Customization & Scenarios

- **Adjust capacities**: edit `generators.csv` or use the scenario function to scale VRE/storage.
- **Temporal window**: call `plot_dispatch(n, time="YYYY-MM-DD", days=N, regions=[…])`.
- **Solver choice**: switch between HiGHS (default) and Gurobi in the optimisation cell.

---


**Key References:**

- [TU Berlin: Data Science for Energy System Modelling](https://fneum.github.io/data-science-for-esm/intro.html#jupyter.org/)  
- [PyPSA Documentation and Components](https://pypsa.readthedocs.io/en/latest/user-guide/components.html)  
- [PyPSA Earth Documentation](https://pypsa-earth.readthedocs.io/en/latest/)  
- [GitHub PyPSA Sources](https://github.com/PyPSA)  
- [PyPSA-PH: High-Resolution Open Source Power System Model for the Philippines](https://github.com/arizeosalac/PyPSA-PH/tree/main)  
- [2024 Integrated System Plan (ISP)](https://aemo.com.au/energy-systems/major-publications/integrated-system-plan-isp/2024-integrated-system-plan-isp)  
- [Open Electricity](https://openelectricity.org.au/)

## View Quarto rendered HTML

[A High-level Open Source Model for the Australian National Electricity Market (NEM)](https://chalg.github.io/pypsa/)

Note: I'm rendering a single html file (with asssociated assets), which I'm renaming and moving to the `docs/` directory via the `qaurto_to_html_render.py` script. I'm not creating a full website and my project type is not Website Project. 


Using `quarto render High-level_NEM.qmd --to html` will render the Quarto document to HTML in the root directory - however I need it to be in the `docs/` directory so that it can be served by GitHub Pages.

