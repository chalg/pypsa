# High‑Level NEM PyPSA Model

A concise, five‑node PyPSA model of the Australian National Electricity Market (NEM), designed for learners and stakeholders exploring energy transition scenarios and renewable integration.

---

## Table of Contents

- [High‑Level NEM PyPSA Model](#highlevel-nem-pypsa-model)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Customization \& Scenarios](#customization--scenarios)
  - [Contributing](#contributing)
  - [License \& References](#license--references)

---

## Overview

The **High‑Level NEM** model provides a baseline representation of the NEM’s five regions (one bus per region) at hourly resolution, using actual 2024 generation and demand data. It enables:

- Analysis of current dispatch and grid topology
- Exploration of high‑renewable scenarios with storage
- Visualization of generation mix, curtailment, and interconnector flows

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

## Prerequisites

- Python 3.8+
- [PyPSA](https://pypsa.readthedocs.io/en/latest/)
- pandas, NumPy, Matplotlib, Cartopy
- (Optional) Gurobi solver for faster optimisation

---

## Installation

1. **Clone repository**

   ```bash
   git clone https://github.com/your-org/high-level-nem.git
   cd high-level-nem
   ```

2. **Create virtual environment & install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   pip install -r requirements.txt
   ```

3. **Download data**\
   Place the following CSVs into `data/`:

   - `generators.csv`, `buses.csv`, `loads.csv`
   - `lines.csv`, `links.csv`, `storage_units.csv`
   - Time series: `loads_t.p_set.csv`, `generators_t.p_max_pu.csv`

---

## Usage

1. **Run the Jupyter notebook OR Quarto**

   ```bash
   jupyter notebook High-level_NEM.ipynb
   ```
 
   OR

   ```bash
   quarto render High-level_NEM.qmd --to html
   ```

2. **Walk through sections**

   - **Data Import & Preprocessing**
   - **Network Construction** (buses, loads, generators, storage, lines/links)
   - **Optimisation & Scenario Analysis**
   - **Visualisation** (`plot_dispatch` examples for static & interactive plots)

3. **Export results**

   ```python
   n.export_to_netcdf("results/high-level_nem.nc")
   ```

---

## Project Structure

```
├── data/                         # Input CSV files
│   ├── buses.csv
│   ├── generators.csv
│   ├── loads.csv
│   ├── lines.csv
│   ├── links.csv
│   ├── storage_units.csv
│   ├── loads_t.p_set.csv
│   └── generators_t.p_max_pu.csv
├── notebooks/
│   └── High-level_NEM.ipynb      # Main modelling & analysis notebook
├── results/                      # Exported NetCDF & figures
├── requirements.txt              # Python dependencies
└── README.md                     # You are here
```

---

## Customization & Scenarios

- **Adjust capacities**: edit `generators.csv` or use the scenario function to scale VRE/storage.
- **Temporal window**: call `plot_dispatch(n, time="YYYY-MM-DD", days=N, regions=[…])`.
- **Solver choice**: switch between HiGHS (default) and Gurobi in the optimisation cell.

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/xyz`)
3. Commit changes and push (`git push origin feature/xyz`)
4. Open a Pull Request and describe your improvements

---

## License & References

**License:** MIT

**Key References:**

- PyPSA Documentation
- AEMO 2024 ISP Inputs & Assumptions
- Open Electricity Facilities Database
- TU Berlin: Data Science for Energy System Modelling

**View Quarto output**

   Open the generated Quarto markdown file [High-level\_NEM.qmd.md](High-level_NEM.md)

