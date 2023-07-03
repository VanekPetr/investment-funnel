![investment-funnel](https://socialify.git.ci/VanekPetr/investment-funnel/image?font=Inter&issues=1&language=1&owner=1&stargazers=1&theme=Dark) 
<p align="center">
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white" align="center">
<img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white" align="center">
<img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white "align="center">
<img src="https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor "align="center">
</p>

Open source project for developing and backtesting investment strategies. Used by more than 500 students during 
asset allocation classes at Copenhagen University & Danish Technical University as well as by amateur quants. 
The main aim of this project is to get a better overview of the ETF/Mutual fund market, to experiment with a different investment techniques & algorithms
and finally to backtest investment strategies.<br/>


<!-- toc -->
- [Technologies and Models](#technologies-and-models)
- [How to start in 3 steps](#how-to-start-in-3-steps)
- [Usage](#usage)
- [Configuration](#configuration)
- [Authors of the project](#authors-of-the-project)
- [Research related to Investment Funnel](#research-related-to-investment-funnel)
- [Contributing](#contributing)
- [Versioning](#versioning)
- [License](#license)
<!-- tocstop -->

# Technologies and Models
### TODO cookbook
### TODO intro picture
Our tool is based on various optimization models and ML methods:<br/>
Minimum Spanning Tree, Hierarchical Clustering of Assets, Monte Carlo, Bootstrapping, Stochastic CVaR Model, Benchmark generation

# How to start in 3 steps
STEP 1: create and activate python virtual environment
``` bash
python -m venv venv
source venv/bin/activate
```

STEP 2: install requirements
``` bash
pip install -r requirements.txt
```
STEP 3: run dash application
``` bash
python app.py 
```
The app is running on http://127.0.0.1:8050

# Usage

# Configuration
write about MOSEK licence and data


## Authors of the project

* **Petr Vanek** - *Co-founder & Initial work* - [VanekPetr](https://github.com/VanekPetr)
* **Kourosh Rasmussen** - *Co-founder* - [AlgoStrata](https://algostrata.com) & [Penly](https://penly.dk)
* **Gábor Balló** - *Implementation of CVaR model with CVXPY and MOSEK* - [szelidvihar](https://github.com/szelidvihar) & [MOSEK](https://github.com/MOSEK)
* **Auður Anna Jónsdóttir** - *Initial work for MST and Hierarchical Clustering*
* **Chanyu Yang** - *First contributor to our dash application* - [cicadaa](https://github.com/cicadaa)

## Research related to Investment Funnel
* **Arnar Tjörvi Charlesson & Thorvaldur Ingi Ingimundarson** - *Self-Organizing Maps and Strategic Fund Selection* (Master Thesis, DTU, 2023)
* **Dimosthenis Karafylias** - *Deep Reinforcement Learning For Portfolio Optimisation* (Master Thesis, DTU, 2022)
* **Carlos Daniel Pinho Ventura** - *Designing Hybrid Investment Packages of Cryptocurrencies with Rewards and Index Funds* (Master Thesis, DTU, 2022)
* **Peter Emil Dinic Holmsø** - *Optimal Life Cycle Planning using Stochastic Simulation* (Master Thesis, DTU, 2021)
* **Alexandros Giannakakis & Rasmus Blirup Jensen** *AI-Based Portfolio Analysis and Risk Management of Index Funds and Cryptocurrencies* (Master Thesis, DTU, 2021)
* **Idriss El Quassimi** *Graph Theoretical Methods in Strategic Asset Allocation* (Master Thesis, DTU, 2021)
* **Jorge Bertomeu Genís** *Portfolio Optimization using Index Funds and a Basket of Cryptocurrencies* (Master Thesis, DTU, 2021)
* **Andrias Poulsen** - *Performance Analysis of Sustainable Investment Portfolios* (Bachelor Thesis, DTU, 2021)
* **Auður Anna Jónsdóttir** - *Feature Selection in Asset Allocation* (Master Thesis, DTU, 2020)
* **Petr Vanek** - *Performance Analysis of the most traded Mutual Funds versus Optimal Portfolios of Exchange Traded Funds* (Master Thesis, KU, 2020)

## Contributing
Thank you for considering contributing to this project! We welcome contributions from everyone. Before getting started, please take a moment to review our [Contribution Guidelines](CONTRIBUTING.md).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/VanekPetr/investment-funnel/tags). 

## License

This repository is licensed under [MIT](LICENSE) (c) 2019 GitHub, Inc.

<div align='center'>
<a href='https://github.com/vanekpetr/investment-funnel/releases'>
<img src='https://img.shields.io/github/v/release/vanekpetr/investment-funnel?color=%23FDD835&label=version&style=for-the-badge'>
</a>
<a href='https://github.com/vanekpetr/investment-funnel/blob/main/LICENSE'>
<img src='https://img.shields.io/github/license/vanekpetr/investment-funnel?style=for-the-badge'>
</a>
</div>

