![investment-funnel](https://socialify.git.ci/VanekPetr/investment-funnel/image?font=Inter&issues=1&language=1&owner=1&stargazers=1&theme=Dark) 
<p align="center">
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white" align="center">
<img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white" align="center">
<img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white "align="center">
<img src="https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor "align="center">
</p>

Open source investment tool used by students at Copenhagen University & Danish Technical University and 
amateur quants to get a better overview of the ETF market, to experiment with a different investment techniques & algorithms
and finally to backtest their investment strategies.<br/>

# Technologies
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

## Authors of the project

* **Petr Vanek** - *Co-founder & Initial work* - [VanekPetr](https://github.com/VanekPetr)
* **Kourosh Rasmussen** - *Co-founder* - [AlgoStrata](https://algostrata.com) & [Penly](https://penly.dk)
* **Gábor Balló** - *Implementation of CVaR model with CVXPY and MOSEK* - [szelidvihar](https://github.com/szelidvihar) & [MOSEK](https://github.com/MOSEK)
* **Auður Anna Jónsdóttir** - *Initial work for MST and Hierarchical Clustering*
* **Chanyu Yang** - *First contributor to our dash application* - [cicadaa](https://github.com/cicadaa)

## Research related to or with a use of Investment Funnel
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

