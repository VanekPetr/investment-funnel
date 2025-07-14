<p>
  <img width="100%" src="funnel/assets/InvestmentFunnel 3.png"></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white" align="center">
<img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white" align="center">
<img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white "align="center">
<img src="https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor "align="center">
</p>


🎉 UPDATE - tests our new no-code Investment Funnel [dashboard application](https://investment-funnel.algostrata.com) 🎉

Welcome to our open-source project for developing and backtesting investment strategies.

Having been utilized by over 500 students from Asset Allocation classes at Copenhagen University and Danish Technical
University, this project is also a pivotal tool for amateur/beginner quants.

The primary goal of this project is to provide a better overview of the ETF/Mutual fund market and to allow users to
experiment with various investment techniques and algorithms. Ultimately, it offers a platform to backtest and refine investment strategies.

<!-- toc -->
- [Technologies and Models](#technologies-and-models)
  * [Portfolio optimization models](#portfolio-optimization-models)
  * [ML models](#ml-models)
  * [Scenario generation algorithms](#scenario-generation-algorithms)
- [How to start](#how-to-start)
- [Usage](#usage)
  * [Market Overview](#market-overview)
  * [AI Feature Selection](#ai-feature-selection)
  * [Backtesting](#backtesting)
  * [Develop your own model](#develop-your-own-model)
- [Further configuration for professionals or students](#further-configuration-for-professionals-or-students)
- [Authors of the project](#authors-of-the-project)
- [Research related to Investment Funnel](#research-related-to-investment-funnel)
- [Testing](#testing)
  * [Testing Dash Callbacks](#testing-dash-callbacks)
- [Contributing](#contributing)
- [Versioning](#versioning)
- [License](#license)
<!-- tocstop -->

## Technologies and Models

The Investment Funnel brings together various optimization models for asset allocation, machine learning (ML)
methodologies for feature selection, and algorithms for scenario generation. Coupled with the backtesting
framework and Dash application, it presents a comprehensive environment for the development and backtesting of
investment strategies

### Portfolio optimization models
1. Conditional Value at Risk (CVaR) model - [read more](https://docs.mosek.com/portfolio-cookbook/riskmeasures.html#conditional-value-at-risk)
2. Mean-Variance (Markowitz) model - [read more](https://docs.mosek.com/portfolio-cookbook/markowitz.html)

### ML models
1. Minimum Spanning Tree (MST) model - [read more](https://en.wikipedia.org/wiki/Minimum_spanning_tree)
2. Hierarchical Clustering (HCA) model - [read more](https://en.wikipedia.org/wiki/Hierarchical_clustering)

### Scenario generation algorithms
1. Monte Carlo scenario simulation - [read more](https://en.wikipedia.org/wiki/Monte_Carlo_method)
2. Bootstrap scenario simulation - [read more](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

To further enhance your knowledge on mathematical optimization in finance, we highly recommend the [MOSEK Portfolio Optimization Cookbook](https://github.com/MOSEK/PortfolioOptimization).

## How to start

Run the dash application by

```bash
make funnel
```

This will install [uv](https://github.com/astral-sh/uv), create a virtual environment
and will start the app running on http://127.0.0.1:8050.

## Usage

Investment Funnel contains multiple portfolio optimization models, machine learning methods and algorithms located in
[models folder](funnel/models). <br/>

Furthermore, this project contains dash application for visualizing the data, output of ML methods as well as results from backtesting.
You can explore the dash application by running [app.py](funnel/app.py) file. <br/>

### Market Overview

On the first page of our Dash application, you'll find an overview of the performance of the ETF/Mutual fund market
in terms of risk and returns. This can provide a clearer understanding of the data included in the project.

Moreover, you have the option to search and select one or more assets for a comparison against the entire
universe of assets. For even deeper insight, you can repeat this experiment for various time periods.

<p>
  <img width="100%" src="funnel/assets/market_overview.png"></a>
</p>

### AI Feature Selection

An integral part of optimal portfolio allocation involves feature selection. In this regard, we've implemented
two machine learning methods, Minimum Spanning Tree and Hierarchical Clustering, to streamline the number of assets
needed for the optimization model.

To gain a deeper understanding of these two ML models, you're afforded the opportunity to experiment with their
configurations and visualize the outcomes in interactive graphs. This empowers you to delve into which assets were
selected, and scrutinize the performance, specifically the risk and returns, of the selected assets over a given time period.

<p>
  <img width="100%" src="funnel/assets/feature_selection.png"></a>
</p>

### Backtesting

Backtesting is arguably the most crucial aspect of this project. It allows you to test your investment strategies
on historical data and compare their performance with other models.

You have the flexibility to select your own train (out-of-sample) and test (in-sample) periods. You can choose
an optimization portfolio allocation model as well as a machine learning model for feature selection - this helps
optimize the number of assets for your model.

Further customization can be achieved by specifying your machine learning model's configurations and the algorithm
for scenario generation. And lastly, you have the option to select the benchmark for comparison.

Once your backtest run completes, you will be presented with a comparative view of your optimal portfolio's performance
against this benchmark for the test period.
<p>
  <img width="100%" src="funnel/assets/backtesting1.png"></a>
</p>

This performance review will offer insights into portfolio value development, allocation to individual assets for
each investment period, as well as comparisons in terms of average annual return, standard deviation, and Sharpe ratio.

<p>
  <img width="100%" src="funnel/assets/backtesting2.png"></a>
</p>

### Develop and test your own model

Lastly, you have the option to develop your own optimization and machine learning models for portfolio allocation or
feature selection, and seamlessly integrate those into the investment funnel. By utilizing our Dash application,
you can leverage the backtesting framework to visualize your model's results and conveniently compare its performance
against those of existing models in this repository.

## Further configuration for professionals or students

Are you intrigued by the Investment Funnel project? Do you wish to utilize it for your own research, teaching,
or the development of investment strategies?

To make the best of this project, you'll likely need access to up-to-date financial data and a professional solver.

* For the data, please reach out to Kourosh Rasmussen from [AlgoStrata](https://algostrata.com/about-us) or [Petr Vanek](https://www.linkedin.com/in/vanekpetr42/). They can guide you through the next steps.
* As for the solver, we recommend using [MOSEK](https://www.mosek.com/). It's free for the first 30 days and fosters many academic collaborations.

## Authors of the project

* **Petr Vanek** - *Co-founder & Initial work* - [VanekPetr](https://github.com/VanekPetr)
* **Kourosh Rasmussen** - *Co-founder* - [AlgoStrata](https://algostrata.com) & [Penly](https://penly.dk)
* **Gábor Balló** - *Implementation of CVaR model with CVXPY and MOSEK* - [szelidvihar](https://github.com/szelidvihar) & [MOSEK](https://github.com/MOSEK)
* **Thomas Schmelzer** - *Help with [Aging of the code](https://www.linkedin.com/pulse/aging-code-thomas-schmelzer-7o4jf/), GitOps and Maintainability of an OpenSource project* - [tschm](https://github.com/tschm) & [ADIA](https://www.adia.ae) & [cvxgrp/simulator](https://github.com/cvxgrp/simulator)
* **Mikkel Bech Mogensen** - *Lifecycle investments page* - [mikkelbechmogensen](https://github.com/mikkelbechmogensen)
* **Mariska Van de Sompele** - *Implementation of minimum asset portfolio weights constraint for CVaR and Markowitz models* - [MariskaVandeSompele](https://github.com/MariskaVandeSompele)
* **Auður Anna Jónsdóttir** - *Initial work for MST and Hierarchical Clustering*
* **Chanyu Yang** - *First contributor to our dash application* - [cicadaa](https://github.com/cicadaa)
* **Alexandra Mourier** - *Design of our GitHub README banner*

## Research related to Investment Funnel

* **Arnar Tjörvi Charlesson & Thorvaldur Ingi Ingimundarson** - *Self-Organizing Maps and Strategic Fund Selection* (Master Thesis, DTU, 2023)
* **Dimosthenis Karafylias** - *Deep Reinforcement Learning For Portfolio Optimisation* (Master Thesis, DTU, 2022)
* **Carlos Daniel Pinho Ventura** - *Designing Hybrid Investment Packages of Cryptocurrencies with Rewards and Index Funds* (Master Thesis, DTU, 2022)
* **Peter Emil Dinic Holmsø** - *Optimal Life Cycle Planning using Stochastic Simulation* (Master Thesis, DTU, 2021)
* **Alexandros Giannakakis & Rasmus Jensen** *AI-Based Portfolio Analysis and Risk Management of Index Funds and Cryptocurrencies* (Master Thesis, DTU, 2021)
* **Idriss El Quassimi** *Graph Theoretical Methods in Strategic Asset Allocation* (Master Thesis, DTU, 2021)
* **Mikkel Bech Mogensen** *A Stochastic CVaR Optimization Model for Leveraged Asset Allocation Strategies* (Bachelor Thesis, KU, 2021)
* **Jorge Bertomeu Genís** *Portfolio Optimization using Index Funds and a Basket of Cryptocurrencies* (Master Thesis, DTU, 2021)
* **Andrias Poulsen** - *Performance Analysis of Sustainable Investment Portfolios* (Bachelor Thesis, DTU, 2021)
* **Auður Anna Jónsdóttir** - *Feature Selection in Asset Allocation* (Master Thesis, DTU, 2020)
* **Petr Vanek** - *Performance Analysis of the most traded Mutual Funds versus Optimal Portfolios of Exchange Traded Funds* (Master Thesis, KU, 2020)

Do you want to write your thesis on Investment Funnel? Please reach out and let us know.

## Testing

### Testing Dash Callbacks

The project includes tests for the Dash callbacks using two approaches:

1. **Traditional Testing Approach**: The `tests/dashboard/test_app_callbacks.py` file contains tests for the callbacks using a traditional approach with mocks and patches. This approach is useful for unit testing callbacks in isolation, without running a Dash application.

2. **Testing with dash[testing]**: The same file also contains tests using the dash[testing] package, which provides utilities for end-to-end testing of Dash applications. These tests are wrapped in a try-except block to handle the case where dash[testing] is not installed.

The following callbacks are tested using dash[testing]:
- `display_page`: Tests that the correct page is displayed based on the URL pathname
- `update_output`: Tests that the output text is updated based on the slider value
- `plot_backtest`: Tests that the backtest figures are updated when the button is clicked
- `plot_ml`: Tests that the ML figures are updated when the button is clicked
- `plot_lifecycle`: Tests that the lifecycle figures are updated when the button is clicked
- `plot_dots`: Tests that the dots figure is updated when the button is clicked

To run the dash[testing] tests, you need to:

1. Install dash[testing] by adding it to the dev dependencies in pyproject.toml:
   ```toml
   dev = [
       "pytest-cov>=6.0.0",
       "pytest>=8.3.3",
       "pre-commit>=4.0.1",
       "ifunnel>=0.0.6",
       "ecos>=2.0.14",
       "dash[testing]>=3.1.0",
   ]
   ```

2. Install the dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the tests:
   ```bash
   pytest tests/dashboard/test_app_callbacks.py::test_display_page_routing_with_dash_testing -v
   pytest tests/dashboard/test_app_callbacks.py::test_update_output_with_dash_testing -v
   pytest tests/dashboard/test_app_callbacks.py::test_plot_backtest_with_dash_testing -v
   pytest tests/dashboard/test_app_callbacks.py::test_plot_ml_with_dash_testing -v
   pytest tests/dashboard/test_app_callbacks.py::test_plot_lifecycle_with_dash_testing -v
   pytest tests/dashboard/test_app_callbacks.py::test_plot_dots_with_dash_testing -v
   ```

   Or run all dash[testing] tests at once:
   ```bash
   pytest tests/dashboard/test_app_callbacks.py -k "with_dash_testing" -v
   ```

#### Benefits of dash[testing]

Using dash[testing] provides several benefits over the traditional approach:

1. **End-to-end testing**: Tests the entire application stack, including the Dash framework, callbacks, and browser interactions.
2. **Real browser interaction**: Tests how the application behaves in a real browser, which can catch issues that unit tests might miss.
3. **Visual testing**: Can be used with Percy for visual regression testing.
4. **Integration testing**: Tests how different components of the application work together.

#### When to Use Each Approach

- Use the traditional approach for unit testing callbacks in isolation, when you want to test the logic of the callback function without running a Dash application.
- Use dash[testing] for integration and end-to-end testing, when you want to test how the application behaves in a real browser and how different components work together.

## Contributing

Thank you for considering contributing to this project! We welcome contributions from everyone. Before getting started, please take a moment to review our [Contribution Guidelines](CONTRIBUTING.md).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/VanekPetr/investment-funnel/tags).

## License

This repository is licensed under [MIT](LICENSE) (c) 2023 GitHub, Inc.

<div align='center'>
<a href='https://github.com/vanekpetr/investment-funnel/releases'>
<img src='https://img.shields.io/github/v/release/vanekpetr/investment-funnel?color=%23FDD835&label=version&style=for-the-badge'>
</a>
<a href='https://github.com/vanekpetr/investment-funnel/blob/main/LICENSE'>
<img src='https://img.shields.io/github/license/vanekpetr/investment-funnel?style=for-the-badge'>
</a>
</div>
