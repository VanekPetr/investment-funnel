# Practical Financial Optimization 2023

Dear students 🧑‍🎓,

Welcome to the Practical Financial Optimization (PFO) course 🎉. 

As part of your curriculum, you will be experimenting 
with the [Investment Funnel](https://github.com/VanekPetr/investment-funnel) open-source Python project. You can find 
more information about the project in the project's [README file](https://github.com/VanekPetr/investment-funnel/blob/main/README.md).

In the following lines, you will find brief instructions on how to get started with the project. If you have any 
questions, please feel free to contact me at my email address `petrr.vanekk@gmail.com`. Additionally, during the 
second week of the course, I will provide an introductory lecture on Git and the Investment Funnel, where you will 
have the opportunity to ask questions as well.

## Prerequisites

To be able to run the Investment Funnel Dashboard, you need to follow these steps.

### [I] Install Python

The project is written in Python, therefore you need to install Python on your computer. It is recommended to install 
Python 3.9, but versions 3.8 or 3.10 should work as well. You can find a great tutorial on how to install Python on
your computer on the [Real Python](https://realpython.com/installing-python/) website.

### [II] Download the project
There are two ways to download the code:

1. **[Recommended]** Install [Git](https://git-scm.com/) and clone the project repository to your computer. A great
tutorial on how to install Git can be found [here](https://github.com/git-guides/install-git). To clone the repository 
using HTTPS, you can follow this [tutorial](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository). 
If you haven't done so already, I would also recommend creating a GitHub account. It is free and you can use it to 
store your code projects (for example, the one you will write during this course). This option is recommended because during 
the course, the Investment Funnel repository will change, and you will be able to easily update your local copy of 
the repository using the `git pull` command.

2. Download the repository as a ZIP file. You can do this by clicking on the green button labeled `Code` and then selecting `Download ZIP`.
<p>
  <img width="100%" src="images/download_zip.png"></a>
</p>

### [III] In the terminal, navigate to the project folder
For Mac/Linux, use
```bash
cd path/to/the/investment-funnel
```
For Windows, use
```shell
cd path\to\the\investment-funnel
```

### [IV] In the project folder, create and activate a python virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### [V] Install the project dependencies
```bash
pip install -r requirements.txt
```

### [VI] Run the dashboard
Now you should be ready to run the code. To do that, run the following command in the terminal
```bash
python app.py 
```
The app is running on your local host http://127.0.0.1:8050. You can open it in your browser and see the dashboard 
attached in the picture below.
<p>
  <img width="100%" src="images/dash.png"></a>
</p>

### [non-compulsory] Install a code editor

To efficiently study the code, edit it, and write your own code, you need to install a code editor. I would recommend
PyCharm, which is my favorite code editor. However, you can use any other editor that you prefer. You can find a great 
tutorial on how to install PyCharm [here](https://www.jetbrains.com/help/pycharm/installation-guide.html). The free 
PyCharm Community Edition is available to download from [here](https://www.jetbrains.com/pycharm/download/)
(scroll down on the page).