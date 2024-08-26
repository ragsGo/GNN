Subsampling graphical neural networks for genomic prediction of quantitative phenotypes
This project creates a Graphical Neural network model for genomic datasets to predict the phenotypes.

## Getting Started

These instructions will get you a copy of the project up and running on your Linux/Ubuntu machine for development and testing purposes.

### Prerequisites

The libraries and packagesyou need to install the software are listed in the requirements.txt file and how to install them is mentioned in the installing section

### Installing

Download the project -
Download the zip file from the prject homepage
Or
clone the project in your work directory by
git clone https://github.com/ragsGo/GNN.git

Follow the steps below for setup and run the project
python3 -m venv venv #install the virtual environment
source venv/bin/activate #activate the virtual environment
cd GNN #change to working directory
pip install -r requirements.txt # install the required pacakges and libraries

PYTHONPATH=. python3 gnn/main.py #run the project/

## Changing parameters

At the bottom of `gnn/main.py` you can change the parameters that the program uses
