# Explainable Artificial Intelligence (XAI) in Pain Research:Understanding the Role of Electrodermal Activity for Automated Pain Recognition

## Description
This is the source code of the paper '**Explainable Artificial Intelligence (XAI) in Pain Research: Understanding the Role of Electrodermal Activity for Automated Pain Recognition**' published in _Sensors_ as part of the Special Issue 'Explainable AI in Medical Sensors'. The paper can be found [here](https://www.mdpi.com/1424-8220/23/4/1959).

## How to use this repository
- Clone the project
    ```bash 
    git clone https://github.com/gouverneurp/XAIinPainResearch.git
    ```
- Install python (tested under 'Python 3.9.7').
- Create a python environment and activate it. Windows:
    ```bash
    python -m venv venv
    \venv\Scripts\activate
    ```
- Install the requirements
    ```bash 
    pip install -r requirements.txt
    ```
- Place the datasaets in the directories under "datasets" following the description in the readme files.
- Run "create_np_files.py" to create the Numpy dataset.
- Run "hcf.py" to create hand-crafted features (HCF).
- Run "main.py" or other scripts to evaluate the implemented methods.

## Please cite our paper if you use our code.
```bibtex
@article{gouverneur2023explainable,
    AUTHOR = {Gouverneur, Philip and Li, Frédéric and Shirahama, Kimiaki and Luebke, Luisa and Adamczyk, Wacław M. and Szikszay, Tibor M. and Luedtke, Kerstin and Grzegorzek, Marcin},
    TITLE = {Explainable Artificial Intelligence (XAI) in Pain Research: Understanding the Role of Electrodermal Activity for Automated Pain Recognition},
    JOURNAL = {Sensors},
    VOLUME = {23},
    YEAR = {2023},
    NUMBER = {4},
    ARTICLE-NUMBER = {1959},
    URL = {https://www.mdpi.com/1424-8220/23/4/1959},
    ISSN = {1424-8220},
    DOI = {10.3390/s23041959}
}
```