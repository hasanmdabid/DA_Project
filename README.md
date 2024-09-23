# DA_PROJECT: Data Augmentation with Conditional GANs

Welcome to **DA_PROJECT**, a comprehensive project designed for enhancing datasets through data augmentation techniques using Conditional Generative Adversarial Networks (cGANs). This project includes the implementation of cGANs for five datasets: **OPPORTUNITY**, **DEAP**, **HAR**, **BVDB**, and **PMDB**, offering a powerful tool for improving machine learning models by augmenting training data. The results and the findings of the project are submitted to "Plos-one" Journal for pubication. Currently the submission of the paper is going through the peer review process. The details of the project installation and instruction are given here.  

## Datasets
In this project we have used 5 datasets to present the efficacy of data augmentation on werabale sensor data. The shoer description of the selected da are given below: 
- OPPORTUNITY: Wearable sensor dataset for activity recognition.
- DEAP: Dataset for emotion analysis using physiological signals.
- HAR: Human activity recognition dataset.
- BVDB & PMDB: Biomedical signal datasets (BioVid and PMDB databases).
Ensure that you have the datasets downloaded and placed in the respective folders before running the scripts.

## Project Structure

The project is organized into four main directories:
- **Opportunity/**: Contains the data and scripts related to the OPPORTUNITY dataset.
- **Deap/**: Contains the data and scripts for the DEAP dataset.
- **HAR/**: Contains the data and scripts for the HAR dataset.
- **BioVid/**: Contains the data and scripts for the BVDB and PMDB datasets.

Each directory has a subfolder for the respective conditional GAN model:
- **oppo_cGAN/**: Located in the `Opportunity/` folder.
- **cDEAPGAN/**: Located in the `Deap/` folder.
- **cHARGAN/**: Located in the `HAR/` folder.
- **cPAINGAN/**: Located in the `BioVid/` folder.

## Instructions for Using the Project

### Step 1: Train the Conditional GAN Models
For each dataset, the conditional GAN model is already implemented in its respective subfolder. To train the models, follow the instructions inside the subfolders.

1. **Navigate to the relevant folder**:
   - For OPPORTUNITY: `Opportunity/oppo_cGAN/`
   - For DEAP: `Deap/cDEAPGAN/`
   - For HAR: `HAR/cHARGAN/`
   - For BVDB and PMDB: `BioVid/cAINGAN/`

2. **Run the Conditional GAN scripts** as per the instructions given in the respective subfolders.  

3. Save the generated data in the folder path:
./<DatasetName>/cGAN/Data/cGAN_Generated_data/

## Step 2: Evaluate the Data Augmentation Techniques
Each folder contains a main.py script that evaluates the proposed data augmentation techniques, including the cGAN-generated data. To evaluate the performance:
 1. Run the main.py script for each dataset. Example for OPPORTUNITY dataset:
 ```bash
 cd Opportunity/
 python main.py
 ```
 2. Classification results will be automatically saved in the respective folder path:
./<DatasetName>/results/

## Step 3: Analyze the Results
The results folder (./<DatasetName>/results/) will contain the evaluation metrics such as accuracy, precision, recall, etc. Analyze these results to assess the effectiveness of the data augmentation techniques for your specific use case.

## Prerequisites

You can install all dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Contribution
Feel free to open issues or submit pull requests to improve this project. Let's enhance the world of data augmentation together!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

You can directly copy this code into the `README.md` file of your GitHub repository. Let me know if you'd like any more changes!
