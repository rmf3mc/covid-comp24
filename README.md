
# EfficientNet-SAM: A Novel EfficientNet with Spatial Attention Mechanism for COVID-19 Detection in Pulmonary CT Scans

## Overview

Repository for the paper titled **"EfficientNet-SAM: A Novel EfficientNet with Spatial Attention Mechanism for COVID-19 Detection in Pulmonary CT Scans,"** presented at CVPR 2024. This project leverages the power of EfficientNet architecture enhanced with a Spatial Attention Mechanism (SAM) to detect COVID-19 from pulmonary CT scans.

## Dependencies

Before running any code, ensure that you have all the necessary dependencies installed. You can find the required packages listed in `requirements.txt`. To install them, use the following command:

```bash
pip install -r requirements.txt
```

## Getting Started

This section for setting up the environment and running the implementation in this repository:

1. **Clone the Repository**: Start by cloning this repository to your local machine:

   ```bash
   git clone https://github.com/rmf3mc/covid-comp24
   cd EfficientNet-SAM
   ```

2. **Install Dependencies**: As mentioned above, install the dependencies listed in `requirements.txt`.

## Running the Project

### Data Preparation

The first step involves preparing your data. We provide notebooks for selecting and filtering slices from pulmonary CT scans. Execute these notebooks in sequence:

1. **Slice Selection**:
   Open and run the following notebook to select relevant slices:
   
   
   jupyter notebook step0/a-SliceSelection.ipynb
   

2. **Filtering**:
   After selecting slices, filter the data using the next notebook:
   
   jupyter notebook step0/b-Filtering.ipynb
   

### Model Training

Once the data is prepared, proceed to train the model. The training process is managed through the following notebook:

1. **Training**:
   Open and run the training notebook to start the model training process:
   
   jupyter notebook Training/Training.ipynb
   


## Credits

This project builds upon the foundational work by Chih-Chung Hsu in [SBBT_COV19D_2023](https://github.com/jesse1029/SBBT_COV19D_2023). We express our gratitude for their contributions which significantly inspired and facilitated the development of this project.
