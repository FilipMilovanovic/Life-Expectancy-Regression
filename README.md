# Life Expectancy Prediction

This project explores **regression** as a form of supervised machine learning, using **R** to predict **life expectancy** based on real-world data.

---

## 📂 Project Structure
- `data/` – contains the dataset (`life_expectancy.csv`)  
- `linear_regression/` – Linear Regression implementation  
Each folder includes a single R script that handles the complete pipeline.

---

## ⚙️ Process Performed in the Script
- Data Preparation and Cleaning  
- Exploratory Data Analysis (EDA)  
- Model Training  
- Model Evaluation  
- Diagnostic Plots and Interpretation  

---

## 📈 Dataset
- **Source:** [Kaggle – Life Expectancy (WHO dataset)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/data)  
- **Format:** CSV  
- **Location:** `data/life_expectancy.csv`  

---

## 🧪 How to Run  
Open a desired .R script in **RStudio** and run the code.
Make sure the dataset file is located in the same directory as the script.
If the dataset is in a different location, update the path accordingly.

Example of reading the dataset from the script:

```r
data <- read.csv("life_expectancy.csv", stringsAsFactors = FALSE)
