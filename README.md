# Predictive Analysis of Employee Attrition

## Project Overview
This project focuses on the predictive analysis of employee attrition using R. It encompasses the end-to-end data science lifecycle, from generating a realistic synthetic HR dataset to performing Exploratory Data Analysis (EDA) and training a machine learning model to predict whether an employee is likely to leave the company.

## Key Features
1. **Synthetic Data Generation**: Creates a robust simulated HR dataset with features like Age, Department, Job Satisfaction, OverTime, Monthly Income, and Years at Company to model realistic employee scenarios.
2. **Data Cleaning & Preprocessing**: Handles missing values and structures categorical variables for optimal analysis and model performance.
3. **Exploratory Data Analysis (EDA)**: Utilizes `ggplot2` to generate insightful visualizations, including:
   - Overall distribution of employee attrition.
   - Relationship between Age and Attrition.
   - Impact of OverTime on Attrition probabilities.
   - Monthly Income density mapped against Attrition.
4. **Predictive Modeling**: Implements a Logistic Regression model (binomial classification) to predict employee attrition based on the generated features.
5. **Model Evaluation**: Evaluates the model's performance on a dedicated test dataset (70/30 train-test split) using a Confusion Matrix and accuracy metrics.

## Technologies Used
- **Language**: R
- **Libraries/Packages**: 
  - `dplyr`, `tidyr` (Data Manipulation)
  - `ggplot2` (Data Visualization)
  - `caret`, `e1071` (Machine Learning & Evaluation)

## Project Structure
- `Employee_Attrition_Analysis.R`: The core R script containing the code for data generation, EDA, modeling, and evaluation.
- `Employee_Attrition_Project.Rmd`: R Markdown document for generating a comprehensive and structured analytical report.
- `Rplots.pdf`: Exported plots generated during the Exploratory Data Analysis phase.

## How to Run
1. Ensure R and RStudio are installed on your system.
2. Open `Employee_Attrition_Analysis.R`.
3. The script will automatically install missing dependencies (like `ggplot2` and `caret`) and run the complete analysis, outputting the dataset summary, plots, model statistics, and the final accuracy score.
