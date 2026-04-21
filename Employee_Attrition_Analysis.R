# ==============================================================================
# Project: Predictive Analysis of Employee Attrition
# Author: Student
# Description: This script generates a synthetic HR dataset, performs EDA,
#              and trains a Logistic Regression model to predict attrition.
# ==============================================================================

# 1. Install required packages if not present
packages <- c("dplyr", "tidyr", "ggplot2", "caret", "e1071")
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages], repos = "https://cran.r-project.org/")
}

library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)

# ------------------------------------------------------------------------------
# 2. Data Generation
# ------------------------------------------------------------------------------
cat("Generating Synthetic HR Dataset...\n")

set.seed(42) # For reproducibility
n <- 1000

# Generating features
Age <- round(rnorm(n, mean = 38, sd = 10))
Age <- pmax(18, pmin(Age, 65)) # Bound ages between 18 and 65

Department <- sample(c("Sales", "Research & Development", "Human Resources"), n, replace = TRUE, prob = c(0.3, 0.6, 0.1))
JobSatisfaction <- sample(1:4, n, replace = TRUE, prob = c(0.15, 0.2, 0.35, 0.3))
OverTime <- sample(c("Yes", "No"), n, replace = TRUE, prob = c(0.28, 0.72))

# Income depends on department to make it realistic
MonthlyIncome <- ifelse(Department == "Sales", rnorm(n, 5500, 1500),
                 ifelse(Department == "Research & Development", rnorm(n, 6500, 2000),
                        rnorm(n, 4800, 1200)))
MonthlyIncome <- round(abs(MonthlyIncome)) + 2000

# Years at company bounded by working age
working_years <- Age - 18
YearsAtCompany <- round(runif(n, min = 0, max = working_years))

# Simulating Attrition based on a logistic function
# Formula weights (arbitrary but realistic relationships)
logit_score <- 1.5 - 0.04 * Age + 1.8 * (OverTime == "Yes") - 0.6 * JobSatisfaction - 0.00015 * MonthlyIncome + 0.05 * YearsAtCompany
logit_score <- logit_score + rnorm(n, 0, 1) # add some random noise
prob_attrition <- 1 / (1 + exp(-logit_score))

Attrition <- ifelse(prob_attrition > 0.50, "Yes", "No")

# Combine into a dataframe
hr_data <- data.frame(
  Age, Department, JobSatisfaction, OverTime, 
  MonthlyIncome, YearsAtCompany, Attrition
)

# ------------------------------------------------------------------------------
# 3. Data Cleaning & Preprocessing
# ------------------------------------------------------------------------------
cat("Cleaning and Structuring Data...\n")

# Check for missing values
cat("Total Missing Values: ", sum(is.na(hr_data)), "\n")

# Convert categorical variables to factors
hr_data <- hr_data %>%
  mutate(
    Department = as.factor(Department),
    JobSatisfaction = as.factor(JobSatisfaction),
    OverTime = as.factor(OverTime),
    Attrition = as.factor(Attrition)
  )

# Display summary
summary(hr_data)

# ------------------------------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
cat("Running Exploratory Data Analysis (EDA)...\n")

# 4.1 Attrition Distribution
plot_attrition <- ggplot(hr_data, aes(x = Attrition, fill = Attrition)) +
  geom_bar() +
  scale_fill_manual(values = c("No" = "#4CAF50", "Yes" = "#F44336")) +
  labs(title = "Distribution of Employee Attrition", x = "Attrition", y = "Count") +
  theme_minimal()
print(plot_attrition)

# 4.2 Age vs Attrition
plot_age <- ggplot(hr_data, aes(x = Attrition, y = Age, fill = Attrition)) +
  geom_boxplot() +
  labs(title = "Employee Age vs Attrition", x = "Attrition", y = "Age") +
  theme_light()
print(plot_age)

# 4.3 OverTime vs Attrition
plot_overtime <- ggplot(hr_data, aes(x = OverTime, fill = Attrition)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Proportion of Attrition by OverTime", x = "OverTime", y = "Percentage") +
  theme_minimal()
print(plot_overtime)

# 4.4 Monthly Income Distribution
plot_income <- ggplot(hr_data, aes(x = MonthlyIncome, fill = Attrition)) +
  geom_density(alpha = 0.5) +
  labs(title = "Monthly Income Density by Attrition", x = "Monthly Income", y = "Density") +
  theme_minimal()
print(plot_income)

cat("EDA completed. Proceeding to modeling...\n")

# ------------------------------------------------------------------------------
# 5. Machine Learning Model (Logistic Regression)
# ------------------------------------------------------------------------------
# Set seed for reproducibility
set.seed(123)

# Split data: 70% Train, 30% Test
trainIndex <- createDataPartition(hr_data$Attrition, p = 0.7, list = FALSE, times = 1)
data_train <- hr_data[trainIndex, ]
data_test  <- hr_data[-trainIndex, ]

cat("Training Logistic Regression Model...\n")

# Build Logistic Regression Model
# Using family="binomial" for binary classification
logit_model <- glm(Attrition ~ Age + Department + JobSatisfaction + OverTime + MonthlyIncome + YearsAtCompany, 
                   data = data_train, 
                   family = binomial)

# Model summary
print(summary(logit_model))

# ------------------------------------------------------------------------------
# 6. Model Evaluation
# ------------------------------------------------------------------------------
cat("Evaluating Model on Test Data...\n")

# Make predictions on test set (probabilities)
pred_prob <- predict(logit_model, newdata = data_test, type = "response")

# Convert probabilities to class labels (threshold = 0.5)
pred_class <- ifelse(pred_prob > 0.5, "Yes", "No")
pred_class <- as.factor(pred_class)

# Ensure levels match
levels(pred_class) <- levels(data_test$Attrition)

# Confusion Matrix
conf_matrix <- confusionMatrix(pred_class, data_test$Attrition)
print(conf_matrix)

# Print final accuracy
accuracy <- conf_matrix$overall['Accuracy']
cat(sprintf("\n=> Final Model Accuracy on Test Set: %.2f%%\n", accuracy * 100))

# ------------------------------------------------------------------------------
# End of Script
# ------------------------------------------------------------------------------
