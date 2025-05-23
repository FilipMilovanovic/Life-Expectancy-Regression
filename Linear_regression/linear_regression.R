# Loading the dataset
data <- read.csv("../data/Life Expectancy Data.csv",stringsAsFactors = F)

summary(data)

# Checking for missing values

# Variables Alcohol, Hepatitis.B, BMI, Polio, Total.expenditure, Diphtheria, GDP, Population, thinness..1.19.years, thinness.5.9.years, Income.composition.of.resources,Schooling contain NA values
apply(data,2,function(x) sum(is.na(x)))
apply(data,2,function(x) sum(x == "",na.rm = T))
apply(data,2,function(x) sum(x == "-",na.rm = T))
apply(data,2,function(x) sum(x == " ",na.rm = T))
apply(data,2,function(x) sum(x == "N/A",na.rm = T))


# Only complete cases for the dependent variable are used
data <- data[complete.cases(data$Life.expectancy),]

length(unique(data$Country))
# Categorical variable Country has too many different values and is not suitable for linear regression
data$Country <- NULL

length(unique(data$Status))
data$Status <- ifelse(data$Status == "Developed", 1, 0)
table(data$Status)

# Checking distribution using Shapiro test for variables with missing values
apply(data,2,function(x) sum(is.na(x)))
apply(data[,c(6,8,10,12,13,14,16,17,18,19,20,21)],2, shapiro.test)

# None of them follow normal distribution, we impute missing values with median
medianAlcohol <- median(data$Alcohol,na.rm = T)
data$Alcohol[is.na(data$Alcohol)] <- medianAlcohol

medianHepatitis.B <- median(data$Hepatitis.B,na.rm = T)
data$Hepatitis.B[is.na(data$Hepatitis.B)] <- medianHepatitis.B

medianBMI <- median(data$BMI,na.rm = T)
data$BMI[is.na(data$BMI)] <- medianBMI

medianPolio <- median(data$Polio,na.rm = T)
data$Polio[is.na(data$Polio)] <- medianPolio

medianTOtal.expenditure <- median(data$Total.expenditure,na.rm=T)
data$Total.expenditure[is.na(data$Total.expenditure)] <- medianTOtal.expenditure

medianDiphtheria <- median(data$Diphtheria,na.rm=T)
data$Diphtheria[is.na(data$Diphtheria)] <- medianDiphtheria

medianGDP <- median(data$GDP,na.rm = T)
data$GDP[is.na(data$GDP)] <- medianGDP

medianPopulation <- median(data$Population,na.rm=T)
data$Population[is.na(data$Population)] <- medianPopulation

medianthinness1 <- median(data$thinness..1.19.years,na.rm = T)
data$thinness..1.19.years[is.na(data$thinness..1.19.years)] <- medianthinness1

medianthinness5 <- median(data$thinness.5.9.years,na.rm=T)
data$thinness.5.9.years[is.na(data$thinness.5.9.years)] <- medianthinness5

medianIncome <- median(data$Income.composition.of.resources,na.rm=T)
data$Income.composition.of.resources[is.na(data$Income.composition.of.resources)] <- medianIncome

medianSchooling <- median(data$Schooling,na.rm=T)
data$Schooling[is.na(data$Schooling)] <- medianSchooling

all(complete.cases(data))


# Creating correlation matrix
matrica <- cor(data)
library(corrplot)
corrplot(matrica, method = "number", type ="upper",diag = F)

# We include in the model only variables highly correlated with the outcome variable.
# Selecting variables with correlation > 0.6 with Life.expectancy
sort(abs(cor(data)[, "Life.expectancy"]), decreasing = TRUE)

# In every machine learning task, the dataset is divided into two subsets: one for training (model development) and another for evaluation (testing) to assess the model’s performance. A model is always built exclusively using the training data, while its effectiveness is measured by computing metrics on the test set.
# We use caret because when splitting data into training and testing sets, it is essential to ensure that observations are randomly assigned to each subset. This prevents any bias from influencing the distribution and maintains fairness. The selection should be completely random.
#  However, a certain pattern may emerge if data is collected over successive time intervals (early observations might behave differently compared to later ones). Therefore, when splitting a dataset, the distribution of values in the training set should match the distribution in the test set. This is crucial because if a model learns to predict numerical values based on one distribution but encounters test data that follows a different distribution, its predictions may be inaccurate.
# Additionally, a seed is set to ensure reproducibility. By setting a seed value, we guarantee that every time the program is run, the same results are obtained, allowing for consistent experimentation and validation.

library(caret)
set.seed(1) 
indexes <- createDataPartition(data$Life.expectancy, p = 0.8, list = FALSE)
train_data <- data[indexes, ]
test_data <- data[-indexes, ]

summary(train_data$Life.expectancy)
summary(test_data$Life.expectancy)

#pravimo prvi model sa varijablama koje imaju korelisanost >0.6
lm1 <- lm(Life.expectancy ~ Schooling + Adult.Mortality + Income.composition.of.resources, data = train_data)

summary(lm1)
# Interpreting the initial regression model:

# The F-statistic serves as a fundamental criterion for assessing the viability of a given model. This statistic is used to evaluate a statistical test, wherein the null hypothesis assumes that none of the variables within the model are significant predictors of the dependent variable, meaning that all coefficients are equal to zero. If the test yields a very low probability, we reject this null hypothesis and deem the model valid. The probability associated with this assessment is given by the p-value. Since the observed p-value is sufficiently small, it is reasonable to consider this model a viable candidate for further analysis.

# R-squared is an indicator that quantifies the proportion of variability in the dependent variable explained by the model. The objective is to account for as much variability as possible. In this case, the R-squared value is 0.7148. This metric is particularly useful when dealing with a single independent variable.

# Adjusted R-squared becomes relevant when multiple independent variables are included in the model. It represents a modified R-squared value, addressing the fact that R-squared tends to increase as more variables are added, regardless of their actual contribution to explaining the dependent variable. Adjusted R-squared penalizes models that incorporate additional predictors which do not improve explanatory power. If Adjusted R-squared is approximately equal to R-squared, it suggests that the additional variables do not degrade the model's integrity.

# The coefficients are considered statistically significant. The Estimate represents the estimated coefficient for each variable, indicating the relationship with the target variable (whether positive or negative). The T-statistic serves as a key metric in hypothesis testing, where the null hypothesis assumes that the observed variable has no correlation with the dependent variable. If the coefficient were zero, the predictor and the outcome variable would be entirely independent. A very low probability leads to the rejection of this null hypothesis, indicating that the variable is indeed significant for the model. The presence of more asterisks suggests stronger statistical significance. Std. Error represents the standard error of the estimated coefficient.

# Residuals represent the differences between the actual and predicted values produced by the model. Ideally, they should be randomly distributed around zero. In linear regression, it is expected that the average residual is close to zero and that residuals follow an approximately normal distribution.

# The intercept represents the predicted value of the response variable when all other predictors are zero. While this scenario is often unrealistic in practice, it provides a reference point for the model. A one-unit increase in the Schooling variable is associated with an increase of approximately 1.0366 in the response variable, assuming all other variables remain constant.

# The coefficients are estimated by the linear regression algorithm. To make a prediction for a new observation, the model multiplies each predictor value by its corresponding coefficient, sums the results, and adds the intercept to compute the predicted outcome.


coef(lm1) 
# Function for extracting model coefficients

confint(lm1,level = 0.95) 

# The coefficients are estimated based on a given sample of data. However, since we're working with a sample and not the entire population, there's always some uncertainty around the exact values of these estimates. This is where confidence intervals come in.
 
# A 95% confidence interval gives us a range in which the true value of the coefficient is likely to lie. Specifically, if we were to repeat the sampling process 100 times, we would expect the true coefficient to fall within this interval in approximately 95 out of those 100 samples.


# Fitting a model using all available predictors
lm2 <- lm(Life.expectancy ~. ,data = train_data)
summary(lm2)
# To improve model interpretability and reduce potential overfitting, we retain only the predictors that are statistically significant at the 1% level (p < 0.01), indicated by ** or *** in the summary output.

lm3 <- lm(Life.expectancy ~ Status +
            Adult.Mortality +
            infant.deaths +
            Hepatitis.B +
            BMI +
            under.five.deaths +
            Polio +
            Diphtheria +
            HIV.AIDS +
            GDP +
            Income.composition.of.resources +
            Schooling,
          data = train_data)

summary(lm3)

# Before finalizing the model, it's essential to check for multicollinearity among predictors. Multicollinearity occurs when two or more independent variables are highly correlated (either positively or negatively). Including highly collinear variables can distort the coefficient estimates, leading to unreliable interpretations

library(car)
# Checking for multicollinearity using the Variance Inflation Factor (VIF)
sort(sqrt(vif(lm3)))


# The output revealed two potentially problematic variables: infant.deaths (VIF = 12.641) and under.five.deaths (VIF = 12.697). We removed them one at a time and re-evaluated the VIFs to improve model stability.



lm4 <- lm(Life.expectancy ~ Status +
            Adult.Mortality +
            infant.deaths +
            Hepatitis.B +
            BMI +
            Polio +
            Diphtheria +
            HIV.AIDS +
            GDP +
            Income.composition.of.resources +
            Schooling,
          data = train_data)

summary(lm4)
sort(sqrt(vif(lm4)))
#The multicollinearity issue was resolved

# Interpreting the final regression model

# The F-statistic indicates that the model is statistically significant overall
# The R-squared value is 0.8082 and the Adjusted R-squared is 0.8073, suggesting that around 81% of the variance in life expectancy is explained by the predictors.
# Most of the variables included in the model are statistically significant and contribute meaningfully to predicting the target variable

# To validate the assumptions of linear regression, I generated the standard set of diagnostic plots

par(mfrow=c(2,2)) 
plot(lm4)


# Residuals vs Fitted Plot

# This plot checks the assumption of linearity. The x-axis represents the fitted (predicted) values, while the y-axis displays the residuals. Ideally, residuals should hover around zero. On the plot, we see a dashed horizontal line at zero, which represents this ideal state. While we can't expect every point to lie exactly on this line, we aim for the residuals to be symmetrically scattered above and below it and to remain relatively close. The red smoother line reflects the trend in residuals. In this case, it is nearly flat, suggesting that the linearity assumption is reasonably satisfied. This supports the idea that a linear relationship exists between the predictors and the target variable 

# Normal Q-Q Plot

# Plot helps us assess whether the residuals are normally distributed, one of the key assumptions in linear regression. If the residuals follow a normal distribution, the points should closely follow the diagonal line. In this case, most of the residuals align well with the line, suggesting that the normality assumption is reasonably satisfied.


# Scale - Location
#
# Checking the assumption of homoscedasticity, which refers to the uniform spread of residuals.
# Ideally, the residuals should be randomly scattered around the horizontal line,
# without forming any clear pattern or structure. This randomness indicates that the variance 
# of the residuals is constant across all levels of the predicted values.
# In this plot, we look for a cloud of points spread evenly, not funnel-shaped or curved
# which would suggest a violation of this assumption.

# Residuals vs Leverage

# Identifies observations that may have unusual values or combinations of predictor variables, which could potentially influence the regression model. While a few observations slightly exceed the Cook’s distance threshold, they do not exert strong enough influence to distort the model’s coefficient estimates. Therefore, no immediate action is required.


lm4.pred <- predict(lm4, newdata = test_data)
test_data$Life.expectancy_pred <- lm4.pred


library(ggplot2)
ggplot(test_data) +
  geom_density(aes(x = Life.expectancy, color = 'actual')) +
  geom_density(aes(x = Life.expectancy_pred, color = 'predicted'))

# Evaluating regression model performance using standard metrics

# Residual Sum of Squares (RSS) measures the total squared differences between predicted and actual values.
# A lower RSS indicates a better fit, meaning the model's predictions are closer to the true values.
RSS_lm4 <- sum((lm4.pred - test_data$Life.expectancy)^2)

# Total Sum of Squares (TSS) represents the total variation in the actual life expectancy values.
# It shows how far the real values are, on average, from the overall mean

TSS <- sum((mean(train_data$Life.expectancy) - test_data$Life.expectancy)^2)

# R-squared represents the proportion of variance in the dependent variable that is explained by the model.
R4 <- 1-RSS_lm4 / TSS

R4
summary(lm4)

# Root Mean Square Error (RMSE) measures the average magnitude of prediction errors.
# It is in the same unit as the dependent variable and easier to interpret.
RMSE_lm4 <- sqrt(RSS_lm4 / nrow(test_data))
RMSE_lm4 

RMSE_lm4 / mean(test_data$Life.expectancy)

# The RMSE is approximately 4.11 years, which represents only about 5.9% of the mean life expectancy in the test set. 
# This indicates that the model performs quite well, with relatively small prediction errors in the context of the target variable's scale.