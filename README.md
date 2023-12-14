# KY-Public-School-Budget-Analysis
## Introduction
Within the intricate realm of public education finance, it is vital to comprehend the nuanced elements that impact budget allocations for effective resource management. Chapter 13 in Data Science for Business and Decision Making introduces Multiple Regression Models, offering a means to explore the interconnections among different variables. This study endeavors to employ Multiple Regression Analysis to anticipate education budget allocations based on a specified set of predictor variables (Favero & Belfiore, 2019).
## Background
The landscape of public education finance is complex and demands a thorough comprehension of the factors that shape budget allocations. In recent times, the discourse around school finance has gained prominence, with numerous governors addressing the issue in their State of the State addresses. Additionally, teacher strikes have drawn attention to finance matters concerning salaries, class sizes, and resource allocation. The implementation of the Every Student Succeeds Act (ESSA) further amplifies the significance of discussing school-by-school spending data (Rosa, 2022). 
Given the heightened attention to school finance, there is a pressing need to bridge knowledge gaps among education leaders, teachers, parents, and the broader public. A study conducted by Georgetown University's Edunomics Lab and research associates delves into focus group and one-on-one interviews, revealing substantial gaps in understanding how schools are funded. This lack of comprehensive knowledge underscores the necessity for effective communication about education finance (Rosa, 2022). 
## Problem Statement
The primary objective of this investigation is to gain a profound understanding of how student enrollment, socio-economic status, and teacher-student ratios in Kentucky's school districts collectively influence the annual district budget. This study is guided by the following questions: How do various factors, such as student enrollment, socio-economic status, and teacher-student ratios, collaboratively impact education budget allocations? Furthermore, what is the specific scenario that is happening in Kentucky?
To answer these questions, I have formulated the following hypotheses:
Null Hypothesis: There is no substantial linear relationship between the set of independent variables and Education Budget.
Alternative Hypothesis: At least one independent variable exhibits a significant linear relationship with Education Budget Allocations.
Through an examination of these hypotheses, I seek to understand how student enrollment, socio-economic status, and teacher-student ratios for districts in KY affect the amount given for the annual district budget.
## Literature Review
The literature pertinent to this study explores key concepts related to multicollinearity, data science applications in decision-making, public school district finance data, regression analysis, and effective communication in the realm of school finance.
Bhandari (2023) discusses the phenomenon of multicollinearity, emphasizing its causes, effects, and detection methods using the Variance Inflation Factor (VIF). Multicollinearity, as explained, poses challenges in regression analysis, impacting the reliability of coefficients. Understanding these nuances is crucial for robust statistical modeling in the context of education budget predictions (Bhandari, 2023).
Favero and Belfiore (2019) contribute to the literature by presenting insights from "Data Science for Business and Decision Making" (Favero & Belfiore, 2019). This source is a valuable reference for understanding how data science principles can be applied to inform decision-making processes, providing a theoretical foundation for the study on education budget allocations. As a textbook, though, it is limited in its broad scope, meaning other sources will be required to develop a more complete understanding of the topic.
The National Center for Education Statistics (2019) serves as a primary data source for the study. This repository offers comprehensive public school district finance data, allowing us to delve into the intricacies of financial patterns, disparities, and trends across various districts (National Center for Education Statistics, 2019). The challenge, though, is accessing the direct data: The system requires manually searching and pulling data from individual districts, rather than offering a complete downloadable set containing all information, thus adding a significant amount of time and effort for carrying out this investigation. This will be addressed in greater detail later in this paper.
Perktold (2023) provides documentation on the statsmodels library, specifically focusing on the OLS (Ordinary Least Squares) regression. This technical resource is instrumental in the methodological approach, guiding the implementation of regression analysis for predicting education budget allocations (Perktold, 2023).
Roza (2022) addresses the vital aspect of effective communication in school finance. As I navigate through statistical analyses, this source provides an important reminder about the need of clear communication to engage stakeholders, foster transparency, and build trust in the context of education budget discussions (Roza, 2022). 
Collectively, these sources synthesize key concepts, methodologies, and insights regarding data science and effective communication of results. They lay the groundwork for the study on education budget allocations, emphasizing the interdisciplinary nature of the research and the need for a holistic understanding of statistical techniques, data sources, and effective communication strategies. At the same time, these sources highlight the lack of formal data analysis and resulting discourse in the field of education. Thus, the aim of this study is to begin filling that gap and spark further discussion of the utility of data science in this sector.
### Presentation of the Data
In my pursuit of data for the study on education budget allocations in Kentucky, I encountered challenges in obtaining comprehensive information on student enrollment, socio-economic status, and teacher-student ratios across all districts. Initially, I faced the obstacle of a lack of readily available datasets specific to my variables of interest. To address this issue, I turned to the National Center for Education Statistics (NCES) as a reliable source for education-related data. However, the nature of collecting data for all 187 districts in Kentucky posed practical difficulties. To overcome this, I created a random sample comprising 30 districts. I selected random sampling in this context as it provides a representative subset of the entire population, allowing for generalizability of findings to the broader set of districts in Kentucky. This method ensures that the selected districts are not biased, providing a more accurate and efficient approach to data collection and analysis (Favero & Belfiore, 2019). 
After obtaining a random sample of 30 districts, I began to explore the data. I first began by importing packages that I would need for my analysis: 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import statsmodels.api as sm



Next, using the pandas package, I loaded my data:

```df = pd.read_csv('/Users/samaguiar/Desktop/university-of-the-cumberlands/MSDS_530/week6_assignments/ky_districts_19_20_subset.csv')```


Next, I displayed the first 5 rows of my data set to get an overview:
```df.head()```


This output highlights the attributes shown in the data set. For this study, I will be focusing on the following attributes: Total Revenue, number of students, Student Teacher Ratio, and % of families below poverty level. I further investigated the dataframe by looking at the classification of each attribute: 

```df.info()``


In this output, I notice two attribute types that will cause issues in my data analysis. Both Total Revenue and number of students are considered objects, when I need these attributes to be numerical. However, there are no missing values in the data set, as seen by the 30 non-null output. This means that I will not have to clean the data set to remove missing values. 
To deal with the incorrect type of the two attributes, I worked to transform this data into numerical variables. First, I removed the commas in Total Revenue as this was likely the cause of the object classification. Next, I converted the attribute to numeric values using the pandas package:

```df['Total Revenue'] = pd.to_numeric(df['Total Revenue'].replace(',', '', regex=True), errors='coerce')```


In this code, I also added errors='coerce' to ensure that any errors would be transformed into NaN values, instead of breaking the code block. I used a similar process for the attribute number of students :
```df['number of students'] = pd.to_numeric(df['number of students'].replace(',', '', regex=True), errors='coerce')```


Lastly, I transformed the attributes into integer values to be able to complete numerical calculations: 
```df['Total Revenue'] = df['Total Revenue'].astype(int)
df['number of students'] = df['number of students'].astype(int)```


To ensure that the attributes are now numerical, I used the following code to get a numerical summary of each attribute:
```df.describe()```



Since both attributes are included in the numerical summary, I know that the data has been transformed to a numerical value. 
I used the Seaborn library's pairplot function on the dataframe to conduct a visual examination of the relationships between variables:
```sns.pairplot(df)```


The pairplot generated a matrix of scatterplots, allowing for a comprehensive visual exploration of how each variable correlates with every other variable in the dataset (Perktold, 2022). Many of the attributes showed little to no correlation, except for the scatter plots comparing:
- Number of students vs. Total Revenue; and
- Number of schools vs. Total Revenue. 
This indicates that Total Revenue may be highly determined by the amount of students that the school district needs to serve, meaning the larger the district, the larger the budget.

## Analysis
After understanding my dataframe and attributes in consideration, I proceeded to begin my analysis. I wanted to use a linear regression model to predict the Total Revenue in an educational context. First, I defined the attributes I wanted to use to conduct my multivariable analysis:
```X = df[['number of students', 'Student Teacher Ratio', 'total number of schools', '%of families below poverty level']]
y = df['Total Revenue']```


In this, I identified my independent variables as, number of students, Student Teacher Ratio, total number of schools, %of families below poverty level, to see if any of these had a large effect on the total revenue a district received. 
I divided my dataset into training and testing sets, with 80% used for training and 20% for testing:

```X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)```


I then used the scikit-learn library's LinearRegression class to create the regression model, which is then fitted to the training data, determining the optimal coefficients for the linear equation:

```# Fit the regression model
model = LinearRegression()
model.fit(X_train, y_train)```


 I employed the model to make predictions on the test set:

```
# Make predictions on the test set
y_pred = model.predict(X_test)
```


Lastly, I assessed the performance of the model using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). This provided a comprehensive evaluation of the model's accuracy and predictive capability by comparing the predicted values to the actual values in the test set (Favero & Belfiore, 2019):

```# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred) ** 0.5)```


```Mean Absolute Error: 8902824.68742247
Mean Squared Error: 102422187306684.56
Root Mean Squared Error: 10120384.741040459
```

The errors reflected in the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics raised concerns about the linear regression model's accuracy. To address this issue, I wanted to further investigate and identify potential outliers or influential data points that could be influencing the model's performance. Outliers, being data points significantly different from the rest, can disproportionately impact the model, leading to inaccuracies (Favero & Belfiore, 2019). To do this, I created a scatter plot:
```# Visualize predictions vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Education Budget')
plt.ylabel('Predicted Education Budget')
plt.title('Scatter Plot of Predicted vs. Actual Values')

# Add a diagonal line for reference (perfect predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Predictions')

# Add legend
plt.legend()

# Show the plot
plt.show()```

 



In the scatter plot, the diagonal line serves as a reference for perfect predictions, helping assess the model's accuracy. Notably, an outlier is observed in the data, specifically in Jefferson County. The correlation between the variables appears to be strong and positive. To formally understand the correlation, I determined the R-value:

```# Print R-squared value
print('R-squared:', model.score(X_test, y_test))```

```R-squared: 0.9979827739410068```

The R-squared is high, suggesting a nearly perfect linear relationship between the predictor variables and the total revenue in the context of the education budget allocation. The presence of a strong positive correlation reinforces the model's ability to capture the underlying patterns in the data, except for the identified outlier that may warrant further investigation.
Lastly, I conducted an Ordinary Least Squares (OLS) regression analysis using  statsmodels library. The OLS model is a statistical method used to estimate the relationships between variables by minimizing the sum of the squared differences between observed and predicted values (Favero & Belfiore, 2019):

```# Add a constant term to the independent variables (X)
X_with_const = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X_with_const).fit()

# Print the model summary
print(model.summary())

# Extract p-values for hypothesis testing
p_values = model.pvalues

# Set your significance level
alpha = 0.05```


Hypothesis testing was subsequently performed on the coefficients to discern the significance of each variable in influencing the education budget allocations. This analytical approach provided a robust foundation for identifying key variables that significantly contribute to the model's predictive capabilities:
```# Perform hypothesis testing
significant_vars = p_values[p_values < alpha].index

# Print the results
print(f'Significant variables: {significant_vars}')```


The regression analysis results indicated a highly favorable fit of the model to the data, with an R-squared value of 0.999, signifying that 99.9% of the variance in Total Revenue is explained by the model. 
The F-statistic, with a significant p-value of less than 0.05, further underscores the model's overall efficacy (Favero & Belfiore, 2019). This implies that at least one independent variable significantly influences the variance in Total Revenue. The coefficients associated with each independent variable offer valuable information. The constant term's coefficient is 1.422e+07, representing the estimated Total Revenue when all independent variables are zero. The number of students stands out as a significant variable, with a coefficient of 1.498e+04 and a low p-value of 0.000, indicating a substantial impact on Total Revenue.
However, the other independent variables, namely Student Teacher Ratio, total number of schools, and % of families below the poverty level, do not exhibit statistical significance based on their p-values, suggesting limited influence on Total Revenue in this context.
The condition number of 2.09e+05 raises concerns about multicollinearity, indicating potential numerical problems in the model. It is crucial to address multicollinearity, as it can impact the reliability of coefficient estimates. The identified significant variable, number of students, aligns with the results of the hypothesis testing mentioned earlier. (Bhandari, 2019).

## Summary of Results and Next Steps
The number of students emerges as a significant factor influencing the Total Revenue allocated to each district. The significant F-statistic allows the rejection of the null hypothesis, indicating that at least one independent variable plays a substantial role in explaining the variance in the dependent variable. The presence of considerable errors in the model is notable, suggesting potential multicollinearity. To enhance the robustness of the analysis, I would want to address outliers in the dataset, consider a larger sample size for Kentucky, and expand the scope to include data for the entire United States. Additionally, experimenting with the inclusion or exclusion of variables such as the number of ML students, types of revenue, or the number of schools in the district could provide insights into how significance changes and contribute to a more comprehensive understanding of the factors influencing Total Revenue. The rejection of the null hypothesis and acceptance of the alternative hypothesis underscore the importance of the number of students as a significant predictor of Total Revenue in each district.

## Conclusion
In conclusion, this study aimed to investigate the dynamics of education budget allocations in Kentucky's school districts by employing Multiple Regression Analysis. The investigation looked into the relationships among variables such as student enrollment, socio-economic status, and teacher-student ratios, aiming to understand their collective influence on annual district budgets.
The literature review drew insights from various sources, including discussions on multicollinearity, data science applications in decision-making, public school finance data, regression analysis, and effective communication in school finance. The study identified key challenges, such as the practical difficulties in obtaining comprehensive data and the need for effective communication to bridge knowledge gaps among stakeholders.
Upon presenting the data, challenges in obtaining comprehensive information led to the use of a random sample of 30 districts. The data preparation involved addressing data type issues, transforming attributes into numerical variables, and conducting visual explorations through pairplots and scatter plots.
Analysis was conducted by using a linear regression model to predict total revenue based on selected independent variables. Performance evaluation metrics raised concerns about model accuracy, leading to a closer examination of potential outliers, particularly in Jefferson County. Despite high R-squared values indicating a strong linear relationship, caution was warranted due to the presence of outliers.
The Ordinary Least Squares (OLS) regression analysis confirmed a favorable fit of the model, with a high R-squared value of 0.999. The F-statistic supported the overall efficacy of the model, while the identified significant variable, the number of students, emerged as a key predictor of total revenue. However, concerns about multicollinearity and the non-significance of other variables indicated the need for further refinement.
The study's limitations, including sample size constraints and potential outliers, suggest avenues for future research. Recommendations include addressing multicollinearity, expanding the sample size, considering additional variables, and exploring a broader geographical scope.
This research contributes to the ongoing discourse on education finance, emphasizing the significance of the number of students in shaping total revenue allocations. As education leaders, policymakers, and stakeholders navigate the complexities of budgeting, this study underscores the importance of continuous refinement and adaptation in statistical modeling to enhance the accuracy and reliability of predictions in the ever-evolving landscape of public education finance.


## References
Bhandari, A. (2023, November 9). Multicollinearity: Causes, effects and detection using VIF (updated 2023). Analytics Vidhya. https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/  
Favero, L. P., & Belfiore, P. (2019). Data Science for Business and Decision Making. Elsevier S & T. https://reader2.yuzu.com/books/9780128112175 
National Center for Education Statistics. (2019). Public School District Finance Data. https://nces.ed.gov/edfin/search/peergroupdata.asp?dataid=1&amp;mt=0&amp;subdataid=1&amp;bleaid=2105700&amp;jobid=%7B875E2C11-077B-4720-BC73-B169C943BDFB%7D 
Perktold, J. (2023). Statsmodels.regression.linear_model.ols. statsmodels.regression.linear_model.OLS - statsmodels 0.14.0. https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html 
Roza, M. (2022, July 13). Understanding School Finance is one thing. being effective in communicating about it is another skill entirely. Learning Policy Institute. https://learningpolicyinstitute.org/blog/understanding-school-finance-one-thing-being-effective-communicating-about-it-another-skill  
