### Predicting Monthly Vehicle Sales: Project Summary

**Ruben D. Colmenares**

#### Executive summary
This project explores the impact of various incentives on monthly vehicle sales for an automotive company operating in Mexico.
By understanding the effectiveness of different incentive programs, the company aims to optimize its sales strategies and enhance resource allocation.
The project involves the analysis of historical sales data, application of regression models, and in-depth feature engineering — including the use of SHAP values for interpretability — to provide actionable insights to both technical and non-technical stakeholders.

#### Rationale
Why should anyone care about this question?
Understanding the impact of incentives on monthly vehicle sales is crucial for the company's growth and competitiveness in the Mexican automotive market.
By leveraging machine learning techniques, the company can make informed decisions regarding which incentives to prioritize and invest in, leading to better resource allocation and increased sales revenue.

#### Research Question
What are you trying to answer?
Can classification techniques predict the influence of specific incentives on monthly vehicle sales in an automotive company based in Mexico?

#### Data Sources
The project sources historical data from the automotive company, encompassing information on monthly vehicle sales, incentive programs offered, pricing data, 
and other relevant factors affecting sales performance.What data will you use to answer you question?

#### Methodology
What methods are you using to answer the question?
1. Data Preprocessing: Clean and structure the data, removing irrelevant or missing information.
2. Feature Engineering: Select and transform features to enhance model performance.
3. Regression Models: Utilize Random Forests to evaluate the influence of specific incentives on monthly vehicle sales.
4. Hyperparameter Tuning: Employ techniques like GridSearch and RandomizedSearch to optimize model parameters and achieve superior performance.
5. Feature Importance Analysis with SHAP: Dive deep into influential features using SHAP values, offering a granular view of the contribution of each feature to the model's predictions.

#### Results
What did your research find?
1. Initial Model: An initial Linear Regression model achieved limited predictive performance (R-squared score: ~0.07).
2. Random Forest Model: Improved performance was achieved using the Random Forest model (R-squared score: ~0.53).
3. Feature Importance Analysis: SHAP values highlighted plan cost and discount as key factors influencing sales.
4. Segment-Specific Analysis: In-depth analysis conducted for specific segments, using SHAP force plots for interpretability.

#### Next steps
What suggestions do you have for next steps?
Model Refinement: Continuously improve models by experimenting with hyperparameters, feature engineering, and advanced algorithms to gain deeper insights into the importance of specific incentives.
Segment-Specific Strategies: Tailor incentive programs based on insights from segment-specific analysis to optimize sales strategies.
Collect More Data: Gather additional data to further enhance the model's predictive power and gain deeper insights.
Foster Collaboration: Encourage collaboration between technical and non-technical teams to facilitate data-driven decision-making.

#### Outline of project
Data sparated in individual folders for each segment.

- /script]/()
- [Link to notebook 2]()
- [Link to notebook 3]()


##### Contact and Further Information
rubendario11@gmail.com
+52 442 642 5436