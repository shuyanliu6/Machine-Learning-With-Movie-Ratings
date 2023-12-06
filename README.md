# Machine Learning with Movie Ratings

## Project Overview
This project explores the intriguing domain of movie ratings, applying sophisticated machine learning techniques to predict and analyze various facets of movie success. It utilizes an extensive dataset containing ratings for a wide array of movies, alongside respondents' demographic information and movie-watching preferences, to uncover patterns and insights into what influences movie ratings and success.

## Dataset Overview
The `movieReplicationSet.csv` dataset is a comprehensive collection featuring:
- Ratings for approximately 475 movies, covering a diverse range of genres and years.
- Additional variables related to the respondents' demographics, movie-watching habits, and preferences.
- Data points reflecting individual viewer responses, providing a rich basis for analysis.

## Key Objectives
- **Predictive Modeling:** Develop models to predict movie ratings and success based on various features extracted from the dataset.
- **Analytical Insights:** Extract meaningful insights from data to understand the dynamics behind movie ratings and success.
- **Algorithm Comparison:** Evaluate and compare different machine learning algorithms for their effectiveness in predicting movie outcomes.

## Models Employed
1. **L1 Regularization (Lasso Regression):** For feature selection and addressing overfitting, highlighting influential factors in movie ratings.
2. **L2 Regularization (Ridge Regression):** Implemented to tackle multicollinearity, enhancing prediction accuracy.
3. **Multiple Linear Regression:** Used to model the relationship between multiple independent variables and the dependent variable of movie rating.
4. **Logistic Regression:** Utilized for classification tasks, such as categorizing movies based on success metrics.

## Methodology
- **Data Preprocessing:** Cleaning and transforming data for optimal model performance.
- **Model Training:** Careful training and testing of each model on the dataset.
- **Performance Evaluation:** Models are assessed based on metrics like AUC, accuracy, precision, recall, and F1-score.

## Key Findings and Insights
- Demonstrated strong predictive capabilities with high AUC scores across the test set.
- Insights into features significantly impacting movie ratings and success.
- Comparative analysis of models provides understanding of their applicability in different scenarios.

## Tools and Technologies
- Python and Jupyter Notebooks for data analysis and modeling.
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, etc.

## How to Use
1. Clone or download this repository.
2. Ensure Python and necessary libraries are installed.
3. Run the Jupyter notebook to view the analysis or use it as a basis for further exploration.

## Contributions
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.
