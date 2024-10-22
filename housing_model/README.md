# Model For Predicting House Prices
This project shows the lifecycle of an end-to-end machine learning project. The goal is to create a predicition model that can predict the prices of houses in Carlifornia.

## Steps Outlined
1. Look at the bigger picture
    * Frame the problem
    * Select a performance measure. Root Mean-Squared Error is the measure used for this model.
    * Check any assumptions made
2. Get the data
    * Download the training data
    * Explore the data to get a feel of the data structure
    * Set aside a test set. The method used to set aside the test set in this example is `Stratified Sampling`
3. Discover and visualize the data to gain insights
    * Take a deeper look at the data. Includes steps such as; finding correlations, experimenting attribute combinations, etc.
    * Exploration should be done on the training set only.
4. Prepare the data for the ML algorithm
    * Data cleaning to address missing values
    * handling text and categorical attributes
    * Custom tranformers
    * Feature scaling
    * Transformation pipelines
5. Select and train a model
    * Perform training and evaluation on the training set
    * Experiment with at least 5 different models then select the best 2 candidates
    * Use cross-validation for better model performance analysis.
    * Saving models settings and state for future reference.
6. Fine-tune the model
    * Fiddle with hyperparameters using `GridSearchCV` and `RandomizedSearchCV` to find the best hyperparameter settings.
    * Ensemble models
    * Analyze the best models and their errors
    * Evaluate the model on the training set
7. Launch, monitor and maintain the system.

**For more details, check the associated notebook**
The `.pkl` file for the final and best model using `RandomForest` has not been uploaded since the size is larger than the allowed limit.