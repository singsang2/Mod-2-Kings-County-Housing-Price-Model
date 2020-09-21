
# Module 2 Final Project 
<br>Author: Sung Hoon Bae
<br>Instructor: James Irving
<br>Review Instructor: Matt Sparr
<hr>

# Introduction

Multivariable linear regression models can be used to predict the outcomes of an event that depends on multiple factors. There are mainly two types of goals we could focus on when making a model and they are:
    1) predictable model
    2) interpretable model.
<img src='different_models.png'>
source: https://towardsdatascience.com/the-balance-accuracy-vs-interpretability-1b3861408062

    - Predictable model
        These models have high accurcay which would be great in predicting values from predictors.
        
    - Interpretable model
        These models have high interpretability which would be great in understanding the models and how different predictors play in determining the target value.

Even though every model has certain aspect of predictability and interpretability, in this module project, we will focus on making a model with high interpretability.

## Objectives
The objective is to use the given data along with other relevant outside data to make more 'interpretable' model to show insights to the house owners of what they could do to increase their house sale price.

## Finding from EDA
Based on multiple EDA's, we have concluded the following factors might be affecting the house prices:

### Seasonal Factor
<img src="img/seasons.png"> <img src='img/month.png'>
As we can see from the two graphs above, depending on time of the year, the total house sale price differ.
> 'season' was added as categorical data in the model

### Geographical Factor
<img src='img/geography.png'>
From the figure above, it becomes apparent that the house prices are affected by geographical locations. For example, the following graph shows house prices for depending on the distance from Microsoft HQ.
<img src='img/MSHQ.png'>
According the graph above, the mean distance and the price have negative association which makes sense since further away from valuable site (MS HQ), the land value also could drop as well.

> The following were added as continuous predictors to our model:
    - average distance from top attraction sites
    - distance from Microsoft HQ
    - distance from Seattle

### Mortgage Rate Factor
Mortgage rate is one of the major factor that could shift people from buying or hold off on buying a house. Buyers definitely would prefer having lower rate. So we included mortgage rate as one of our factors.
> Rate was added to our model as continuous predictor.

### Available Living Space Factor
The square footage of the living space was divided by the number of rooms to represent how spacious each room would be for a given house. 
<img src='img/living_space.png'>
We observed a positive associated relationship between them.
> living square footage / bedrooms was added as a continuous predictor in our model.

### Quality Bathrooms Factor
Lastly, we have noticed that `bathroom` numbers had a strong positive relationship with the price as well as `grade`. So these two quantities were multiplied to provide
        <br>1) amplified benefits for bathrooms with good qualities and
        <br>2) some penalties with bathrooms with poor qualities.
The following regplot was generated using seaborn:
<img src='img/bathrooms.png'>
The graph showed a strong positive association.
> Bathrooms x grades was added as a continuous predictor in our model.

# The Dataset
The following data sets were used.
1. Given King's County house sale data
    - We were given with King's County house sale data between 2014-2015. 
    - This data will be used as our main data
    
2. Past mortgage rates by month
    - Source: http://www.freddiemac.com/pmms/pmms30.html

3. Top attractions near Seattle
    - Source: https://www.timeout.com/seattle/
---
The following API was used to obtain additional data:
1. Google Geocode API
    - This API was used to obtain latitude and longitude data of various locations
   

# Method

"MakeModel" class was made to process and make models in repeatable ways. The following are major methods in the "MakeModel" class that was used to produce multiple models:
<hr>

1. <code>MakeModel.col_classifier()</code> - allows the user to classify each column in the dataframe either (1) continous, (2) categrorical or (3) drop the column.
<br>
2. <code>MakeModel.outliers()</code> - allows the user to filter out outliers if any using either (1) z-score method or (2) IQR method.
<br>
3. <code>MakeModel.scaler()</code> - allows the user to scale data using (1) standardscale, (2) min-max scale, or (3) logarithmic scale
<br>
4. <code>MakeModel.multicoliearity()</code> - allows the user to drop predictors that might be multicolinear using either (1) correlation matrix or (2) variance inflaction factor (VIF).
<br>
5. <code>MakeModel.ohe()</code> - allows the user to do one-hot-encode categorical predictors of user's choice
<br>
6. <code>MakeModel.split()</code> - splits the data into train and test sets
<br>
7. <code>MakeModel.regression()</code> - using statsmodels library, creates a regression model.
<br>
8. <code>MakeModel.model.summary()</code> - prints out the summary of the model
<br>
9. <code>MakeModel.validate_model()</code> - prints out QQ plots, residual graphs, r2_score, and MSE of both train and test data for validation.
<hr>

>The process mentioned above was repeated multiple times until the desired model was created.

# Result (Model)

<img src="img/regression.png" width=500>

>The $r^2$ value turned out to be 0.725 which is not an ideal value but good 'enough' for an interpretable model.


<img src="img/FinalQQ.png">

>Even though there are slight non-normality at either ends, the residuals seem normal.
<hr>
<img src="img/ResidGraph.png">

>This residual graph shows that our model is homoscedastic but biased.
    - the model overpredicts low-priced houses
    - the model underpredicts high-priced houses

# Interpretation

According the result the following factors have been identified as positive effectors and negative effectors:
<br>
<hr>
<b>Positive Effectors:</b>

Major Positive Effectors:
    1. waterfront = water scenery matters
    2. average room space = more space is better
    3. grade x bathroom = having great bathrooms count!
    4. Seattle = Further you live from Seattle is better

Minor Positive Effectors: 
    1. Effective age 
    2. Selling in Spring
    3. Selling in Summer
<hr>
<b>Negative Effectors:</b>

Major Negative Effectors:
    1. mean distance from main attractions = further you live, the value goes down generally
    2. distance from Microsoft HQ = further you live, the value goes down generally

Minor Negative Effectors: 
    1. Mortgage rate – higher rate lowers the house value
    2. Selling in winter
<hr>


# Conclusion

The final model is limited in accuracy ($R^2 = 0.725$). However it can give us a good idea of what factors we can control and change to change the house price

<b>Actionable changes</b>
    - Renovate to 
        - Increase open space
        - Waterfront presence
        - Better quality bathrooms
    - What NOT to do
        - increase number of bathrooms and bedrooms in total
        - sell in the winter season
        - sell during high mortgage rate
	


# Further Studies

The following features could be added to increase the accuracy of the model:
    - school district for elementary, middle and high school rating
    - average traffic time to major attractions
    - noise level 
    - flood factors


