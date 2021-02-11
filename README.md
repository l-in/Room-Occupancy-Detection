# Room-Occupancy-Detection

The detection of room occupancy has in recent years become an important topic with applications in the energy efficiency of buildings, the implementation of security features, and the monitoring of residents for health and safety purposes. The objective of this project is to use data from temperature, humidity, light, and carbon dioxide sensors to detect when a room in an office building in Mons, Belgium is occupied. In other words, the goal is to use environmental conditions to detect room occupancy without the use of a camera (for privacy reasons). 

This project is based on the data set and analysis from *Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models* (Candanedo and Feldheim, 2016).  

Note: variables and packages are surrounded by square bracketes (ie. [variable] and [package]). 

## The Dataset

The dataset stores information taken from a room in an office building in Mons, Belgium in the wintertime (February 2015). Specifically, temperature, humidity, light, and carbon dioxide measurements were taken, along with occupancy data. These measurements were taken by sensors every minute, and occupancy status was determined from pictures taken by a digital camera in the room. The pictures were manually sorted as “occupied” or “not occupied” and the corresponding environmental condition observations were labelled as such in the dataset. The data file had three datasets: one for training and two for testing.

## Data Wrangling

- the YYYY:MM:DD subsection of the [date] string was extracted to make the [year.month.date] variable
- [id.rel] variable was created by assigning an index to each observation

## Methodology

1. Augmented Dickey-Fuller (ADF) test to test non-stationarity of predictors
2. Data analysis
- modelling each variable (a univariate time series) using: 
    - min-max normalization
    - kernel density estimation 
    - spline smoothing
    - Lowess smoothing
    - Fourier series
    - ARMA models
    - changepoint detection 
- using the variables (time series) as predictors in classification models
    - decision trees ([rpart] package)
    - random forests ([randomForest] package)
    - support vector machines ([e1071] package)
    - artificial neural networks ([neuralnet] package)
    - logistic regression
