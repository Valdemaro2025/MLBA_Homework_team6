# MLBA_Homework_team6

# :house: Russia Real Estate 2018-2021

## Dataset description

- **Date** — date of publication of the announcement
- **Time** — the time when the ad was published
- **Geo_lat** — latitude
- **Geo_lon** — longitude
- **Region** — region of Russia
- **Building type** — Panel / Monolithic / Brick / Blocky / Wooden / Other
- **Object_type** — Secondary / New building
- **Level** — apartment floor
- **Levels** — number of storeys
- **Rooms** — number of living rooms (–1 = studio)
- **Area** — total area of the apartment
- **Kitchen area** — kitchen area
- **Price** — price in rubles

## ML tasks

### Regression
**Goal:** Predict the exact price of a property based on its features.

### Classification
**Goal:** Classify a property into a price bracket (e.g., "budget", "mid-range", "luxury").

### Clustering
**Goal:** Discover natural clusters of similar properties based on features.

---

## Chosen Task
**Property Price Prediction (Regression)**

### Why
- widely used in real-estate analytics  
- large dataset with rich feature space  
- clear target variable: **price**

## Business problem statement

The real-estate market is **very inconsistent**: prices vary widely across regions, building types,  
and apartment characteristics. Manual valuation is **inefficient**, making it difficult for buyers,  
sellers, and platforms to determine a fair market price.

---

### Goal
Build an automated property **price prediction** model that provides accurate, data-driven valuation  
based on property features.

---

### Business value

**For buyers & sellers:**  
- Avoid overpaying or underselling by getting an objective price estimate.

**For realtors / agencies:**  
- Speed up property valuation from hours to seconds.  
- Improve deal conversion.

**For platforms (real-estate marketplaces):**  
- More transparent and standardized pricing across regions.

## ML approach and method

### Approach: Supervised Learning
The task is framed as **supervised learning**, meaning the model learns patterns from historical examples  
where the correct property prices are already known.  
This enables the model to generalize from past data and make predictions for new, unseen apartments.

**Key ideas:**
- learning is based on past labeled examples  
- the dataset contains input features and ground-truth price values  
- the model optimizes its parameters by minimizing prediction error  

---

## Methods

Below are the main machine-learning algorithms considered for price prediction, along with their strengths  
and limitations.

### 🔹 Linear Regression
A simple and fast baseline model.

**Pros:**
- very fast to train and interpret  

**Cons:**
- struggles with nonlinear patterns in the data  

---

### 🔹 Decision Trees
Tree-based models that split data into logical segments.

**Pros:**
- naturally capture nonlinear relationships  

**Cons:**
- prone to overfitting without constraints  

---

### 🔹 Random Forest
An ensemble of many decision trees trained on random subsets of data.

**Pros:**
- significantly reduces overfitting compared to a single tree  
- works well with mixed feature types and nonlinearities  

**Cons:**
- slower than a single tree (but still efficient)  

---

### 🔹 Hist Gradient Boosting
A modern, efficient gradient boosting implementation.

**Pros:**
- faster than classic gradient boosting  
- typically provides high predictive accuracy  

**Cons:**
- requires hyperparameter tuning to reach optimal performance  

---

## Metrics

To evaluate model quality, two common regression metrics are used:

### **Mean Absolute Percentage Error (MAPE)**
Measures the average percentage error between predicted and actual prices.  
Useful for understanding the *relative* accuracy of the model (e.g., "the model is off by ~7% on average").

### **Root Mean Squared Error (RMSE)**
Penalizes large errors more strongly.  
Helpful when it’s important to reduce big mispricing cases, which is crucial in real-estate valuation.

## Implementation: Data preparation

Before training the model, the dataset was cleaned and transformed to ensure consistent structure,  
remove unrealistic values, and prepare features for machine learning algorithms.

---

## Data cleaning

The following filtering rules were applied to reduce noise and eliminate obviously incorrect entries:

- Removed rows with **negative price values** (e.g., –365), as such prices are invalid and introduce anomalies.
- Removed **extreme price values** outside a reasonable range to prevent strong outliers from distorting the model.
- Excluded records where **room count equals zero**, since such listings typically correspond to incorrect or placeholder data.
- Dropped flats with **total area below 10 m²** or above **500 m²**, which fall outside realistic limits for residential property.
- Removed observations where **kitchen area < 2 m²** or **kitchen area > 50%** of the entire apartment — both cases indicate errors in reporting.
- Filtered out properties with **floor numbers outside the valid range** for their building.

These steps help stabilize the dataset and ensure that the model is trained on realistic, meaningful housing data.

---

## Data type conversion

To make the dataset more suitable for analysis and modeling, several fields were transformed:

- Converted **date and time fields** from raw numeric formats into proper `datetime` objects for clearer interpretation and temporal feature extraction.
- Re-encoded the **object type** (e.g., "new building" vs. "secondary") from numeric codes → to strings → to boolean flags, making the feature easier to use.
- Transformed **building type** into boolean or categorical format to improve compatibility with tree-based models.

Overall, these transformations ensure that the data has consistent types and semantics, making feature engineering and modeling more reliable.

## Implementation: Data visualization

To better understand the structure of the dataset and identify key patterns affecting apartment prices,  
a series of exploratory data visualizations was created. These visual checks help validate assumptions,  
reveal relationships between features, and highlight potential issues in the data.

---

## Types of visual analysis performed

### 🔹 Pricing analysis
Explored the distribution of property prices, identified skewness, checked for long tails,  
and verified how reasonable the values are after cleaning. This helps understand the general  
price landscape and potential modeling challenges.

### 🔹 Correlation matrix
Generated a heatmap illustrating correlations between numerical features.  
It highlights which variables strongly influence price (e.g., area, number of rooms)  
and which features may be redundant or weakly related.

### 🔹 Number of listings by region
Visualized the distribution of dataset entries across regions.  
This exposes data imbalance — some regions have significantly more listings than others,  
which can influence model behavior.

### 🔹 Price by region
Compared median or average prices across regions to see geographical differences.  
Useful for understanding regional pricing patterns and ensuring the model incorporates  
location-related variation properly.

### 🔹 Feature relationships
Built pairplots or scatterplot grids to observe how key features (area, rooms, kitchen size, floors)  
interact with each other. This reveals nonlinear dependencies and helps with feature engineering.

---

<img width="1446" height="1079" alt="image" src="https://github.com/user-attachments/assets/40cccaf1-1e77-4530-b945-634d7938750a" />

<img width="1042" height="1079" alt="image" src="https://github.com/user-attachments/assets/f90f0539-6ca5-4651-b477-9dcfb1f527cb" />

<img width="1915" height="619" alt="image" src="https://github.com/user-attachments/assets/ffa23185-1c6a-4129-a142-cac4908cf9d7" />

<img width="1913" height="750" alt="image" src="https://github.com/user-attachments/assets/bedb3d1a-f05b-43b1-b64f-fa1a2a22c61a" />


