# Exploring the Impact of Behavioral and Contextual Factors on School Shootings (1966–2023)

## Overview
This project analyzes incident-level data from the CHDS School Shooting Safety Compendium (1966–2023) to explore long-term trends, severity patterns, and the relationships between behavioral, contextual, and incident characteristics. The analysis combines exploratory data analysis (EDA), correlation analysis, and exploratory machine learning models to better understand how multiple factors interact in these rare but high-impact events.

## Problem Statement
School shootings represent a persistent and complex public safety challenge in the United States. Prevention efforts often rely on limited or reactive indicators. This project aims to use historical data to identify patterns and co-occurring factors that may inform prevention-oriented discussions, resource allocation, and future research—without claiming causal or individual-level predictions.

## Data
The analysis uses the **CHDS School Shooting Safety Compendium**, a historical dataset covering incidents from 1966 to 2023.  
Key variables include:
- Incident year and location  
- Number injured and fatalities  
- Total firearms brought to the scene  
- Behavioral indicators (e.g., psychiatric medication, paranoia, isolation, depressed mood, childhood trauma)  

Data preparation steps include:
- Selecting variables with sufficient coverage  
- Converting and validating numeric fields  
- Handling missing values  
- Creating derived measures such as total casualties  

## Methods
The workflow includes:
- Exploratory data analysis (histograms, boxplots, trend plots)  
- Correlation analysis to examine relationships among behavioral and contextual variables  
- Exploratory machine learning models (e.g., bagging and boosting approaches)  
- Model evaluation using confusion matrices and classification metrics, with attention to class imbalance and rare-event challenges  

## Results
- Incident frequency shows an overall increase over time, with clustering in more recent decades.  
- Injury counts and firearm counts are highly right-skewed, indicating that most incidents are low-severity with a small number of extreme outliers.  
- Correlation patterns suggest that some behavioral and contextual variables co-occur, but relationships are complex and generally modest in strength.  
- Predictive performance for higher injury counts is limited, highlighting the difficulty of modeling rare, extreme events.

## Limitations and Ethics
- The dataset contains missing and unevenly reported variables.  
- The analysis is observational: correlations do not imply causation.  
- Results should not be used for individual profiling or predictive enforcement.  
- Ethical priorities include careful framing, privacy protection, and avoiding stigmatization of mental health conditions.  
- Findings are intended to support prevention-oriented discussion and further research.

## Tools and Technologies
- Python  
- pandas, NumPy  
- scikit-learn  
- matplotlib, seaborn  

## Conclusion
This project demonstrates how data science can be applied to a sensitive and complex social issue using exploratory analysis and modeling. While no single factor explains these events, the results highlight the importance of multi-factor perspectives, careful interpretation, and responsible, prevention-focused use of data.

