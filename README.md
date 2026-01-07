# NYC Taxi Trip Analytics (PySpark)

Large-scale analysis of NYC Yellow Taxi trips using PySpark, focused on
trip efficiency, temporal patterns, pickup locations, weather effects,
and basic predictive modelling.

---

## Overview

This project explores what drives **taxi trip efficiency** in New York City,
measured mainly as earnings per minute and per kilometre.

Key areas:
- efficiency by hour and day
- pickup zone performance
- impact of weather (rain, temperature)
- baseline prediction of trip efficiency

---

## Stack

PySpark, Python, Spark MLlib, Pandas, Matplotlib

---

## Data

NYC TLC Yellow Taxi trips (2019) and NOAA weather data.  
Raw datasets are excluded due to size and licensing.  
This repo reconstructs a university assignment after submission.

---

## Structure
src/
01_load_and_clean.py
02_feature_engineering.py
03_weather_join.py
04_analysis.py
05_models.py

---

## Modelling

- Linear regression to predict earnings per minute
- Logistic regression to classify high-efficiency trips

---

## Note: Modules are designed to be composed in sequence within a Spark session.

---

## Author

Habib Haadi  
Data Science @ University of Melbourne  
https://github.com/habibhaadi

