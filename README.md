# WRF_Analysis
This repository contains Python scripts for analyzing temperature data obtained from the (Digital Urban Climate Twin simulator)[https://fass.nus.edu.sg/srn/2024/01/08/cooling-singapore-2-0-digital-urban-climate-twin/] as well as some ancillary files read by these scripts as input.

The main analysis code is in the 'analysis' class in the analysis.py file. This class contains numerous functions for processing the temperature data. 

First, the input data for the user-specified run and time is read from a file. The input data can be in the form of a Pandas dataframe stored in the '.pkl' format or as a raster in the '.tif' format. Next, this data is processed to perform various types of calculations such as computing the average, minimum and/or maximum temperature within a user-specified region. The data can then be visualized by plotting the spatial temperature distribution as a contour plot or by plotting the average temperature within a user-specified region over the course of the simulation time.

The 'analysis' class also makes use of the (DBSCAN)[https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html] clustering algorithm implemented in the scikit-learn library to identify clusters of locations whose temperature is larger than a user-specified threshold at a user-specified time.
