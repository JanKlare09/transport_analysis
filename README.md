# transport_analysis
In September 2020, I submitted my dissertation "Combining Origin-Destination Transport Data with Local Area Data" to the University of Manchester for the degree MSc Data Science.

My dissertation describes a way to integrate origin-destination matrices for motorised individual transport with local area data from the 2011 UK Census. Thereby, it lays the foundation for an analysis of the correlations between socioeconomic characteristics and trip production patterns across Greater Manchester. Furthermore, the study evaluates the use of Geographically Weighted Regression (GWR) and Random Forest models for more comprehensive transport demand models.

As GWR and Random Forest models incorporate spatial heterogeneity and non-linear relationships, they capture the explanatory power of the socioeconomic variables more extensively than the basic linear regression model. Therefore, the study shows that it may be advisable to use GWR or Random Forest approaches for complex transport demand models.

Since the transport data used in my dissertation is not available to the public, it is impossible for other users to reproduce my results. However, the files in this repository show the script I used for my disseration.

As visible in PART 1 and PART 2 of the main file, I pre-processed the data using numpy, pandas and geopandas. The correlation analysis (PART 3) was carried out using scipy.stats. I built the GWR models with the help of the mgwr module (PART 4 & PART 5), while I used scikit-learn for the Random Forest models (PART 6 & PART 7). PART 8 contains the self-developed functions I applied to cross-validate the model results.

The additional map_plot.py script presents the code I used to create complex multi-layer maps such as mod_2_w_sh.png, a map the displays the regression coefficients for the variable w_sh across the study area.
