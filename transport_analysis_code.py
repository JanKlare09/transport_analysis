### LOAD MODULES ###
import numpy as np
import pandas as pd
import scipy.stats

import geopandas as gpd
import libpysal as lps
import esda
import mgwr
import mapclassify

import sklearn.ensemble

import matplotlib.pyplot as plt



### PART 1: OD MATRIX PRE-PROCESSING ###

# Load original dataset and shapefile with 892 HW zones
AM_uc1_original = pd.read_csv(path, names = ['ORIG', 'DEST', 'PCU']) 
hw_shp = gpd.read_file(path)
hw_zones_gm = list(hw_shp['HW1022'])

# Filter for flows with O and D within Greater Manchester
hw_gm = AM_uc1_original.loc[(AM_uc1_original['ORIG'].isin(hw_zones_gm)) &\
                            (AM_uc1_original['DEST'].isin(hw_zones_gm))]

# Intrazonal demand
hw_intra = hw_gm[hw_gm['ORIG'] == hw_gm['DEST']][['ORIG', 'PCU']]  
hw_intra.rename(columns = {'ORIG': 'HW1022'}, inplace = True)

# Interzonal demand
hw_gm = hw_gm[hw_gm['ORIG'] != hw_gm['DEST']]

# Trip production
hw_gm_O = pd.DataFrame(hw_gm.groupby('ORIG').sum()['PCU'])
hw_gm_O.reset_index(inplace = True)
hw_gm_O.rename(columns = {'ORIG': 'HW1022'}, inplace = True)
trip_prod_shp = hw_shp.merge(hw_gm_O, on = 'HW1022')

# Trip attraction
hw_gm_D = pd.DataFrame(hw_gm.groupby('DEST').sum()['PCU'])
hw_gm_D.reset_index(inplace = True)
hw_gm_D.rename(columns = {'DEST': 'HW1022'}, inplace = True)
trip_attr_shp = hw_shp.merge(hw_gm_D, on = 'HW1022')



### PART 2: CHOOSING THE FINAL SET OF 633 ZONES ###

# Load lookup table Output Areas <--> HW zones
lookup_oa_hw = pd.read_csv(path)

# Load trip production dataframe and shapefile
hw_prod = pd.read_csv(path)
trip_prod_shp = gpd.read_file(path)
hw = pd.DataFrame(hw_prod['HW1022'])

# Load mid-year population estimates
pop_est = pd.read_csv(path)

# Aggregation on HW level according to lookup table
pop_est_hw = pop_est.rename(columns = {'GEOGRAPHY_CODE':'OA11CD'}).join(\
                                    lookup_oa_hw.set_index('OA11CD'),\
                                    on = 'OA11CD')

pop_est_hw = pop_est_hw[['OA11CD', '2011', '2014','2017', 'HW1022']].groupby(\
                    'HW1022').sum()
pop_est_hw.reset_index(inplace = True)

# Add HW zones that do not have any population centroid within their boundaries
pop_est_hw = hw.merge(pop_est_hw, on = 'HW1022', how = 'outer')

# Get population change 2011-2014, 2014-2017 and 2011-2017
pop_est_hw['c_11_14'] = (pop_est_hw['2014']/pop_est_hw['2011'])-1
pop_est_hw['c_14_17'] = (pop_est_hw['2017']/pop_est_hw['2014'])-1
pop_est_hw['c_11_17'] = (pop_est_hw['2017']/pop_est_hw['2011'])-1

# Tag HW zones with 500 inhabitants and more in 2011 and 2017 as 'suitable'
# 680 zones are left
pop_est_hw['suitable'] = np.nan
pop_est_hw.loc[(pop_est_hw['2011'] >= 500) & (pop_est_hw['2017'] >= 500), 'suitable'] = 1
suitable_zones = pop_est_hw.loc[pop_est_hw['suitable'] == 1, ['HW1022', '2011', '2017', 'c_11_17']]

# Remove 13 outliers with salient population changes
# 667 zones are left
suitable_zones = suitable_zones[(suitable_zones['c_11_17'] >= -0.2) &\
                                (suitable_zones['c_11_17'] <= 0.45)]




# Load usual residential population (KS 101 EW) #
ks_101_ew = pd.read_csv(path)

# Aggregation on PT level according to lookup table
ks_101_hw = ks_101_ew.rename(columns = {'GEOGRAPHY_CODE':'OA11CD',\
                                        'All usual residents': 'res_pop',\
                                        'Females': 'f',\
                                        'Males': 'm'}).join(\
                                    lookup_oa_hw.set_index('OA11CD'),\
                                    on = 'OA11CD')
res_pop_hw = ks_101_hw[['OA11CD', 'res_pop', 'f', 'm', 'HW1022']].groupby(\
                    'HW1022').sum() 
res_pop_hw.reset_index(inplace = True)                                                   
res_pop_hw = res_pop_hw[['HW1022', 'res_pop']]
suitable_zones = suitable_zones.merge(res_pop_hw, on = 'HW1022')
 
# Get PCU per 100 residents
suitable_zones_map_shp = trip_prod_shp.merge(suitable_zones, on = 'HW1022')
suitable_zones_map_shp = suitable_zones_map_shp[['HW1022', 'res_pop', 'c_11_17', 'PCU', 'geometry']]
suitable_zones_map_shp['d_pc_2011'] = suitable_zones_map_shp['PCU']/suitable_zones_map_shp['res_pop']
suitable_zones_map_shp['d_pc_2011'] = suitable_zones_map_shp['d_pc_2011']*100


# Remove 5% of the observations (34 zones) to obtain final dataset
# Get final dataset with 633 zones
final_zones_shp = suitable_zones_map_shp[(suitable_zones_map_shp['d_pc_2011']\
										> suitable_zones_map_shp['d_pc_2011'].quantile(q = 0.025))&\
                                         (suitable_zones_map_shp['d_pc_2011']\
										 < suitable_zones_map_shp['d_pc_2011'].quantile(q = 0.975))]
final_zones_shp = final_zones_shp[['HW1022', 'PCU', 'd_pc_2011', 'geometry']]



### PART 3: CORRELATION ANALYSIS ###

# Load final dataset and final shapefile
data_df = pd.read_csv(path)
data_shp = gpd.read_file(path)
dependent_var = data_shp[['HW1022', 'PCU', 'd_pc_2011']]

# 1.) Correlation coefficients
# Generate correlation coefficients for socioeconomic variables and trip production rate
correlation_table = pd.DataFrame()
correlation_table['variable'] = var_data_df

def correlation_lookup (coeff, row, p_value = False):
    if coeff == 'Pearson':
        if p_value == False:
            output = scipy.stats.pearsonr(np.array(\
                                    data_df[row]), np.array(\
                                    data_df['d_pc_2011']))[0]
        else:
            output = scipy.stats.pearsonr(np.array(\
                                    data_df[row]), np.array(\
                                    data_df['d_pc_2011']))[1]
    else:
        if p_value == False:
            output = scipy.stats.spearmanr(np.array(\
                                    data_df[row]), np.array(\
                                    data_df['d_pc_2011']))[0]
        else:
            output = scipy.stats.spearmanr(np.array(\
                                    data_df[row]), np.array(\
                                    data_df['d_pc_2011']))[1]
    
    return output
	
	
	# Helper Function to get folds
def get_folds_ols (X,y,fold, num_folds):
    X_folds = np.array_split(X, num_folds)
    test_X = X_folds.pop(fold)
    train_X = np.concatenate(X_folds)
    
    y_folds = np.array_split(y, num_folds)
    test_y = y_folds.pop(fold)
    train_y = np.concatenate(y_folds)
    return train_X, train_y, test_X, test_y

def cv_ols(X,y,num_folds):
    cv_ols_losses = []
    for i in range (num_folds):
        # Get data
        train_X, train_y, test_X, test_y = get_folds_ols(X,y,i,num_folds)
        # Fit model
        exog_train = sm.add_constant(train_X)        
        endog_train = train_y
        #print(exog_train)
        
        model = sm.OLS(endog_train, exog_train)
        result = model.fit()
        #print(result.summary())

        # Predict
        exog_test = sm.add_constant(test_X)
        pred_y = result.predict(exog_test)           
        
        # Get loss
        mean_fold_loss = np.sqrt(np.sum(np.square(pred_y-test_y))/test_y.shape[0])
        cv_ols_losses.append(mean_fold_loss)
        
    return np.mean(cv_ols_losses)

test = cv_ols(X_2,y_2, 10)

correlation_table['pearson_coeff'] = correlation_table['variable'].apply(\
                                lambda row: correlation_lookup('Pearson', row))

correlation_table['pearson_p'] = correlation_table['variable'].apply(\
                                lambda row: correlation_lookup('Pearson', row, True))                                               

correlation_table['spearman_coeff'] = correlation_table['variable'].apply(\
                                lambda row: correlation_lookup('Spearman', row))

correlation_table['spearman_p'] = correlation_table['variable'].apply(\
                                lambda row: correlation_lookup('Spearman', row, True))                                               

    
# Adjust to linearise correlation coefficients for skewed and leptokurtic distributions
adjusted_list = ['ya_I_sh', 'ya_II_sh', 'p_rent_sh', 'm_sh', 'o_depr_sh',\
                 'p_ph', 'w_sh']

def adjust_replace (row):
    if row in adjusted_list:
        output = 1
    else:
        output = 0
    return output

correlation_table['adjusted'] = correlation_table['variable'].apply(\
                                lambda row: adjust_replace(row))

for var in adjusted_list:
        correlation_table.loc[correlation_table['variable'] == var,'pearson_coeff']\
            = scipy.stats.pearsonr(np.log(data_df[var]),\
                                  np.log(data_df['d_pc_2011']))[0]
        correlation_table.loc[correlation_table['variable'] == var,'pearson_p']\
            = scipy.stats.pearsonr(np.log(data_df[var]),\
                                  np.log(data_df['d_pc_2011']))[1]
        correlation_table.loc[correlation_table['variable'] == var,'spearman_coeff']\
            = scipy.stats.spearmanr(np.log(data_df[var]),\
                                  np.log(data_df['d_pc_2011']))[0]
        correlation_table.loc[correlation_table['variable'] == var,'spearman_p']\
            = scipy.stats.spearmanr(np.log(data_df[var]),\
                                  np.log(data_df['d_pc_2011']))[1]       
                      
correlation_table['pearson_coeff_abs'] = correlation_table['pearson_coeff'].abs()               
correlation_table.sort_values(by = 'pearson_coeff_abs', axis = 0, ascending = False, inplace = True)
correlation_table = correlation_table[['variable', 'pearson_coeff', 'pearson_p',\
                                       'spearman_coeff', 'spearman_p', 'adjusted']]
								

								
# 2.) Generate one GWR model per variable								
def gwr_uni(variable, y):
    
    # build X matrix
    X = np.array(data_shp[variable]).reshape((-1,1))

    # run GWR
    sel = mgwr.sel_bw.Sel_BW(coords, y, X, fixed = False, kernel = 'bisquare')
    bw = sel.search(criterion = 'AIC')

    model = mgwr.gwr.GWR(coords, y, X, bw)
    results = model.fit()

    # get results
    adj_R2 = results.adj_R2
    beta_mean = np.mean(results.params[:,1])
    beta_std = np.std(results.params[:,1])

    # adjust results for HW zones where beta is significant
    t_vals = results.filter_tvals(alpha = 0.05)[:,1]
    # get share of zones with significant t
    sig_share = np.count_nonzero(t_vals)/t_vals.shape[0]
    # get mean and std
    sig_mean = np.mean(results.params[np.nonzero(t_vals),1])
    sig_std = np.std(results.params[np.nonzero(t_vals),1])
    
    return pd.Series([adj_R2, beta_mean, beta_std, sig_share, sig_mean, sig_std])


# Add results to correlation table
correlation_table[['adj_R2', 'beta_mean', 'beta_std', 'sig_share', 'sig_mean', 'sig_std']] =\
correlation_table['variable'].apply(lambda row: gwr_uni(row, y))



### PART 4: GWR MODEL 1 ###

# Define variables
y = np.array(data_shp['d_pc_2011']).reshape((-1,1))
w_sh = np.array(data_shp['w_sh']).reshape((-1,1))
h_man_sh = np.array(data_shp['h_man_sh']).reshape((-1,1))
s_rent_sh = np.array(data_shp['s_rent_sh']).reshape((-1,1))
X = np.hstack([w_sh, h_man_sh, s_rent_sh])																										   
																										   
# Run the GWR
sel_1 = mgwr.sel_bw.Sel_BW(coords, y, X, fixed = False, kernel = 'bisquare')
bw_1 = sel_1.search(criterion = 'AIC')
model_1 = mgwr.gwr.GWR(coords, y, X, bw_1)
results_1 = model_1.fit()



### PART 5: GWR MODEL 2 ###

# Define variables
y = np.array(data_shp['PCU']).reshape((-1,1))
w_sh = np.array(data_shp['w_sh']).reshape((-1,1))
h_man_sh = np.array(data_shp['h_man_sh']).reshape((-1,1))
s_rent_sh = np.array(data_shp['s_rent_sh']).reshape((-1,1))
res_pop = np.array(data_shp['res_pop']).reshape((-1,1))
X = np.hstack([w_sh, h_man_sh, s_rent_sh, res_pop])

# Run the GWR
sel_2 = mgwr.sel_bw.Sel_BW(coords, y, X, fixed = False, kernel = 'bisquare')
bw_2 = sel_2.search(criterion = 'AIC')
model_2 = mgwr.gwr.GWR(coords, y, X, bw_2)
results_2 = model_2.fit()



### PART 5: RANDOM FOREST MODEL 1 ###

# Define variables
y_1 = np.array(data_shp['d_pc_2011'])
w_sh = np.array(data_shp['w_sh']).reshape((-1,1))
h_man_sh = np.array(data_shp['h_man_sh']).reshape((-1,1))
s_rent_sh = np.array(data_shp['s_rent_sh']).reshape((-1,1))
X_1 = np.hstack([w_sh, h_man_sh, s_rent_sh])

# Run RF with 10 different random seeds
def rf_model_1 (seed):
    regr_1 = sklearn.ensemble.RandomForestRegressor(n_estimators = 1000, oob_score = True,\
                                              max_depth = 10, min_samples_split = 30,\
                                                  random_state = seed)
    regr_1.fit(X_1,y_1)
    importances = regr_1.feature_importances_
    score = regr_1.score(X_1,y_1)
    return importances[0], importances[1], importances[2], score

random_seeds = [1956, 1957, 1963, 1965, 1995, 1996, 1997, 2002, 2011, 2012]

importance_1, importance_2, importance_3, score_list = zip(*[rf_model_1(seed) for seed in random_seeds])



### PART 6: RANDOM FOREST MODEL 2 ###

# Define variables
y_2 = np.array(data_shp['PCU'])
w_sh = np.array(data_shp['w_sh']).reshape((-1,1))
h_man_sh = np.array(data_shp['h_man_sh']).reshape((-1,1))
s_rent_sh = np.array(data_shp['s_rent_sh']).reshape((-1,1))
res_pop = np.array(data_shp['res_pop']).reshape((-1,1))
X_2 = np.hstack([w_sh, h_man_sh, s_rent_sh, res_pop])

# Run RF with 10 different random seeds
def rf_model_2 (seed):
    regr_2 = sklearn.ensemble.RandomForestRegressor(n_estimators = 1000, max_features = 3, oob_score = True,\
                                              max_depth = 10, min_samples_split = 30,\
                                            random_state = seed)
    regr_2.fit(X_2,y_2)
    importances = regr_2.feature_importances_
    score = regr_2.score(X_2,y_2)
    oob_score = regr_2.oob_score_
    return importances[0], importances[1], importances[2], importances[3], score, oob_score

importance_1, importance_2, importance_3, importance_4, score_list, oob_score_list = zip(*[rf_model_2(seed) for seed in random_seeds])



### PART 7: CROSS-VALIDATION OF BOTH MODELS ###

# Cross-validation for GWR
def get_folds (X,y,coords,fold, num_folds):
    X_folds = np.array_split(X, num_folds)
    test_X = X_folds.pop(fold)
    train_X = np.concatenate(X_folds)
    
    coords_folds = np.array_split(np.array(coords), num_folds)
    test_coords = coords_folds.pop(fold)
    train_coords = np.concatenate(coords_folds)
    
    y_folds = np.array_split(y, num_folds)
    test_y = y_folds.pop(fold)
    train_y = np.concatenate(y_folds)
    return train_X, train_y, test_X, test_y, train_coords, test_coords

def cv_gwr(X,y,coords,num_folds,bw):
    for i in range (num_folds):
        # Get data
        train_X, train_y, test_X, test_y, train_coords, test_coords = get_folds(X,y,coords,i,num_folds)
        # Fit model
        train_coords_list = list(zip((np.array(train_coords)[:,0]), (np.array(train_coords)[:,1])))
        model = mgwr.gwr.GWR(train_coords_list, train_y, train_X, bw = bw, fixed = False, kernel = 'bisquare')

        # Predict
        predictions = model.predict(test_coords, test_X)
        pred_y = predictions.predy
        
        # Get loss
        mean_fold_loss = np.sqrt(np.sum(np.square(pred_y-test_y))/test_y.shape[0])
        cv_gwr_losses.append(mean_fold_loss)
        
    return cv_gwr_losses

cv_gwr_results = cv_gwr(X,y,coords,10,51)  

# Cross-validation for RF
def get_folds_rf (X,y,fold, num_folds):
    X_folds = np.array_split(X, num_folds)
    test_X = X_folds.pop(fold)
    train_X = np.concatenate(X_folds)
    
    y_folds = np.array_split(y, num_folds)
    test_y = y_folds.pop(fold)
    train_y = np.concatenate(y_folds)
    return train_X, train_y, test_X, test_y

def cv_rf(X,y,num_folds, seed):
    cv_rf_losses = []
    for i in range (num_folds):
        # Get data
        train_X, train_y, test_X, test_y = get_folds_rf(X,y,i,num_folds)
        # Fit model
        regr_2 = sklearn.ensemble.RandomForestRegressor(n_estimators = 1000, max_features = 3,\
                                              max_depth = 10, min_samples_split = 30,\
                                                  random_state = seed)
        regr_2.fit(train_X, train_y)

        # Predict
        pred_y = regr_2.predict(test_X)
        
        # Get loss
        mean_fold_loss = np.sqrt(np.sum(np.square(pred_y-test_y))/test_y.shape[0])
        cv_rf_losses.append(mean_fold_loss)
        
    return np.mean(cv_rf_losses)

cv_rf_results = [cv_rf(X_2, y_2, 10, seed) for seed in random_seeds]

# END