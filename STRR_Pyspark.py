import s3fs
import pandas as pd
import numpy as np
pd.options.display.max_columns = None
# csv file
# df_all = pd.read_csv("s3://eqrs-ngmc-datascience/Datascience/x.csv")
df_all = pd.read_csv("x_strr_sample.csv")
df = df_all#[:50000]
del df_all

'''
data$strr_pyf_start <- as.Date(data$strr_pyf_start, format="%m/%d/%Y")
data$provfs <- as.character(data$provfs)
    #data2=data[order( data$provfs,data$ptnt_id,factor(data$year),data$strr_pyf_start ),]
data2=data[order( data$provfs,data$ptnt_id,factor(data$year) ),]
'''
df['strr_pyf_start'] = pd.to_datetime(df['strr_pyf_start']).dt.strftime('%m/%d/%Y')
# df['provfs'] = df['provfs'].astype(str)
df['provfs'] = pd.to_numeric(df['provfs'], downcast='integer', errors='coerce').fillna(0)
df['provfs'] = df['provfs'].astype(int)
col_names = df.columns

# sorting with three columns
z_pd = df.to_records()
z_pd.sort(order=["provfs", "ptnt_id", "year"])
d2 = pd.DataFrame(z_pd)

classnames = (
"pdiab", "pdismiss", "pnhprev", "bmi_msg", "ashd1", "othcardiac1",
"carfail", "noambul", "pulmon", "notrans", "cancer", "diabetes",
"pvasc", "cva", "smoke", "alcoh", "drug", "inci_one", "inci_miss"
)

#R     z <- match(classnames, names(data2))
z = [ list(d2.columns).index(x) if x in d2.columns else None for x in  classnames]

for i in range(0, len(z)):
    col_mean = float(d2.iloc[:, [z[i]]].mean())
    if(col_mean<0.001 or col_mean>0.999):
        print("WARNING: Covariate", classnames[i],"has less than 0.1% variation.  The distribution of this covariate is", col_mean)
    else:
        print("WARNING: Covariate", classnames[i], "has missing values.")



cols = ["ptnt_id", "provfs", "t_trans", "strr_pyf_start", "wt_trans", "ot_trans","pdiab",
        "pdismiss", "pnhprev",  "logbmi", "bmi_msg", "agecat6", "year", "pyf_period_esrd",
        "t_start", "t_stop", "ashd1", "othcardiac1", "carfail", "noambul",  "pulmon",
        "notrans", "cancer", "diabetes", "pvasc", "cva",  "smoke", "alcoh", "drug", "inci_one", "inci_miss"]
z2 = d2[cols]

missing_z2 = z2.isna()

if sum(missing_z2.sum(axis=0))>0:
  print("Warning: There are", sum(missing_z2.sum(axis=0)), "rows with missing data")
  print("Any rows with missing data will be deleted")

allnames= ["ptnt_id","provfs","t_trans", "strr_pyf_start","wt_trans","ot_trans","pdiab","pdismiss",
    "pnhprev","logbmi","bmi_msg","agecat6","year","pyf_period_esrd","t_start","t_stop","ashd1",
    "othcardiac1", "carfail","noambul", "pulmon", "notrans", "cancer", "diabetes",
    "pvasc", "cva", "smoke","alcoh", "drug","inci_one","inci_miss",
    "strr_period", "pstrr", "trans_yar", "trans_dar"]


data_sub = d2[allnames]
data_sub_complete=data_sub

########################################################################################
#  SECTION 7 (ISSUE #3): CREATE TRANSFUSION EVENT FLAG                                 #
#    -CHECK FOR SMALL NUMBER OF EVENTS                                                 #
########################################################################################
data_sub_complete['t_trans0'] = np.where(data_sub_complete['t_trans']>0, 1, 0)
data_sub_complete['t_trans']=data_sub_complete['t_trans'].apply(lambda x: 1 if x > 0 else 0).copy()
# data_sub_complete.loc[data_sub_complete.t_trans > 1, 't_trans'] = 1
# data_sub_complete.loc[data_sub_complete.t_trans <= 0, 't_trans'] = 0

########################################################################################
#   SECTION 8: SUBSET DATA TO THOSE WITH AT LEAST 1 DAY AT RISK                        #
#     -ALSO SUBSET TO THOSE WITH APPROPRIATE AGE AND ESRD CATEGORIES                   #
#   NOTE: THIS IS MOSTLY DONE JUST TO BE PRECAUTIOUS FOR TEST DATA                     #
########################################################################################
data_sub_complete2 = data_sub_complete[(data_sub_complete['pyf_period_esrd']>0) & (data_sub_complete['agecat6']!=1) & (data_sub_complete['trans_dar']>0)]

########################################################################################
#   SECTION 9 (ISSUE #5): CHECK FOR LINEARLY DEPENDENT COVARIATES                      #
#      NOTE: THIS WON'T PREVENT MODEL FROM RUNNING                                     #
########################################################################################

z_pd = data_sub_complete2.to_records()
z_pd.sort(order=["provfs", "ptnt_id", "year", "strr_pyf_start"])
data_sub_sort = pd.DataFrame(z_pd)
data_chk_rank = data_sub_sort.drop(["strr_pyf_start","provfs","ptnt_id"], axis = 1)

########################################################################################
#   SECTION 10: RUN DATA THROUGH TWO COX PROPORTIONAL HAZARDS MODELS                   #
########################################################################################
data_sub_sort['year'] = pd.Categorical(data_sub_sort['year'], ordered=False)
data_sub_sort['pdismiss'] = pd.Categorical(data_sub_sort['pdismiss'], categories=[0, 1], ordered=False)
data_sub_sort['pnhprev'] = pd.Categorical(data_sub_sort['pnhprev'], categories=[0, 1], ordered=False)
data_sub_sort['bmi_msg'] = pd.Categorical(data_sub_sort['bmi_msg'], categories=[0, 1], ordered=False)
data_sub_sort['ashd1'] = pd.Categorical(data_sub_sort['ashd1'], categories=[0, 1], ordered=False)
data_sub_sort['othcardiac1'] = pd.Categorical(data_sub_sort['othcardiac1'], categories=[0, 1], ordered=False)
data_sub_sort['carfail'] = pd.Categorical(data_sub_sort['carfail'], categories=[0, 1], ordered=False)
data_sub_sort['noambul'] = pd.Categorical(data_sub_sort['noambul'], categories=[0, 1], ordered=False)
data_sub_sort['pulmon'] = pd.Categorical(data_sub_sort['pulmon'], categories=[0, 1], ordered=False)
data_sub_sort['notrans'] = pd.Categorical(data_sub_sort['notrans'], categories=[0, 1], ordered=False)
data_sub_sort['cancer'] = pd.Categorical(data_sub_sort['cancer'], categories=[0, 1], ordered=False)
data_sub_sort['diabetes'] = pd.Categorical(data_sub_sort['diabetes'], categories=[0, 1], ordered=False)
data_sub_sort['pvasc'] = pd.Categorical(data_sub_sort['pvasc'], categories=[0, 1], ordered=False)
data_sub_sort['cva'] = pd.Categorical(data_sub_sort['cva'], categories=[0, 1], ordered=False)
data_sub_sort['smoke'] = pd.Categorical(data_sub_sort['smoke'], categories=[0, 1], ordered=False)
data_sub_sort['alcoh'] = pd.Categorical(data_sub_sort['alcoh'], categories=[0, 1], ordered=False)
data_sub_sort['drug'] = pd.Categorical(data_sub_sort['drug'], categories=[0, 1], ordered=False)
data_sub_sort['inci_one'] = pd.Categorical(data_sub_sort['inci_one'], categories=[0, 1], ordered=False)
data_sub_sort['inci_miss'] = pd.Categorical(data_sub_sort['inci_miss'], categories=[0, 1], ordered=False)
data_sub_sort['smoke'] = pd.Categorical(data_sub_sort['smoke'], categories=[0, 1], ordered=False)

data_sub_sort['provfs'] = pd.Categorical(data_sub_sort['provfs'], ordered=False)

# coxph_control <- coxph.control(eps = 1e-8)
# data_sub_sort.transform(lambda x: x + 1)

data_model = data_sub_sort




#######################################################################################################################################################
# Run Stage-1 Cox Model
######################################################################################################################################################
if (False):
    import h2o
    from h2o.estimators.coxph import H2OCoxProportionalHazardsEstimator
    h2o.init()

    data_modelt_Full_Sample=data_model
    data_modelt_Full_Sample.reset_index(drop=True, inplace=True)

    data_modelt_Full_Sample_hex = h2o.H2OFrame(data_modelt_Full_Sample)
    data_modelt_Full_Sample_hex[data_modelt_Full_Sample_hex["provfs"].isna(), "provfs"] = 0
    # data_modelt_Full_Sample_hex["provfs"] = data_modelt_Full_Sample_hex["provfs"].ascharacter()
    # data_modelt_Full_Sample_hex["provfs"] = data_modelt_Full_Sample_hex["provfs"].asfactor()
    predictorsSt = list(["agecat6", "pdiab", "pdismiss","notrans","cancer",
                    "diabetes","pvasc","year", "pnhprev", "logbmi",
                    "bmi_msg","cva","smoke","alcoh","drug",
                    "inci_one", "ashd1", "pulmon","inci_miss","year",
                    "othcardiac1", "carfail", "noambul", "pulmon"])

    interaction_pairs = [   ("agecat6","pdiab"),
                            ("pdiab", "pyf_period_esrd")]

    strr_h2o_moodel = H2OCoxProportionalHazardsEstimator(
        start_column="t_start",
        stop_column="t_stop",
        offset_column="ot_trans",
        ties="breslow",
    #     stratify_by=["provfs"],
    #     interaction_pairs=interaction_pairs,
        )

    strr_h2o_moodel.train(x=predictorsSt,
                    y="t_trans0",
                    training_frame=data_modelt_Full_Sample_hex)



    ###########################
    # writing to csv
    # pd.DataFrame(data_modelt_Full_Sample_hex["provfs"]).to_csv('provfs.csv')

    data_modelt_Full_Sample_hex['provfs'].types

    strr_h2o_moodel.coefficients_table
    data_modelt_Full_Sample_hex['t_trans0'].types
    data_modelt_Full_Sample_hex["t_trans0"]
    len(data_modelt_Full_Sample_hex.col_names)
    data_modelt_Full_Sample_hex.types

if (True):
    from matplotlib import pyplot as plt
    from lifelines import CoxPHFitter
    import numpy as np
    import pandas as pd
    from lifelines.datasets import load_rossi

    rossi = load_rossi()
    cph = CoxPHFitter()

    cph.fit(rossi, 'week', 'arrest')

    # show rossi
    rossi

    # show summary
    cph.print_summary(model="untransformed variables", decimals=3)

    cph.check_assumptions(rossi, p_value_threshold=0.05, show_plots=True)

    from lifelines.statistics import proportional_hazard_test

    results = proportional_hazard_test(cph, rossi, time_transform='rank')
    results.print_summary(decimals=3, model="untransformed variables")

    # Stratification
    cph.fit(rossi, 'week', 'arrest', strata=['wexp'])
    cph.print_summary(model="wexp in strata")


if(True):
    # https://spark.apache.org/docs/latest/ml-classification-regression.html#survival-regression
    from pyspark.ml.regression import AFTSurvivalRegression
    from pyspark.ml.linalg import Vectors

    training = spark.createDataFrame([
        (1.218, 1.0, Vectors.dense(1.560, -0.605)),
        (2.949, 0.0, Vectors.dense(0.346, 2.158)),
        (3.627, 0.0, Vectors.dense(1.380, 0.231)),
        (0.273, 1.0, Vectors.dense(0.520, 1.151)),
        (4.199, 0.0, Vectors.dense(0.795, -0.226))], ["label", "censor", "features"])
    quantileProbabilities = [0.3, 0.6]
    aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                                quantilesCol="quantiles")

    model = aft.fit(training)

    # Print the coefficients, intercept and scale parameter for AFT survival regression
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))
    print("Scale: " + str(model.scale))
    model.transform(training).show(truncate=False)

