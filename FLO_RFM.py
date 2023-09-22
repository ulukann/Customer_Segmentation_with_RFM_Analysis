
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option("display.width",600)
pd.options.mode.chained_assignment = None

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()
df.describe([0,0.01, 0.05, 0.25, 0.50, 0.75,0.85, 0.95, 0.99, 1]).T
df_.describe([0,0.01, 0.05, 0.25, 0.50, 0.75,0.85, 0.95, 0.99, 1]).T

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

df["order_num_total_ever_online"].quantile(0.99)
df["order_num_total_ever_online"].quantile(0.01)
interquantile_range = df["order_num_total_ever_online"].quantile(0.99) - df["order_num_total_ever_online"].quantile(0.01)
up_limit = df["order_num_total_ever_online"].quantile(0.99) + 1.5 * interquantile_range
low_limit =df["order_num_total_ever_online"].quantile(0.01) - 1.5 * interquantile_range


columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)



df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.dtypes

df.describe([0,0.01, 0.05, 0.25, 0.50, 0.75,0.85, 0.95, 0.99, 1]).T


df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)


cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[ns]')).dt.days / 7
cltv_df["T_weekly"] = ((analysis_date- df["first_order_date"]).astype('timedelta64[ns]')).dt.days / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df.head()



bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df["exp_sales_3_month1"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])



cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]



ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

cltv_df.head()

cltv_df.sort_values("cltv",ascending=False)[:20]



cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.groupby("cltv_segment").agg({"cltv":["count","sum","mean","std"]})

