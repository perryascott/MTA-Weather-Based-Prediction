from statsmodels.tsa.seasonal import MSTL

def perform_mstl(df, periods):
    mstl_result = MSTL(df, periods=periods).fit()
    return mstl_result