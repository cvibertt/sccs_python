import pandas as pd
from scipy.stats import chi2

def lrtsccs(model1, model2):
    """
    Likelihood ratio test for two SCCS models.
    
    Parameters:
    model1, model2 : fitted model objects
        Should have 'logtest' attribute with [loglik, df] or similar.
    
    Returns:
    result : DataFrame
        Test statistic, degrees of freedom, p-value.
    """
    # Assuming models have logtest as [loglik, df]
    test = abs(model1.logtest[0] - model2.logtest[0])
    df = abs(model1.logtest[1] - model2.logtest[1])
    pvalue = chi2.sf(test, df)
    
    result = pd.DataFrame({
        'test': [round(test, 4)],
        'df': [df],
        'pvalue': [round(pvalue, 4)]
    })
    return result