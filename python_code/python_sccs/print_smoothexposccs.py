import pandas as pd

def print_smoothexposccs(x, digits=4, signif_stars=True, **kwargs):
    """
    Print summary for smoothed exposure SCCS model.
    
    Parameters:
    x : fitted model object or summary
        Should have attributes like coefficients, conf.int, concordance, smp, crossvalidation, etc.
    """
    # Assume x is already summarized
    if hasattr(x, 'summary'):
        x = x.summary()
    
    if hasattr(x, 'call'):
        print("Call:")
        print(x.call)
        print()
    
    if hasattr(x, 'fail'):
        print(f"Coxreg failed. {x.fail}")
        return
    
    if hasattr(x, 'nevent'):
        print(f", number of events= {x.nevent}")
    
    if hasattr(x, 'na_action') and x.na_action:
        print(f"   ({len(x.na_action)} observations deleted due to missingness)")
    
    if hasattr(x, 'coef') and len(x.coef) == 0:
        print("   Null model")
        return
    
    if hasattr(x, 'coefficients'):
        print()
        print(x.coefficients.round(digits))
    
    if hasattr(x, 'conf_int'):
        print()
        print(x.conf_int)
    
    print()
    
    if hasattr(x, 'concordance'):
        print(f"Concordance= {x.concordance[0]:.3f} (se = {x.concordance[1]:.3f})")
    
    print("Spline based exposure relative incidence function:")
    print(f"Smoothing parameter = {x.smp}")
    print(f"Cross validation score = {x.crossvalidation:.2f}")