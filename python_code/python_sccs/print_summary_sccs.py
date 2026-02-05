import pandas as pd

def print_summary_sccs(x, digits=4, signif_stars=True, **kwargs):
    """
    Print summary for SCCS model.
    
    Parameters:
    x : summary object
        Should have attributes like call, fail, n, nevent, na_action, coef, coefficients, conf_int, concordance.
    """
    if hasattr(x, 'call'):
        print("Call:")
        print(x.call)
        print()
    
    if hasattr(x, 'fail'):
        print(f"Coxreg failed. {x.fail}")
        return
    
    print(f"  n= {x.n}", end="")
    if hasattr(x, 'nevent'):
        print(f", number of events= {x.nevent}")
    else:
        print()
    
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