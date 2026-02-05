import pandas as pd
import numpy as np

def dummyvar(x, data=None, sep="", drop=True, fun=int, verbose=False):
    """
    Create dummy variables from a categorical variable.
    
    Parameters:
    x : str or array-like
        If str, the column name in data. If array-like, the variable values.
    data : pd.DataFrame, optional
        The dataframe containing the variable. Required if x is str.
    sep : str, default ""
        Separator for dummy variable names.
    drop : bool, default True
        Whether to drop the first level (reference category).
    fun : callable, default int
        Function to apply to the dummy matrix (e.g., int, float).
    verbose : bool, default False
        Whether to print verbose messages.
    
    Returns:
    dummies : pd.DataFrame
        The dummy variables.
    """
    if isinstance(x, str):
        if data is None:
            raise ValueError("data must be provided if x is a string")
        name = x
        x_vals = data[x]
    else:
        x_vals = pd.Series(x)
        name = "x"  # default name if not provided
    
    # Convert to categorical, handling NA
    if drop:
        x_cat = pd.Categorical(x_vals)
    else:
        # Include all levels, even if not present
        unique_vals = x_vals.dropna().unique()
        x_cat = pd.Categorical(x_vals, categories=unique_vals)
    
    if len(x_cat.categories) < 2:
        if verbose:
            print(f"Warning: {name} has only 1 level. Producing dummy variable anyway.")
        # Return a single column of 1s
        dummies = pd.DataFrame(np.ones(len(x_vals)), columns=[f"{name}{sep}{x_cat.categories[0]}"])
        return dummies
    
    # Create dummies using pd.get_dummies
    dummies = pd.get_dummies(x_cat, prefix=name, prefix_sep=sep, drop_first=drop, dtype=int)
    
    # Apply fun to the matrix
    dummies = dummies.applymap(fun)
    
    if verbose:
        print(f" {name}: {dummies.shape[1]} dummy variables created")
    
    return dummies