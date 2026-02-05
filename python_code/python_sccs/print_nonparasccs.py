def print_nonparasccs(x, digits=None, signif_stars=None, **kwargs):
    """
    Print summary for non-parametric SCCS model.
    
    Parameters:
    x : fitted model object
        Should have attributes: lambda1, cv1, lambda2, cv2.
    """
    print("Non parametric self controlled case series")
    print(f"Age related relative incidence function:")
    print(f"Smoothing parameter = {x.lambda1:.2e}")
    print(f"Cross validation score = {x.cv1:.2f}")
    print()
    print("Exposure related relative incidence function:")
    print(f"Smoothing parameter = {x.lambda2:.2e}")
    print(f"Cross validation score = {x.cv2:.2f}")