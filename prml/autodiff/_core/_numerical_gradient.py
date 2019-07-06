from prml.autodiff._math._sum import sum


def numerical_gradient(func, eps, *args):
    """
    compute numerical gradient

    Parameters
    ----------
    func : callable
        returns scalar value
    eps : float
        small fluctuation

    Returns
    -------
    list
        list of partial derivatives
    """
    dargs = []
    for i in range(len(args)):
        args_p = []
        args_m = []
        for j, arg in enumerate(args):
            if i == j:
                args_p.append(arg + eps)
                args_m.append(arg - eps)
            else:
                args_p.append(arg)
                args_m.append(arg)
        f_p = func(*args_p)
        f_m = func(*args_m)
        dargs.append((f_p - f_m) / (2 * sum(eps)))
    return dargs
