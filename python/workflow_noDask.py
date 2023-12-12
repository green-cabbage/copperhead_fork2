import itertools

def non_parallelize(func, argset, parameters={}, seq=True):
    """
    `func`: the function to be executed sequentially.
    `argset`: a dictionary that contains a list of possible values for each of the `func` arguments.
    If there is only one possible value for some argument, pass it as a list with a single element.
    All combinations of arguments will be computed, and `func` will be executed for each combination.
    `parameters`: global parameters shared by different methods across the framework, almost every
    function needs some of them.

    Set `seq=True` to force sequential processing for debugging.

    returns: a list containing outputs of `func` executed for each combination of arguments
    """

    # prepare combinations of arguments
    arg_combinations = [
        dict(zip(argset.keys(), values))
        for values in itertools.product(*argset.values())
    ]

    results = []

    for args in arg_combinations:
        results.append(func(args, parameters))

    return results