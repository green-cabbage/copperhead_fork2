import itertools
from functools import partial
import dask.bag as db

def parallelize(func, argset, client, parameters={}, seq=False):
# def parallelize(func, argset, client, parameters={}, seq=True):
    """
    `func`: the function to be executed in parallel.
    `argset`: a dictionary that contains a list of possible values for each of the `func` arguments.
    If there is only one possible value for some argument, pass it as a list with a single element.
    All combinations of arguments will be computed, and `func` will be executed for each combination.
    `client`: Dask client connected to a cluster of workers.
    `parameters`: global parameters shared by different methods across the framework, almost every
    function needs some of them.

    Set `seq=True` to force sequential processing for debugging.

    returns: a list containing outputs of `func` executed for each combination of arguments
    """
    print("parallelize start!")
    if "df" in argset.keys():
        given_npartition = len(argset["df"])
    else:
        given_npartition = None
    print(f"given_npartition: {given_npartition}")
    # prepare combinations of arguments
    argset = [
        dict(zip(argset.keys(), values))
        for values in itertools.product(*argset.values())
    ]
    print(f"type argset: {type(argset)}")
    print("argset done")
    if seq:
        # debug: run sequentially
        results = []
        for args in argset:
            # print(f"args: {args}")
            results.append(func(args, parameters))
    else:
        # run in parallel
        # map_futures = client.scatter(argset)
        # # # # print(f"map_futures: {map_futures}")
        # futures = client.map(partial(func, parameters=parameters), map_futures)
        # results = client.gather(futures)
        # # # results = client.submit(futures)
        # # # results = []
        # # print(f"futures: {futures}")
        # # results = client.gather(futures, asynchronous=True).compute()
        
        # # print(f"argset.keys(): {argset.keys()}")
        # # print(f"argset['df']: {argset['df']}")
        #---------------------------------------------------------------
        if given_npartition is None:
             b = db.from_sequence(argset, npartitions=given_npartition)
        else:
            npartitions_max = 125#100
            if given_npartition > npartitions_max:
                given_npartition = npartitions_max
            b = db.from_sequence(argset, npartitions=given_npartition) #100
        b = b.map(partial(func, parameters=parameters))
        results = b.compute()
        # 

    return results
