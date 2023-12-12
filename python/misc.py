import numpy as np
import awkward as ak


def onedimeval(func, *arrays, tonumpy=True, output_like=0):
    """Evaluate a function on the flattened one dimensional version of arrays

    Parameters
    ----------
    func
        Function to execute
    *arrays
        Arrays that will be planned and fed into ``func``
    tonumpy
        Whether to convert the flattened arrays to numpy arrays
    output_like
        Position of the array in ``arrays`` to take parameters and behavior
        from

    Returns
    -------
        An unflattened version of the array that was returned by ``func``.
        Its behavior and parameters are set according to the array pointed to
        by ``output_like``
    """
    counts_all_arrays = []
    flattened_arrays = []
    for array in arrays:
        flattened = array
        counts = []
        for i in range(flattened.ndim - 1):
            if isinstance(flattened.type.type, ak.types.RegularType):
                counts.append(flattened.type.type.size)
            else:
                counts.append(ak.num(flattened))
            flattened = ak.flatten(flattened)
        if tonumpy:
            flattened = np.asarray(flattened)
        counts_all_arrays.append(counts)
        flattened_arrays.append(flattened)
    res = func(*flattened_arrays)
    for count in reversed(counts_all_arrays[output_like]):
        res = ak.unflatten(res, count)
    for name, val in ak.parameters(arrays[output_like]).items():
        res = ak.with_parameter(res, name, val)
    #res.behavior = arrays[output_like].behavior
    return res
