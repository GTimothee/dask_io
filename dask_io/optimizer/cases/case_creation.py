import glob, os
import h5py
import logging
import dask.array as da

from dask_io.optimizer.utils.get_arrays import get_dask_array_from_hdf5
from dask_io.optimizer.utils.array_utils import get_arr_shapes, get_arr_shapes
from dask_io.optimizer.utils.utils import add_to_dict_of_lists

logger = logging.getLogger(__name__)


def get_arr_chunks(arr, nb_chunks=None, as_dict=False):
    """ Return the list of the chunks of the input array.

    Arguments:
    ----------
        arr: dask array
        nb_chunks: if None then function returns all arrays, else function returns n=nb_chunks arrays

    Returns:
    --------
        a list of dask arrays.
    """
    data = get_arr_shapes(arr)
    if len(data) == 3:
        _, chunk_shape, dims = data
    elif len(data) == 4:
        _, chunk_shape, dims, _ = data
    else:
        raise ValueError()

    arr_list = list()
    positions = list()
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if (nb_chunks and len(arr_list) < nb_chunks) or not nb_chunks:
                    upper_corner = (i * chunk_shape[0],
                                    j * chunk_shape[1],
                                    k * chunk_shape[2])

                    arr_list.append(
                        arr[
                            upper_corner[0]: upper_corner[0] + chunk_shape[0], 
                            upper_corner[1]: upper_corner[1] + chunk_shape[1], 
                            upper_corner[2]: upper_corner[2] + chunk_shape[2]
                        ]
                    )
                    if as_dict:
                        positions.append((i,j,k))

    if as_dict:
        return dict(zip(positions, arr_list))
    else:
        return arr_list


def sum_chunks_case(arr, nb_chunks, compute=False):
    """ A test case where chunks of a dask array are summed up.

    Arguments:
    ----------
        arr: array from which blocks will be sum
        nb_chunks: number of chunks to sum
        compute: to compute directly
    """
    
    arr_list = get_arr_chunks(arr, nb_chunks)
    sum_arr = arr_list.pop(0)
    for a in arr_list:
        sum_arr = sum_arr + a

    if compute:
        return sum_arr.compute()
    return sum_arr


def split_to_hdf5(arr, f, nb_blocks):
    """ Split an array given its chunk shape. Output is a hdf5 file with as many datasets as chunks.
    
    Arguments:
    ----------
        arr: array to split
        f: an open hdf5 file to store data in it.
        nb_blocks: nb blocks we want to extract
    """
    arr_list = get_arr_chunks(arr, nb_blocks)
    datasets = list()

    for i, a in enumerate(arr_list):
        key = '/data' + str(i)
        logger.debug("creating chunk in hdf5 dataset -> dataset path: %s", key)
        logger.debug("storing chunk of shape %s", a.shape)
        datasets.append(f.create_dataset(key, shape=a.shape))

    return da.store(arr_list, datasets, compute=False)


def split_hdf5_multiple(arr, out_dirpath, nb_blocks, file_list):
    """
    Arguments:
    ----------
        arr: Array to split
        file_list: Empty list to store output files' objects.
        nb_blocks: Nb blocks we want to extract. None = all blocks.
    """
    arr_dict = get_arr_chunks(arr, nb_blocks, as_dict=True) # get array blocks as dask array objects

    datasets = list()
    arr_list = list()
    for key, arr_block in arr_dict.items():
        i, j, k = key
        filename = f'{i}_{j}_{k}.hdf5'
        filepath = os.path.join(out_dirpath, filename)
        if os.path.isfile(filepath):
                os.remove(filepath)
        file_list.append(h5py.File(filepath, 'w'))

        datasets.append(file_list[-1].create_dataset('/data', shape=arr_block.shape))
        arr_list.append(arr_block)
    return da.store(arr_list, datasets, compute=False)


def merge_hdf5_multiple(input_dirpath, out_filepath, out_file, dataset_key, store, rechunk_input):
    """ Merge separated hdf5 files into one hdf5 output file.
    
    Arguments: 
    ----------
        input_dirpath: path to input files
        out_filepath: path to output file
        out_file: empty pointer. will contain file object to be free after computations by Merge object.
        dataset_key: dataset key of the block stored into each input file
    """
    def print_blocks(l, depth):
        tab = depth * ['\t']
        if not isinstance(l, list):
            logger.info(''.join(tab) + '%s', l)
        else:
            logger.info(''.join(tab) + '[')
            for e in l:
                print_blocks(e, depth+1)
            logger.info(''.join(tab) + ']')

    # get array parts from input files
    workdir = os.getcwd()
    os.chdir(input_dirpath)
    data = dict()
    for infilepath in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        pos = infilepath.split('_')
        pos[-1] = pos[-1].split('.')[0]
        pos = tuple(list(map(lambda s: int(s), pos)))
        arr = get_dask_array_from_hdf5(infilepath, 
                                       dataset_key, 
                                       logic_cs="dataset_shape")
        if rechunk_input != None and isinstance(rechunk_input, tuple):
            arr = arr.rechunk(rechunk_input)
        data[pos] = arr
    os.chdir(workdir)

    if len(data.keys()) == 0:
        msg = 'Could not find input file matching regex'
        logger.error(msg)
        raise ValueError(msg)

    for pos in data.keys():
        logger.debug('%s', pos)

    # create reconstructed_array
    blocks = to_list(data)
    print_blocks(blocks, 0)
    reconstructed_array = da.block(blocks)

    if not store: 
        return reconstructed_array

    # store new array in output file
    out_file = h5py.File(out_filepath, 'w')
    dset = out_file.create_dataset('/data', shape=reconstructed_array.shape)
    return da.store(reconstructed_array, dset, compute=False)


def to_list(d):
    """ Order input dictionary by the first dimension of the position. 
    Takes a dictionary where:
        - each key is a list representing the position of block in reconstructed array.
        - each value is a block array (a dask array)
    
    See also:
        da.block
    """
    def add_to_dict_of_dicts(d, dim, pair):
        """ For each value in dimension dim we create a dictionary position: array
        d = dict
        dim = we sort in this dimension
        pair = (position: block) to add to 
        """
        if not dim in d.keys():
            d[dim] = dict()

        k, v = pair
        d[dim][k] = v


    if not isinstance(d, dict):
        return d
    
    keys = list()
    sorted_d = dict()
    for k, v in d.items():
        k = list(k)
        i = k.pop(0)
        keys.append(i)

        if len(k) == 0: # last dimension
            sorted_d[i] = v
        else:
            add_to_dict_of_dicts(sorted_d, i, (tuple(k), v))

    keys.sort()
    logger.debug('keys: %s', keys)
    max_key = keys[-1] + 1
    return [to_list(sorted_d[i]) for i in range(max_key)]