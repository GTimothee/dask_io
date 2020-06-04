from enum import Enum
import operator

from dask_io.optimizer.utils.utils import _3d_to_numeric_pos

import logging 
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Axes(Enum):
    i = 0
    j = 1
    k = 2


class Volume:
    def __init__(self, index, p1, p2):
        if (not isinstance(p1, tuple) 
            or not isinstance(p2, tuple)):
            raise TypeError()

        self.index = index
        self.p1 = p1  # bottom left corner
        self.p2 = p2  # top right corner


    def add_offset(self, offset):
        """
        offset: a tuple
        """
        self.p1 = self._add_offset(self.p1, offset)
        self.p2 = self._add_offset(self.p2, offset)
            

    def _add_offset(self, p, offset):
        if isinstance(offset, list):
            offset = tuple(offset)
        elif not isinstance(offset, tuple):
            raise TypeError("Expected tuple")
        return tuple(map(operator.add, p, offset))


    def get_corners(self):
        return (self.p1, self.p2)


    def equals(self, volume):
        if not self.index == volume.index:
            return False 
        if not self.p1 == volume.p1:
            return False 
        if not self.p2 == volume.p2:
            return False 
        return True

    def print(self):
        logger.debug("\tVolume name: %s, p1: %s, p2: %s", self.index, self.p1, self.p2)


def hypercubes_overlap(hypercube1, hypercube2):
    """ Evaluate if two hypercubes cross each other.
    """
    if not isinstance(hypercube1, Volume) or \
        not isinstance(hypercube2, Volume):
        raise TypeError()

    lowercorner1, uppercorner1 = hypercube1.get_corners()
    lowercorner2, uppercorner2 = hypercube2.get_corners()
    nb_dims = len(uppercorner1)
    
    for i in range(nb_dims):
        if not uppercorner1[i] > lowercorner2[i] or \
            not uppercorner2[i] > lowercorner1[i]:
            return False

    return True


def get_blocks_shape(big_array, small_array):
    """ Return the number of small arrays in big array in all dimensions as a shape. 
    """
    return tuple([int(b/s) for b, s in zip(big_array, small_array)])


def get_crossed_outfiles(buffer_of_interest, outfiles_volumes):
    """ Returns list of output files that are crossing buffer at buffer_index.

    Arguments: 
    ----------
        outfiles_volumes: dict of volumes representing the output files, indexed in storage order.
    """
    crossing = list()
    for outfile in outfiles_volumes.values():
        if hypercubes_overlap(buffer_of_interest, outfile):
            crossing.append(outfile)  # we add a Volume obj
    return crossing


def merge_volumes(volume1, volume2):
    """ Merge two volumes into one.
    """
    if not isinstance(volume1, Volume) or \
        not isinstance(volume2, Volume):
        raise TypeError()

    lowercorner1, uppercorner1 = volume1.get_corners()
    lowercorner2, uppercorner2 = volume2.get_corners()
    lowercorner = (min(lowercorner1[0], lowercorner2[0]), 
                   min(lowercorner1[1], lowercorner2[1]),
                   min(lowercorner1[2], lowercorner2[2]))
    uppercorner = (max(uppercorner1[0], uppercorner2[0]), 
                   max(uppercorner1[1], uppercorner2[1]),
                   max(uppercorner1[2], uppercorner2[2]))
    return Volume(None, lowercorner, uppercorner)


def included_in(volume, outfile):
    """ Alias of hypercubes_overlap. 
    We do not verify that it is included but by definition
    of the problem if volume crosses outfile then volume in outfile.

    Arguments: 
    ----------
        volume: Volume in buffer
        outfile: Volume representing an output file
    """
    return hypercubes_overlap(volume, outfile)


def add_to_array_dict(array_dict, outfile, volume):
    """ Add volume information to dictionary associating output file index to 

    Arguments:
    ----------
        outfile: outfile volume
        volume: volume from buffer
    """
    if (not isinstance(outfile.index, int) 
        or not isinstance(volume, Volume) 
        or not isinstance(outfile, Volume)):
        raise TypeError()

    if not outfile.index in array_dict.keys():
        array_dict[outfile.index] = list()
    array_dict[outfile.index].append(volume)


def convert_Volume_to_slices(v):
    if not isinstance(v, Volume):
        raise TypeError()
    p1, p2 = v.get_corners()
    return tuple([slice(p1[dim], p2[dim], None) for dim in range(len(p1))])


def clean_arrays_dict(arrays_dict):
    """ From a dictionary of Volumes, creates a dictionary of list of slices.
    The new arrays_dict associates each output file to each volume that must be written at a time.
    """
    for k in arrays_dict.keys():
        volumes_list = arrays_dict[k]
        arrays_dict[k] = [convert_Volume_to_slices(v) for v in volumes_list]


def get_named_volumes(blocks_partition, block_shape):
    """ Return the coordinates of all entities of shape block shape in the reconstructed image.
    The first entity is placed at the origin of the base.

    Returns: 
    ---------
        d: dictionary mapping each buffer numeric index to a Volume representing its coordinates

    Arguments: 
    ----------
        blocks_partition: Number of blocks in each dimension. Shape of the reconstructed image in terms of the blocks considered.
        block_shape: shape of one block, all blocks having the same shape 
    """
    logger.debug("== Function == get_named_volumes")
    d = dict()
    logger.debug("[Arg] blocks_partition: %s", blocks_partition)
    logger.debug("[Arg] block_shape: %s", block_shape)
    for i in range(blocks_partition[0]):
        for j in range(blocks_partition[1]):
            for k in range(blocks_partition[2]):
                bl_corner = (block_shape[0] * i,
                             block_shape[1] * j,
                             block_shape[2] * k)
                tr_corner = (block_shape[0] * (i+1),
                             block_shape[1] * (j+1),
                             block_shape[2] * (k+1))   
                index = _3d_to_numeric_pos((i, j, k), blocks_partition, order='F')
                d[index] = Volume(index, bl_corner, tr_corner)
    logger.debug("Indices of names volumes found: %s", d.keys())
    logger.debug("End\n")
    return d


def apply_merge(volume, volumes, merge_directions):
    """ Merge volume with other volumes from volumes list in the merge directions.

    Arguments: 
    ----------
        volume: volume to merge
        volumes: list of volumes 
        merge_directions: indicates neighbours to merge with
    """
    
    def get_new_volume(volume, lowcorner):
        v2 = get_volume(lowcorner)
        if v2 != None:
            return merge_volumes(volume, v2)
        else:
            return volume

    def get_volume(lowcorner):
        if not isinstance(lowcorner, tuple):
            raise TypeError()  # required for "=="

        for i in range(len(volumes)):
            v = volumes[i]
            if v.p1 == lowcorner:
                logger.debug("\tMerging volume with low corner %s", v.p1)
                return volumes.pop(i)
        
        logger.warning("\tNo volume to merge with")
        return None

    import copy

    logger.debug("\t== Function == apply_merge")

    p1, p2 = volume.get_corners()
    logger.debug("\tTargetting volume with low corner %s", p1)

    if len(merge_directions) == 1:
        if Axes.k in merge_directions:
            p1_target = list(copy.deepcopy(p1))
            p1_target[Axes.k.value] = p2[Axes.k.value]
            new_volume = get_new_volume(volume, tuple(p1_target))

        elif Axes.j in merge_directions:
            p1_target = list(copy.deepcopy(p1))
            p1_target[Axes.j.value] = p2[Axes.j.value]
            new_volume = get_new_volume(volume, tuple(p1_target))

        elif Axes.i in merge_directions:
            p1_target = list(copy.deepcopy(p1))
            p1_target[Axes.i.value] = p2[Axes.i.value]
            new_volume = get_new_volume(volume, tuple(p1_target))

    elif len(merge_directions) == 2:
        logger.debug("\tMerge directions: %s", merge_directions)
        axis1, axis2 = merge_directions

        p1_target = list(copy.deepcopy(p1))
        p1_target[axis1.value] = p2[axis1.value]
        volume_axis1 = get_new_volume(volume, tuple(p1_target))

        new_volume_axis1 = apply_merge(volume_axis1, volumes, [axis2])
        new_volume_axis2 = apply_merge(volume, volumes, [axis2])
        new_volume = merge_volumes(new_volume_axis1, new_volume_axis2)

    elif len(merge_directions) == 3:
        logger.debug("\tMerge directions %s", merge_directions)
        axis1, axis2, axis3 = merge_directions
        
        p1_target = list(copy.deepcopy(p1))
        p1_target[axis1.value] = p2[axis1.value]
        volume_axis1 = get_new_volume(volume, tuple(p1_target))

        new_vol1 = apply_merge(volume, volumes, [axis2, axis3])
        new_vol2 = apply_merge(volume_axis1, volumes, [axis2, axis3])
        new_volume = merge_volumes(new_vol1, new_vol2)

    else:
        raise ValueError()

    logger.debug("\tEnd")
    return new_volume