import math, copy

from dask_io.optimizer.utils.utils import numeric_to_3d_pos, _3d_to_numeric_pos
from dask_io.optimizer.cases.resplit_utils import *


def get_main_volumes(B, T):
    """ I- Get a dictionary associating volume indices to volume positions in the buffer.
    Indexing following the keep algorithm indexing in storage order.
    Position following pillow indexing for rectangles i.e. (bottom left corner, top right corner)

    Arguments:
    ----------
        B: buffer shape
        T: Theta prime shape -> Theta value for C_x(n) (see paper)
    """
    logger.debug("B: %s", B)
    logger.debug("T: %s", T)

    main_volumes = [
        Volume(1,
               (0,0,T[Axes.k.value]),
               (T[Axes.i.value], T[Axes.j.value], B[Axes.k.value]))]
    
    if B[Axes.j.value] == T[Axes.j.value]:
        return main_volumes

    main_volumes.append(Volume(2,
               (0, T[Axes.j.value], 0),
               (T[Axes.i.value], B[Axes.j.value], T[Axes.k.value])))
    main_volumes.append(Volume(3,
               (0, T[Axes.j.value], T[Axes.k.value]),
               (T[Axes.i.value], B[Axes.j.value], B[Axes.k.value])))
    
    if B[Axes.i.value] == T[Axes.i.value]:
        return main_volumes

    bottom_volumes = [
        Volume(4,
               (T[Axes.i.value], 0, 0),
               (B[Axes.i.value], T[Axes.j.value], T[Axes.k.value])),
        Volume(5,
               (T[Axes.i.value], 0, T[Axes.k.value]),
               (B[Axes.i.value], T[Axes.j.value], B[Axes.k.value])),
        Volume(6,
               (T[Axes.i.value], T[Axes.j.value], 0),
               (B[Axes.i.value], B[Axes.j.value], T[Axes.k.value])),
        Volume(7,
               (T[Axes.i.value], T[Axes.j.value], T[Axes.k.value]),
               (B[Axes.i.value], B[Axes.j.value], B[Axes.k.value]))
    ]
    return main_volumes + bottom_volumes


def compute_hidden_volumes(T, O):
    """ II- compute hidden output files' positions (in F0)

    Hidden volumes are output files inside the f0 volume (see paper).
    Those output files can be complete or uncomplete. 
    An uncomplete volume is some output file data that is not entirely contained in f0,
    such that it overlaps with an other buffer. 

    Arguments:
    ----------
        T: Theta shape for the buffer being treated (see paper)
        O: output file shape
    """
    # a) get volumes' graduation on x, y, z axes
    # i.e. find the crosses in the 2D example drawing below:
    #   
    # k
    # ↑
    # -----------------
    # |     f1     |f3|
    # x------------|--|      ▲  ▲ ---------- T[dim==k]
    # |            |  |      |  | ← O[k]
    # |   hidden   |f2|      |  |
    # x------------|  |      |  ▼
    # |   hidden   |  |      | ← Theta[k]
    # x------------x--- → j  ▼   

    points = list()
    for dim in range(3):
        points_on_axis = list()
        nb_hidden_volumes = T[dim]/O[dim] 
        nb_complete_vols = math.floor(nb_hidden_volumes) 

        a = T[dim]
        points_on_axis.append(a)
        for _ in range(nb_complete_vols):
            b = a - O[dim]
            points_on_axis.append(b)
            a = b

        if not 0 in points_on_axis:
            points_on_axis.append(0)

        points_on_axis.sort()
        points.append(points_on_axis)

    # b) compute volumes' corners (bottom left and top right) from points
    blc_index = [0,0,0] # bottom left corner index
    trc_index = [1,1,1] # top right corner index

    # -----------------
    # |     f1     |f3|
    # |---------trc2--|      ▲  ▲ ---------- T[dim==k]
    # |            |  |      |  | ← O[k]
    # |   hidden   |f2|      |  |
    # blc2------trc1  |      |  ▼
    # |   hidden   |  |      | ← Theta[k]
    # blc1-------------      ▼


    hidden_volumes = list()
    index = 7 # key of the volume in the dictionary of volumes, [1 -> 7 included] already taken
    for i in range(len(points[0])-1):
        for j in range(len(points[1])-1):
            for k in range(len(points[2])-1):
                logger.debug("blc_index: %s", blc_index)
                logger.debug("trc_index: %s", trc_index)
                corners = [(points[0][blc_index[0]], points[1][blc_index[1]], points[2][blc_index[2]]),
                           (points[0][trc_index[0]], points[1][trc_index[1]], points[2][trc_index[2]])]
                index += 1
                hidden_volumes.append(Volume(index, corners[0], corners[1]))

                blc_index[Axes.k.value] += 1
                trc_index[Axes.k.value] += 1
            blc_index[Axes.j.value] += 1
            trc_index[Axes.j.value] += 1
            blc_index[Axes.k.value] = 0
            trc_index[Axes.k.value] = 1
        blc_index[Axes.i.value] += 1
        trc_index[Axes.i.value] += 1
        blc_index[Axes.j.value] = 0
        trc_index[Axes.j.value] = 1
        blc_index[Axes.k.value] = 0
        trc_index[Axes.k.value] = 1
    return hidden_volumes

def add_offsets(volumes_list, _3d_index, B):
    """ III - Add offset to volumes positions to get positions in the reconstructed image.
    """
    offset = [B[dim] * _3d_index[dim] for dim in range(len(_3d_index))]
    for volume in volumes_list:
        volume.add_offset(offset)


def get_arrays_dict(buff_to_vols, buffers, outfiles_volumes):
    """ IV - Assigner les volumes à tous les output files, en gardant référence du type de volume que c'est
    """
    array_dict = dict()

    

    for buffer_index, buffer_volumes in buff_to_vols.items():
        crossed_outfiles = get_crossed_outfiles(buffer_index, buffers, outfiles_volumes) # refine search

        for volume in buffer_volumes:
            for outfile in crossed_outfiles:
                if included_in(volume, outfile):
                    add_to_array_dict(array_dict, outfile, volume)
                    break # a volume can belong to only one output file

    return array_dict


def merge_cached_volumes(arrays_dict, volumestokeep):
    """ V - Pour chaque output file, pour chaque volume, si le volume doit être kept alors fusionner
    """
    merge_rules = get_merge_rules(volumestokeep)

    for outfileindex in arrays_dict.keys():
        volumes = arrays_dict[outfileindex]
        
        for voltomerge_index in merge_rules.keys():
            for i in range(len(volumes)):
                if volumes[i].index == voltomerge_index:
                    logger.debug("nb volumes for outfile %s: %s", outfileindex, len(volumes))
                    logger.debug("merging volume %s", voltomerge_index)
                    volumetomerge = volumes.pop(i)
                    logger.debug("POP nb volumes for outfile %s: %s", outfileindex, len(volumes))
                    merge_directions = merge_rules[volumetomerge.index]
                    new_volume = apply_merge(volumetomerge, volumes, merge_directions)
                    logger.debug("BEFORE ADD NEW nb volumes for outfile %s: %s", outfileindex, len(volumes))
                    volumes.append(new_volume)
                    logger.debug("AFTER ADD NEW nb volumes for outfile %s: %s", outfileindex, len(volumes))
                    break

        arrays_dict[outfileindex] = volumes


def get_merge_rules(volumestokeep):
    """ Get merge rules corresponding to volumes to keep.
    See thesis for explanation of the rules.
    """
    rules = {
        1: [Axes.k] if 1 in volumestokeep else None,
        2: [Axes.j] if 2 in volumestokeep else None,
        3: [Axes.k] if 3 in volumestokeep else None,
        4: [Axes.i] if 4 in volumestokeep else None,
        5: [Axes.k] if 5 in volumestokeep else None,
        6: [Axes.j] if 6 in volumestokeep else None,
        7: [Axes.k, Axes.j] if 7 in volumestokeep else None
    }
    rules[3].append(Axes.j) if 2 in volumestokeep else None
    for i in [5,6,7]:
        rules[i].append(Axes.i) if 4 in volumestokeep else None
    for k in list(rules.keys()):
        if rules[k] == None:
            del rules[k]  # see usage in merge_cached_volumes
    return rules


def get_regions_dict(array_dict, outfiles_volumes):
    """ Create regions dict from arrays dict by removing output file offset (low corner) from slices.
    """
    regions_dict = dict()
    regions_dict = copy.deepcopy(array_dict)

    slice_to_list = lambda s: [s.start, s.stop, s.step]
    list_to_slice = lambda s: slice(s[0], s[1], s[2])

    for v in outfiles_volumes.values():
        p1 = v.p1 # (x, y, z)
        outputfile_data = regions_dict[v.index]

        for i in range(len(outputfile_data)):
            slices_list = outputfile_data[i]
            s1, s2, s3 = slices_list

            s1 = slice_to_list(s1) # start, stop, step
            s2 = slice_to_list(s2) # start, stop, step
            s3 = slice_to_list(s3) # start, stop, step
            slices_list = [s1, s2, s3]

            for dim in range(3):
                s = slices_list[dim]
                s[0] -= p1[dim]
                s[1] -= p1[dim]
                slices_list[dim] = list_to_slice(s)

            outputfile_data[i] = slices_list
    return regions_dict


def get_buff_to_vols(R, B, O, buffers_volumes, buffers_partition):
    """ Outputs a dictionary associating buffer_index to list of Volumes indexed as in paper.
    """
    buff_to_vols = dict()
    
    for buffer_index in buffers_volumes.keys():
        _3d_index = numeric_to_3d_pos(buffer_index, buffers_partition, order='F')
        
        T = list()
        for dim in range(len(buffers_volumes[buffer_index].p1)):
            C = ((_3d_index[dim]+1) * B[dim]) % O[dim]
            if C == 0 and B[dim] != O[dim]:  # particular case
                C = O[dim]
            T.append(B[dim] - C)
        volumes_list = get_main_volumes(B, T)  # get coords in basis of buffer
        volumes_list = volumes_list + compute_hidden_volumes(T, O)  # still in basis of buffer
        add_offsets(volumes_list, _3d_index, B)  # convert coords in basis of R
        buff_to_vols[buffer_index] = volumes_list
    return buff_to_vols


def compute_zones(B, O, R, volumestokeep):
    """ Main function of the module. Compute the "arrays" and "regions" dictionary for the resplit case.

    Arguments:
    ----------
        B: buffer shape
        O: output file shape
        R: shape of reconstructed image
        volumestokeep: volumes to be kept by keep strategy
    """
    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)
    outfiles_partititon = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partititon, O)

     # A/ associate each buffer to volumes contained in it
    buff_to_vols = get_buff_to_vols(R, B, O, buffers_volumes, buffers_partition)

    # B/ Create arrays dict from buff_to_vols
    # arrays_dict associate each output file to parts of it to be stored at a time
    arrays_dict = get_arrays_dict(buff_to_vols, buffers_volumes, outfiles_volumes) 
    merge_cached_volumes(arrays_dict, volumestokeep)
    clean_arrays_dict(arrays_dict)

    # C/ Create regions dict from arrays dict
    regions_dict = get_regions_dict(array_dict, outfiles_volumes)

    return arrays_dict, regions_dict