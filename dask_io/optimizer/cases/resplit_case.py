import math, copy

from dask_io.optimizer.utils.utils import numeric_to_3d_pos, _3d_to_numeric_pos, create_csv_file
from dask_io.optimizer.cases.resplit_utils import *

import logging 
logger = logging.getLogger(__name__)

def get_main_volumes(B, T):
    """ I- Get a dictionary associating volume indices to volume positions in the buffer.
    Indexing following the keep algorithm indexing in storage order.
    Position following pillow indexing for rectangles i.e. (bottom left corner, top right corner)

    Arguments:
    ----------
        B: buffer shape
        T: Theta prime shape -> Theta value for C_x(n) (see paper)
    """
    logger.debug("\t== Function == get_main_volumes")
    logger.debug("\t[Arg] T: %s", T)

    main_volumes = [
        Volume(1,
               (0,0,T[Axes.k.value]),
               (T[Axes.i.value], T[Axes.j.value], B[Axes.k.value]))]
    
    if B[Axes.j.value] == T[Axes.j.value]:
        logger.debug("\tEnd")
        return main_volumes

    main_volumes.append(Volume(2,
               (0, T[Axes.j.value], 0),
               (T[Axes.i.value], B[Axes.j.value], T[Axes.k.value])))
    main_volumes.append(Volume(3,
               (0, T[Axes.j.value], T[Axes.k.value]),
               (T[Axes.i.value], B[Axes.j.value], B[Axes.k.value])))
    
    if B[Axes.i.value] == T[Axes.i.value]:
        logger.debug("\tEnd")
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
    logger.debug("\tEnd")
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
    index = 7 # key of the volume in the dictionary of volumes (1 -> 7 included are already taken so keys begin at 8 and more)
    for i in range(len(points[0])-1):
        for j in range(len(points[1])-1):
            for k in range(len(points[2])-1):
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


def get_arrays_dict(buff_to_vols, buffers_volumes, outfiles_volumes, outfiles_partititon):
    """ IV - Assigner les volumes à tous les output files, en gardant référence du type de volume que c'est
    """
    array_dict = dict()

    miss = False
    for buffer_index, volumes_in_buffer in buff_to_vols.items():
        buffer_of_interest = buffers_volumes[buffer_index]
        # crossed_outfiles = get_crossed_outfiles(buffer_of_interest, outfiles_volumes) # refine search

        for volume_in_buffer in volumes_in_buffer:
            crossed=False
            for outfile_volume in outfiles_volumes.values(): # crossed_outfiles:
                if included_in(volume_in_buffer, outfile_volume):
                    add_to_array_dict(array_dict, outfile_volume, volume_in_buffer)
                    crossed=True
                    break # a volume can belong to only one output file
            if not crossed and not miss:
                miss = True
    if miss:
        print(f'WARNING: a volume has not been attributed to any outfile')
                
    # below lies a sanity check
    outfileskeys = list()
    for k, v in outfiles_volumes.items():
        outfileskeys.append(v.index)
    arraysdictkeys = list(array_dict.keys())
    missing_keys = set(outfileskeys) - set(arraysdictkeys)
    if not len(array_dict.keys()) == len(outfileskeys):
        print(f'len(array_dict.keys()): {len(arraysdictkeys)}')
        print(f'len(outfileskeys): {len(outfileskeys)}')
        print(f'nb missing keys: {len(missing_keys)}')
        raise ValueError("Something is wrong, not all output files will be written")
    return array_dict


def merge_cached_volumes(arrays_dict, volumestokeep):
    """ V - Pour chaque output file, pour chaque volume, si le volume doit être kept alors fusionner
    """
    logger.debug("== Function == merge_cached_volumes")
    merge_rules = get_merge_rules(volumestokeep)

    for outfileindex in sorted(list(arrays_dict.keys())):
        logger.debug("Treating outfile n°%s", outfileindex)
        volumes = arrays_dict[outfileindex]
        
        for voltomerge_index in merge_rules.keys():
            for i in range(len(volumes)):
                if volumes[i].index == voltomerge_index:
                    # logger.debug("nb volumes for outfile %s: %s", outfileindex, len(volumes))
                    # logger.debug("merging volume %s", voltomerge_index)
                    volumetomerge = volumes.pop(i)
                    # logger.debug("POP nb volumes for outfile %s: %s", outfileindex, len(volumes))
                    merge_directions = merge_rules[volumetomerge.index]
                    new_volume = apply_merge(volumetomerge, volumes, merge_directions)
                    # logger.debug("BEFORE ADD NEW nb volumes for outfile %s: %s", outfileindex, len(volumes))
                    volumes.append(new_volume)
                    # logger.debug("AFTER ADD NEW nb volumes for outfile %s: %s", outfileindex, len(volumes))
                    break

        arrays_dict[outfileindex] = volumes
        logger.debug("Associated outfile n°%s with list of volumes:", outfileindex)
        for v in volumes: 
            v.print()

    logger.debug("End\n")


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

            outputfile_data[i] = tuple(slices_list)
    return regions_dict


def split_main_volumes(volumes_list, O):
    """ Split the remainder volumes into volumes by the boundaries of the output files.
    """
    
    def get_dim_pts(bound, it, step, pts_list):
        bound = k_min
        it = k_max
        step = Ok
        pts_list = pts_k
        while it > bound:
            it -= step
            if it > bound:
                pts_list.append(it)
    
    def get_points(volume, O):
        upright_corner = volume.p2
        botleft_corner = volume.p1

        Oi, Oj, Ok = O
        i_max, j_max, k_max = upright_corner
        i_min, j_min, k_min = botleft_corner
        pts_i, pts_j, pts_k = [i_max, i_min], [j_max, j_min], [k_max, k_min]
        
        pts_i = get_dim_pts(i_min, i_max, Oi, pts_i)
        pts_j = get_dim_pts(j_min, j_max, Oj, pts_j)
        pts_k = get_dim_pts(k_min, k_max, Ok, pts_k)
        return (pts_i, pts_j, pts_k)

    def get_volumes_from_points(volume, points):
        i, j, k = volume.p1
        pts_i, pts_j, pts_k = points
        
        remainder_hid_vols = list()
        for i in range(len(pts_i)-1):
            for j in range(len(pts_j)-1):
                for k in range(len(pts_k)-1):
                    name = str(volume.index) + '_' + str(index)
                    botleft_corner = (pts_i[i], pts_j[j], pts_k[k])
                    upright_corner = (pts_i[i+1], pts_j[j+1], pts_k[k+1])
                    new_vol = Volume(name, p1, p2)
                    remainder_hid_vols.append(new_vol)
        
        return remainder_hid_vols

    split_volumes = list()
    for volume in volumes_list:
        points = get_points(volume, O)
        hid_vols = get_volumes_from_points(volume, points)
        split_volumes.extend(hid_vols)
    return split_volumes

def get_buff_to_vols(R, B, O, buffers_volumes, buffers_partition):
    """ Outputs a dictionary associating buffer_index to list of Volumes indexed as in paper.
    """

    def get_theta(buffers_volumes, buffer_index, _3d_index, O, B):
        T = list()
        Cs = list()
        for dim in range(len(buffers_volumes[buffer_index].p1)):
            if B[dim] < O[dim]:
                C = 0 
            else:            
                C = ((_3d_index[dim]+1) * B[dim]) % O[dim]
                print(f'{((_3d_index[dim]+1) * B[dim])}mod{O[dim]} = {C}')
                if C == 0 and B[dim] != O[dim]:  # particular case 
                    C = O[dim]

            if C < 0:
                raise ValueError("modulo should not return negative value")

            Cs.append(C)
            T.append(B[dim] - C)   
        print(f'C: {Cs}')
        print(f'theta: {T}')
        return T

    def first_sanity_check(buffers_volumes, buffer_index, volumes_list):
        """ see if volumes coordinates found are inside buffer
        """
        xs, ys, zs = list(), list(), list()
        for volume in volumes_list:
            x1, y1, z1 = volume.p1
            x2, y2, z2 = volume.p2 
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
            zs.append(z1)
            zs.append(z2)
        err = -1
        if not min(xs) == buffers_volumes[buffer_index].p1[0]:
            err = 0
        if not min(ys) == buffers_volumes[buffer_index].p1[1]:
            err = 1
        if not min(zs) == buffers_volumes[buffer_index].p1[2]:
            err = 2
        if not max(xs) == buffers_volumes[buffer_index].p2[0]:
            err = 3
        if not max(ys) == buffers_volumes[buffer_index].p2[1]:
            err = 4
        if not max(zs) == buffers_volumes[buffer_index].p2[2]:
            err = 5
        if err > -1:
            print(f'buffer lower corner: {buffers_volumes[buffer_index].p1}')
            print(f'volumes lower corner: {(min(xs), min(ys), min(zs))}')
            print(f'buffer upper corner: {buffers_volumes[buffer_index].p2}')
            print(f'volumes upper corner: {(max(xs), max(ys), max(zs))}')
            raise ValueError("Error ", err)

    def second_sanity_check(B, O, volumes_list):
        """ see if sum of all volumes equals the volume of the buffer 
        + see if each volume is <= volume of an output file as a volume cannot be bigger than an output file
        """
        volumes_volume = 0
        buffer_volume = B[0]*B[1]*B[2]
        outfile_volume = O[0]*O[1]*O[2]
        for volume in volumes_list:
            x1, y1, z1 = volume.p1
            x2, y2, z2 = volume.p2 
            vol = (x2-x1)*(y2-y1)*(z2-z1)

            if vol > outfile_volume:
                print(f'Outfile volume: {outfile_volume}')
                print(f'Volume considered: {vol}')
                raise ValueError("A volume should not be bigger than outfile")

            volumes_volume += vol

        if buffer_volume != volumes_volume:
            print(f'Buffer volume: {buffer_volume}')
            print(f'Sum of volumes: {volumes_volume}')
            raise ValueError("sum of volumes should be equal to buffer volume")

    logger.debug("== Function == get_buff_to_vols")
    buff_to_vols = dict()
    
    rows = list()
    for buffer_index in buffers_volumes.keys():
        print(f'\nProcessing buffer {buffer_index}')
        buffers_volumes[buffer_index].print()
        _3d_index = numeric_to_3d_pos(buffer_index, buffers_partition, order='F')

        T = get_theta(buffers_volumes, buffer_index, _3d_index, O, B)
        volumes_list = get_main_volumes(B, T)  # get coords in basis of buffer
        volumes_list = split_main_volumes(volumes_list, O) # seek for hidden volumes in main volumes
        volumes_list = volumes_list + compute_hidden_volumes(T, O)  # still in basis of buffer TODO: change naming -> add 0 as prefix
        add_offsets(volumes_list, _3d_index, B)  # convert coords in basis of R

        # debug 
        for v in volumes_list:
            v.print()

        first_sanity_check(buffers_volumes, buffer_index, volumes_list)
        second_sanity_check(B, O, volumes_list)

        buff_to_vols[buffer_index] = volumes_list
        
        # debug csv file
        for v in volumes_list:
            rows.append((
                (v.p1[1], v.p1[2]),
                v.p2[1] - v.p1[1],
                v.p2[2] - v.p1[2],
            ))
            
    # debug csv file
    columns = [
        'bl_corner',
        'width',
        'height'
    ]
    csv_path = '/tmp/compute_zones_buffervolumes.csv'
    csv_out, writer = create_csv_file(csv_path, columns, delimiter=',', mode='w+')
    for row in set(rows): 
        writer.writerow(row)
    csv_out.close()

    logger.debug("End\n")
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
    logger.debug("\n\n-----------------Compute zones [main file function]-----------------\n\n")
    logger.debug("Getting buffer partitions and buffer volumes")
    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)
    print(f'Buffers found:')
    for i, buff in buffers_volumes.items():
        buff.print()
    logger.debug("Getting output files partitions and buffer volumes")
    outfiles_partititon = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partititon, O)

    print(f'buffers partition: {buffers_partition}')
    print(f'outfiles partition: {outfiles_partititon}')

    # A/ associate each buffer to volumes contained in it
    buff_to_vols = get_buff_to_vols(R, B, O, buffers_volumes, buffers_partition)

    # B/ Create arrays dict from buff_to_vols
    # arrays_dict associate each output file to parts of it to be stored at a time
    arrays_dict = get_arrays_dict(buff_to_vols, buffers_volumes, outfiles_volumes, outfiles_partititon) 
    merge_cached_volumes(arrays_dict, volumestokeep)

    logger.debug("Arrays dict before clean:")
    for k in sorted(list(arrays_dict.keys())):
        v = arrays_dict[k]
        logger.debug("key %s", k)
        for e in v:
            e.print()
    logger.debug("---\n")

    clean_arrays_dict(arrays_dict)

    logger.debug("Arrays dict after clean:")
    for k in sorted(list(arrays_dict.keys())):
        v = arrays_dict[k]
        logger.debug("key %s", k)
        for e in v:
            logger.debug("\t%s", e)
    logger.debug("---\n")

    # C/ Create regions dict from arrays dict
    regions_dict = get_regions_dict(arrays_dict, outfiles_volumes)
    logger.debug("Regions dict:")
    for k in sorted(list(regions_dict.keys())):
        v = regions_dict[k]
        logger.debug("key %s", k)
        for e in v:
            logger.debug("\t%s", e)
    logger.debug("---\n")

    logger.debug("-----------------End Compute zones-----------------")
    return arrays_dict, regions_dict