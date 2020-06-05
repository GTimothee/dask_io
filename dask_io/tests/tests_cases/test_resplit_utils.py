from dask_io.optimizer.cases.resplit_utils import *
from dask_io.optimizer.utils.utils import _3d_to_numeric_pos


def test_Volume_add_offset():
    v1 = Volume(1, (1,2,3), (4,5,6))
    v1.add_offset((5, 2, 3))
    assert v1.p1 == (6, 4, 6)
    assert v1.p2 == (9, 7, 9)


def test_Volume_get_corners():
    v1 = Volume(1, (1,2,3), (4,5,6))
    p1, p2 = v1.get_corners()
    assert p1 == (1,2,3)
    assert p2 == (4,5,6)


def test_hypercubes_overlap():
    hypercube1 = Volume(None, (0,0,0), (5,5,5))
    non_overlapping = Volume(None, (3,3,6), (8,8,11))
    overlapping = Volume(None, (3,3,3), (8,8,8))
    assert hypercubes_overlap(hypercube1, non_overlapping) == False
    assert hypercubes_overlap(hypercube1, overlapping) == True


def test_get_blocks_shape():
    big_arrays = [
        (10, 10, 10),
        (24, 5, 50),
        (100, 72, 16)
    ]
    small_arrays = [
        (5, 2, 5),
        (6, 5, 10),
        (10, 9, 4)
    ]
    expected = [
        (2, 5, 2),
        (4, 1, 5),
        (10, 8, 4)
    ]

    for big_array, small_array, blocks in zip(big_arrays, small_arrays, expected):
        assert get_blocks_shape(big_array, small_array) == blocks


def test_Volume_equals():
    v1 = Volume(1, (1,2,3), (4,5,6))
    v2 = Volume(1, (1,2,3), (4,5,6))
    assert v1.equals(v2) == True


def test__3d_to_numeric_pos():
    table_F_order = {
        (0,0,0): 0,
        (0,0,1): 1,
        (0,1,0): 2,
        (0,1,1): 3,
        (1,0,0): 4,
        (1,0,1): 5,
        (1,1,0): 6,
        (1,1,1): 7,
    }

    # table_C_order = { TODO
    #     (0,0,0): 0,
    #     (0,0,1): 4,
    #     (0,1,0): 2,
    #     (0,1,1): 6,
    #     (1,0,0): 1,
    #     (1,0,1): 5,
    #     (1,1,0): 3,
    #     (1,1,1): 7,
    # }

    for _3d_pos, numeric_pos in table_F_order.items():
        res = _3d_to_numeric_pos(_3d_pos, (2,2,2), order='F')
        assert res == numeric_pos

    # for _3d_pos, numeric_pos in table_C_order.items():
    #     res = _3d_to_numeric_pos(_3d_pos, (2,2,2), order='F')
    #     assert res == numeric_pos


def test_get_named_volumes():
    R = (60, 40, 30)
    B = (30, 20, 15)
    blocks_partition = get_blocks_shape(R, B)
    buffers = get_named_volumes(blocks_partition, B)
    expected = {
        0: Volume(0, (0,0,0), (30, 20, 15)),
        1: Volume(1, (0,0,15), (30, 20, 30)),
        2: Volume(2, (0,20,0), (30, 40, 15)),
        3: Volume(3, (0,20,15), (30, 40, 30)),
        4: Volume(4, (30,0,0), (60, 20, 15)),
        5: Volume(5, (30,0,15), (60, 20, 30)),
        6: Volume(6, (30,20,0), (60, 40, 15)),
        7: Volume(7, (30,20,15), (60, 40, 30)), 
    }
    for k in expected.keys():
        assert expected[k].equals(buffers[k])


# def test_get_crossed_outfiles():
#     R = (120, 120, 1)
#     B = (60, 60, 1)
#     O = (40, 40, 1)
#     buffers_shape = get_blocks_shape(R, B)
#     outfiles_shape = get_blocks_shape(R, O)
#     buffers = get_named_volumes(buffers_shape, B)
#     outfiles = get_named_volumes(outfiles_shape, O)

#     expected = {
#         0: [0, 1, 3, 4],
#         1: [1, 2, 4, 5],
#         2: [3, 4, 6, 7],
#         3: [4, 5, 7, 8]
#     }

#     for buffer_index in range(4):
#         crossed = get_crossed_outfiles(buffer_index, buffers, outfiles)
#         indices = [v.index for v in crossed]
#         assert set(expected[buffer_index]) == set(indices)


def test_merge_volumes():
    v1 = Volume(0, (0,40,1), (40,60,1))
    v2 = Volume(1, (0,60,1), (40,80,1))
    v3 = merge_volumes(v1, v2)
    assert v3.p1 == (0,40,1)
    assert v3.p2 == (40,80,1)


def test_convert_Volume_to_slices():
    v = Volume(0, (0,40,1), (40,60,1))
    expected = (slice(0, 40, None), slice(40, 60, None), slice(1, 1, None))
    assert convert_Volume_to_slices(v) == expected


def test_apply_merge():
    def assertion(res_vol, p1, p2):
        assert p1 == res_vol.p1
        assert p2 == res_vol.p2

    volumes = [
        Volume(0, (0,0,0), (5,5,5)),
        Volume(1, (0,0,5), (5,5,10)),
        Volume(2, (0,5,0), (5,10,5)),
        Volume(3, (0,5,5), (5,10,10)),
        Volume(4, (5,0,0), (10,5,5)),
        Volume(5, (5,0,5), (10,5,10)),
        Volume(6, (5,5,0), (10,10,5)),
        Volume(7, (5,5,5), (10,10,10)),
    ]
    
    import copy 
    assertion(apply_merge(volumes[0], copy.deepcopy(volumes), [Axes.k]), (0,0,0), (5,5,10))
    assertion(apply_merge(volumes[0], copy.deepcopy(volumes), [Axes.j]), (0,0,0), (5,10,5))
    assertion(apply_merge(volumes[0], copy.deepcopy(volumes), [Axes.i]), (0,0,0), (10,5,5))
    assertion(apply_merge(volumes[0], copy.deepcopy(volumes), [Axes.j, Axes.k]), (0,0,0), (5,10,10))
    assertion(apply_merge(volumes[0], copy.deepcopy(volumes), [Axes.j, Axes.i]), (0,0,0), (10,10,5))
    assertion(apply_merge(volumes[0], copy.deepcopy(volumes), [Axes.i, Axes.k]), (0,0,0), (10,5,10))
    assertion(apply_merge(volumes[0], copy.deepcopy(volumes), [Axes.i, Axes.j, Axes.k]), (0,0,0), (10,10,10))