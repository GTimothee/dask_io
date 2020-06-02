import os
import sys
import math
import time
import datetime 
import logging

from dask_io.optimizer.clustering import apply_clustered_strategy
from dask_io.optimizer.find_proxies import get_used_proxies, get_array_block_dims

now = datetime.datetime.now()
date_info = now.strftime("%c")
current_dir = os.path.dirname(os.path.abspath(__file__))
logfilename = "/tmp/dask_io_" + date_info + ".log"
# logging.config.fileConfig(os.path.join(current_dir, 'logging_config.ini'), defaults={'logfilename': logfilename}, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def clustered_optimization(graph):
    """ Applies clustered IO optimization on a Dask graph.

    Arguments:
    ----------
        graph : dark_array.dask.dicts
    """
    logger.info(f"Configuration file is at {os.path.join(current_dir, 'logging_config.ini')}")
    logger.info("Log file: %s", logfilename)
    logger.info("Finding proxies.")
    chunk_shape, dicts = get_used_proxies(graph)

    if chunk_shape == None or dicts == None:
        logger.error("Chunk shape or dicts = None. Aborting dask_io optimization.")
        raise ValueError()

    logger.info("Launching optimization algorithm.") 
    apply_clustered_strategy(graph, dicts, chunk_shape)
    return graph


def optimize_func(dsk, keys): # TODO: change function name
    """ Apply an optimization on the dask graph.

    Arguments:
    ----------
        dsk: dask graph
        keys: 

    Returns: 
    ----------
        the optimized dask graph
    """
    t = time.time()
    dask_graph = dsk.dicts
    dask_graph = clustered_optimization(dask_graph)
    logger.info("Time spent to create the graph: {0:.2f} milliseconds.".format((time.time() - t) * 1000))

    log_file_path = os.path.join('/tmp', 'dask_io_output_graph.log')
    logger.info('Output graph log can be found at %s', log_file_path)
    with open(log_file_path, "w+") as f:
        for k, v in dask_graph.items():    
            f.write("\n\n " + str(k))
            f.write("\n" + str(v))
    return dsk


# def keep_algorithm(dsk, keys):
#     print("Inside the keep algorithm")
#     dask_graph = dsk.dicts
#     chunk_shape, dicts = get_used_proxies(dask_graph)
#     return dsk