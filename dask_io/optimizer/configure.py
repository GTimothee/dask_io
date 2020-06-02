from dask_io.optimizer.optimizer import optimize_func
import dask 

import logging
logger = logging.getLogger(__name__)


# def enable_keep():
#     dask.config.set({
#         'optimizations': [keep_algorithm]
#     })


def enable_clustering(buffer_size, mem_limit=True):
    """ Activate cluster strategy.

    Arguments:
    ----------
        buffer_size: size of buffer for clustered reads/writes.
        sched_opti: enable memory constraint on scheduler.
    """
    if not mem_limit: 
        print("Warning: using clustered strategy without memory constraint on scheduler can lead to buffer overflows.")

    dask.config.set({
        'optimizations': [optimize_func],
        'io-optimizer': {
            'memory_available': buffer_size,
            'scheduler_opti': mem_limit
        }
    })


def disable_clustering():
    dask.config.set({
        'optimizations': list(),
        'io-optimizer': None
    })


def configure_dask(config):
    """ Apply configuration to dask to parameterize the optimization function.

    Arguments:
    ---------
        config: A CaseConfig object.
    """
    if not config:
        raise ValueError("Empty configuration object.")
    enable_clustering(config.buffer_size, sched_opti=config.scheduler_opti)


def split():
    # use one thread only if on hdd
    pass