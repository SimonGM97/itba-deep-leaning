# -- coding: utf-8 --
from PyTradeX.config.params import Params
from PyTradeX.utils.general.logging_helper import get_logger
import time
from functools import wraps
from typing import Callable


# Get logger
LOGGER = get_logger(
    name=__name__,
    level=Params.log_params.get('level'),
    txt_fmt=Params.log_params.get('txt_fmt'),
    json_fmt=Params.log_params.get('json_fmt'),
    filter_lvls=Params.log_params.get('filter_lvls'),
    log_file=Params.log_params.get('log_file'),
    backup_count=Params.log_params.get('backup_count')
)


def _func_full_name(func: Callable):
    if not func.__module__:
        return func.__name__
    return "{a}.{b}".format(a=func.__module__, b=func.__name__)


def _human_readable_time(elapsed: float):
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)

    if hours > 0:
        return "{a} hour {b} min {c} sec".format(a=int(round(hours, 0)), b=mins, c=round(secs, 2))
    elif mins > 0:
        return "{a} min {b} sec".format(a=int(round(mins, 0)), b=round(secs, 2))
    elif secs >= 0.1:
        return "{a} sec".format(a=round(secs, 2))
    else:
        return "{a} ms".format(a=int(round(secs * 1000.0, 0)))


def timing(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t1
        
        LOGGER.info(
            "Run time: %s ran in %s",
            _func_full_name(func), 
            _human_readable_time(elapsed)
        )
        # print("Run time: {a} ran in {b}".format(a=_func_full_name(func), b=b))
        return result

    return wrapper