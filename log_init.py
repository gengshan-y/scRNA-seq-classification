import logging
import time

def make_logger(logpath):
    # root logger setting
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # clear handler streams
    for it in logger.handlers:
        logger.removeHandler(it)

    # file handler setting
    debug_handler = logging.FileHandler(logpath + time.strftime("%m_%d_%H_%M"))
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(debug_handler)
    print(logger.handlers[0].__dict__)

    return logger
