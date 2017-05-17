import logging



_formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)d %(message)s')
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)

logger = logging.getLogger('tensorlab')
logger.addHandler(_stream_handler)
logger.setLevel(logging.INFO)


def set_formatter(formater): _stream_handler.setFormatter(formater)

def set_level(lvl): logger.setLevel(lvl)