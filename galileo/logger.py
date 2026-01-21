# logger.py
import logging

# 1) Create a named logger for your application
logger = logging.getLogger("galileo")
logger.setLevel(logging.DEBUG)

# 2) Create and configure a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 3) Attach a simple formatter
fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
ch.setFormatter(fmt)

# 4) Add the handler to your logger
logger.addHandler(ch)