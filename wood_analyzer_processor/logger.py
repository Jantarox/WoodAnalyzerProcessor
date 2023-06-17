import logging

logger = logging.getLogger("WoodAnalyzerProcessor")
logger.setLevel(logging.INFO)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)

c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

c_handler.setFormatter(c_format)

logger.addHandler(c_handler)
