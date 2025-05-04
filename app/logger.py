# logger.py
import logging

# Configura el logger
logger = logging.getLogger("plataforma_logger")
logger.setLevel(logging.INFO)

# Crea un handler para escribir en archivo
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)

# Formato del log
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Añade el handler al logger si no existe
if not logger.handlers:
    logger.addHandler(file_handler)
