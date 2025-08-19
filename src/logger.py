import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d')}.log"
logs_path=os.path.join(os.getcwd(), 'logs', LOG_FILE)#creates filename
#getcwd() gets the current working directory
os.makedirs(logs_path,exist_ok=True) #creates the directory if it does not exist
LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE) #creates the full path for the log file

#overrite the default logger, you need to set config
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
