import os
import logging
import getpass

class MyLog(object):
    def __init__(self, log_file_name, username = "Jyunyu", enable_log = False):
        # user=getpass.getuser()
        self.logger=logging.getLogger(username)
        self.logger.setLevel(logging.DEBUG)
        format='%(asctime)s - %(levelname)s : %(message)s'
        datefmt='%m-%d %H:%M'
        formatter=logging.Formatter(format)
        streamhandler=logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        self.logger.addHandler(streamhandler)

        # for debug mode , we do not want to save logger file
        if enable_log:
            log_dir = os.path.join(os.getcwd(),"log")
            self.check_dir(log_dir)
            logfile=os.path.join(log_dir,log_file_name)
            filehandler=logging.FileHandler(logfile)
            filehandler.setFormatter(formatter)
            self.logger.addHandler(filehandler)
    def debug(self, msg):
        self.logger.debug(msg)
    def info(self, msg):
        self.logger.info(msg)
    def warning(self, msg):
        self.logger.warning(msg)
    def error(self, msg):
        self.logger.error(msg)
    def critical(self, msg):
        self.logger.critical(msg)
    def log(self, level, msg):
        self.logger.log(level, msg)
    def setLevel(self, level):
        self.logger.setLevel(level)
    def disable(self):
        logging.disable(50)
    def check_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
