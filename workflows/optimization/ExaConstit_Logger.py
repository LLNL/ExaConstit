import logging
import sys



def initialize_ExaProb_log(glob_loglvl='debug', filename='logbook_ExaProb.log', restart=False):
    global logger
    
    if glob_loglvl=='error':
        level = logging.ERROR
    if glob_loglvl=='warning':
        level = logging.WARNING
    if glob_loglvl=='info':
        level = logging.INFO
    if glob_loglvl=='debug':
        level = logging.DEBUG

    if restart:
        logging.basicConfig(filename=filename, level=level, format='%(message)s', datefmt='%m/%d/%Y %H:%M:%S ', filemode='a')
    # Make log file to track the runs. This file will be created after the code starts to run.
    else:
        logging.basicConfig(filename=filename, level=level, format='%(message)s', datefmt='%m/%d/%Y %H:%M:%S ', filemode='w')
    logger = logging.getLogger()



def write_ExaProb_log(text, type='info', changeline=False):

    try: logger
    except:
        sys.exit('You did not initialize ExaConstit_Logger.py')

    if changeline == True:
        logger.info('\n')
        print('\n')
    if type=='error':
        logger.error('ERROR: '+text)
        print('ERROR: '+text)
    elif type =='warning':
        logger.warning('WARNING: '+text)
        print('WARNING: '+text)
    elif type =='info':
        logger.info(text)
        print(text)
    elif type == 'debug':
        logger.debug(text)