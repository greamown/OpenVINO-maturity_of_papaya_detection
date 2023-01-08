import logging, os, colorlog
# ===============================================================================================
LOG_LEVEL = {   
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR  
}
# ===============================================================================================
def config_logger(log_name=None, write_mode='a', level='Debug', clear_log=False):

    logger = logging.getLogger()            # get logger
    logger.setLevel(LOG_LEVEL[level])       # set level
    
    if not logger.hasHandlers():    # if the logger is not setup
        basic_formatter = logging.Formatter( "%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)", "%y-%m-%d %H:%M:%S")
        formatter = colorlog.ColoredFormatter( "%(asctime)s %(log_color)s [%(levelname)-.4s] %(reset)s %(message)s %(purple)s (%(filename)s:%(lineno)s)", "%y-%m-%d %H:%M:%S")
        # add stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(LOG_LEVEL[level.lower()])
        logger.addHandler(stream_handler)

        # add file handler
        if clear_log and os.path.exists(log_name):
            logging.warning('Clearing exist log files')
            os.remove(log_name)
        if log_name:
            file_handler = logging.FileHandler(log_name, write_mode, 'utf-8')
            file_handler.setFormatter(basic_formatter)
            file_handler.setLevel(LOG_LEVEL['info'])
            logger.addHandler(file_handler)
    
    
    logger.info('Create logger.({})'.format(logger.name))
    logger.info('Enabled stream {}'.format(f'and file mode.({log_name})' if log_name else 'mode'))
    return logger
# ===============================================================================================
if __name__ == '__main__':
    
    import test

    config_logger(log_name='ivinno-vino.log', write_mode='w', level='debug')
    
    logging.info('start')

    test.start()
