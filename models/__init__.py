import logging
logger = logging.getLogger('base')


def create_model(opt):
    # image restoration
    model = opt['model']
    if model == 'sr':
        from .SIEN_model import SIEN_Model as M
    elif model == 'sr2':
        from .SIEN_model2 import SIEN_Model2 as M
    elif model == 'sr3':
        from .SIEN_model3 import SIEN_Model3 as M
    elif model == 'sr_pre':
        from .SIEN_model_Pre import SIEN_Model_Pre as M
    elif model == 'sr_pre1':
        from .SIEN_model_Pre1 import SIEN_Model_Pre1 as M
    elif model == 'sr_pre2':
        from .SIEN_model_Pre2 import SIEN_Model_Pre2 as M
    elif model == 'sr_pre3':
        from .SIEN_model_Pre3 import SIEN_Model_Pre3 as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
