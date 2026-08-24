# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_valid_then_test_value = 0.0
    best_test_any_value = 0.0
    idx = best_valid_then_test_idx = 0
    best_test_any_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model, mg)
        # debug
        # model training
        (
            best_valid_score,
            best_valid_result,
            best_test_upon_valid,
            best_test_score,
            best_test_result,
        ) = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid, best_test_result))

        # save best test upon best valid
        if best_test_upon_valid[val_metric] > best_valid_then_test_value:
            best_valid_then_test_value = best_test_upon_valid[val_metric]
            best_valid_then_test_idx = idx
        # save globally best test
        if best_test_result[val_metric] > best_test_any_value:
            best_test_any_value = best_test_result[val_metric]
            best_test_any_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('best test upon best valid: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('globally best test: {}'.format(dict2str(best_test_result)))
        logger.info('████Current BEST (Valid->Test)████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n'.format(
                        config['hyper_parameters'],
                        hyper_ret[best_valid_then_test_idx][0],
                        dict2str(hyper_ret[best_valid_then_test_idx][1]),
                        dict2str(hyper_ret[best_valid_then_test_idx][2]),
                    ))
        logger.info('████Current BEST (Test)████:\nParameters: {}={},\n'
                    'Test: {}\n\n\n'.format(
                        config['hyper_parameters'],
                        hyper_ret[best_test_any_idx][0],
                        dict2str(hyper_ret[best_test_any_idx][3]),
                    ))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v, t) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test upon best valid: {},\n best test: {}'.format(
            config['hyper_parameters'], p, dict2str(k), dict2str(v), dict2str(t)))

    logger.info('\n\n█████████████ BEST (Valid->Test) ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n'.format(
        config['hyper_parameters'],
        hyper_ret[best_valid_then_test_idx][0],
        dict2str(hyper_ret[best_valid_then_test_idx][1]),
        dict2str(hyper_ret[best_valid_then_test_idx][2]),
    ))
    logger.info('█████████████ BEST (Test) ████████████████')
    logger.info('\tParameters: {}={},\nTest: {}\n\n'.format(
        config['hyper_parameters'],
        hyper_ret[best_test_any_idx][0],
        dict2str(hyper_ret[best_test_any_idx][3]),
    ))

    return {
        'best_valid_then_test': {
            'parameters': hyper_ret[best_valid_then_test_idx][0],
            'valid': hyper_ret[best_valid_then_test_idx][1],
            'test': hyper_ret[best_valid_then_test_idx][2],
        },
        'best_test_any': {
            'parameters': hyper_ret[best_test_any_idx][0],
            'test': hyper_ret[best_test_any_idx][3],
        },
    }
