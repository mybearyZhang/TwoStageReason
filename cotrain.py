import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.feature as module_feature
import model.reasoning as module_reason
import model.head as module_head
import wandb
import optuna
from parse_config import ConfigParser
from trainer import Cotrainer
from utils import prepare_device, is_valid


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    if is_valid('wandb', config['trainer']):
        wandb.init(project=config['name'], entity='reasoner', config={"x-axis": "epoch"})

    data_loaders = {}
    valid_data_loaders = {}
    for key, _ in config['data_loaders'].items():
        # setup data_loader instances
        strings = ['Bongard']
        # val_dict = {'RAVEN': -1, 'CVR': -1, 'SVRT': -1, 'BongardHOI': -1, 'BongardLogo': -1,\
        #                  'CophyBall': -1, 'CophyTower': -1, 'CophyColli': -1, 'VQA': -1}
        load_option = any(string == key for string in strings)

        if not load_option:
            data_loader = config.init_obj_deep2('data_loaders', key, module_data)
            config['data_loaders'][key]['args']['split'] = "val"
            if 'val_samples' in config['data_loaders'][key]['args']:
                config['data_loaders'][key]['args']['n_samples'] = config['data_loaders'][key]['args']['val_samples']
            else:
                config['data_loaders'][key]['args']['n_samples'] = -1
            valid_data_loader = config.init_obj_deep2('data_loaders', key, module_data)
        else:
            data_loader = config.init_obj_deep2('data_loaders', key, module_data)
            valid_data_loader = data_loader.split_validation()
        data_loaders[key] = data_loader
        valid_data_loaders[key] = valid_data_loader

    feature_models = {}
    head_models = {}
    for key, _ in config['feature_models'].items():
        feature_models[key] = config.init_obj_deep2('feature_models', key, module_feature)
        if 'pretrained' in config['feature_models'][key]:
            checkpoint = torch.load(config['feature_models'][key]['pretrained']['path'])
            if config['feature_models'][key]['pretrained']['cotrain']:
                feature_models[key].load_state_dict(checkpoint['feature_state_dicts'][key])
            else:
                feature_models[key].load_state_dict(checkpoint['feature_state_dict'])
    reason_model = config.init_obj('reason_model', module_reason)
    if 'pretrained' in config['reason_model']:
        checkpoint = torch.load(config['reason_model']['pretrained']['path'])
        reason_model.load_state_dict(checkpoint['reason_state_dict'])
    for key, _ in config['head_models'].items():
        head_models[key] = config.init_obj_deep2('head_models', key, module_head)
        if 'pretrained' in config['head_models'][key]:
            checkpoint = torch.load(config['head_models'][key]['pretrained']['path'])
            if config['head_models'][key]['pretrained']['cotrain']:
                head_models[key].load_state_dict(checkpoint['head_state_dicts'][key])
            else:
                head_models[key].load_state_dict(checkpoint['head_state_dict'])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    # model = model.to(device)
    for _, feature_model in feature_models.items():
        feature_model = feature_model.to(device)
    reason_model = reason_model.to(device)
    for _, head_model in head_models.items():
        head_model = head_model.to(device)
    
    if len(device_ids) > 1:
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        for _, feature_model in feature_models.items():
            feature_model = torch.nn.DataParallel(feature_model, device_ids=device_ids)
        reason_model = torch.nn.DataParallel(reason_model, device_ids=device_ids)
        if type(reason_model).__name__ == 'GNN':
            reason_model.set_device(device)
        for _, head_model in head_models.items():
            head_model = torch.nn.DataParallel(head_model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterions = {}
    for key, _ in config['losses'].items():
        criterions[key] = getattr(module_loss, config['losses'][key])
    metrics = {}
    for key, _ in config['metrics'].items():
        metrics[key] = [getattr(module_metric, met) for met in config['metrics'][key]]
    if 'optimizer' in config.config:
        parameters = []
        for _, feature_model in feature_models.items():
            parameters += list(feature_model.parameters())
        parameters += list(reason_model.parameters())
        for _, head_model in head_models.items():
            parameters += list(head_model.parameters())
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, parameters)

        # TODO: For multitask, are all wd identical?
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Cotrainer(feature_models, reason_model, head_models,
                        criterions, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loaders=data_loaders,
                        valid_data_loaders=valid_data_loaders,
                        lr_scheduler=lr_scheduler)
    elif 'optimizers' in config.config:
        optimizers = {}
        lr_schedulers = {}
        for key, _ in config['optimizers'].items():
            if key == 'reasoning':
                trainable_params = filter(lambda p: p.requires_grad, reason_model.parameters())
                optimizers[key] = config.init_obj_deep2('optimizers', key, torch.optim, trainable_params)
                lr_schedulers[key] = config.init_obj_deep2('lr_schedulers', key, torch.optim.lr_scheduler, optimizers[key])
            else:
                # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
                trainable_params = filter(lambda p: p.requires_grad, list(feature_models[key].parameters()) + list(head_models[key].parameters()))
                optimizers[key] = config.init_obj_deep2('optimizers', key, torch.optim, trainable_params)
                lr_schedulers[key] = config.init_obj_deep2('lr_schedulers', key, torch.optim.lr_scheduler, optimizers[key])
        
        trainer = Cotrainer(feature_models, reason_model, head_models,
                        criterions, metrics, optimizers,
                        config=config,
                        device=device,
                        data_loaders=data_loaders,
                        valid_data_loaders=valid_data_loaders,
                        lr_scheduler=lr_schedulers)
    trainer.train()
    if is_valid('wandb', config['trainer']):
        wandb.finish()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
