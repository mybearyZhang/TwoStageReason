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
import optuna
import wandb
from parse_config import ConfigParser
from trainer import Trainer
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

    strings = ['BongardDataLoader']
    load_option = any(string == config['data_loader']['type'] for string in strings)

    # setup data_loader instances
    if not load_option:
        data_loader = config.init_obj('data_loader', module_data)
        config['data_loader']['args']['split'] = "val"
        if 'val_samples' in config['data_loader']['args']:
            config['data_loader']['args']['n_samples'] = config['data_loader']['args']['val_samples']
        else:
            config['data_loader']['args']['n_samples'] = -1
        valid_data_loader = config.init_obj('data_loader', module_data)
    else:
        data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.split_validation()

    if 'optuna' in config.config:
        def objective(trial):
            if is_valid('wandb', config['trainer']):
                wandb.init(project=config['name'], entity='reasoner', config={"x-axis": "epoch"})
            if 'models' in config['optuna']:
                scale = config['optuna']['models']['dropout']
                config['feature_model']['args']['dropout'] = trial.suggest_uniform("dropout", scale['min'], scale['max'])

            feature_model = config.init_obj('feature_model', module_feature)
            reason_model = config.init_obj('reason_model', module_reason)
            head_model = config.init_obj('head_model', module_head)

            if 'pretrained' in config['feature_model']:
                if config['feature_model']['pretrained']['cotrain']:
                    checkpoint = torch.load(config['feature_model']['pretrained']['path'])['feature_state_dicts']
                else:
                    checkpoint = torch.load(config['feature_model']['pretrained']['path'])['feature_state_dict']
                if list(checkpoint.keys())[0].startswith('module.') and config['n_gpu'] == 1:
                    checkpoint = {k[7:]: v for k, v in checkpoint.items()}
                if config['n_gpu'] > 1:
                    feature_model = torch.nn.DataParallel(feature_model)
                    if not list(checkpoint.keys())[0].startswith('module.'):
                        checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
                feature_model.load_state_dict(checkpoint)
            if 'pretrained' in config['reason_model']:
                checkpoint = torch.load(config['reason_model']['pretrained']['path'])['reason_state_dict']
                if list(checkpoint.keys())[0].startswith('module.') and config['n_gpu'] == 1:
                    checkpoint = {k[7:]: v for k, v in checkpoint.items()}
                if config['n_gpu'] > 1:
                    reason_model = torch.nn.DataParallel(reason_model)
                    if not list(checkpoint.keys())[0].startswith('module.'):
                        checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
                reason_model.load_state_dict(checkpoint)
            if 'pretrained' in config['head_model']:
                if config['head_model']['pretrained']['cotrain']:
                    checkpoint = torch.load(config['head_model']['pretrained']['path'])['head_state_dicts']
                else:
                    checkpoint = torch.load(config['head_model']['pretrained']['path'])['head_state_dict']
                if list(checkpoint.keys())[0].startswith('module.') and config['n_gpu'] == 1:
                    checkpoint = {k[7:]: v for k, v in checkpoint.items()}
                if config['n_gpu'] > 1:
                    head_model = torch.nn.DataParallel(head_model)
                    if not list(checkpoint.keys())[0].startswith('module.'):
                        checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
                head_model.load_state_dict(checkpoint)

            # prepare for (multi-device) GPU training
            device, device_ids = prepare_device(config['n_gpu'])
            # model = model.to(device)
            feature_model = feature_model.to(device)
            reason_model = reason_model.to(device)
            if type(reason_model).__name__ == 'GNN':
                reason_model.set_device(device)
            head_model = head_model.to(device)
            if len(device_ids) > 1:
                # model = torch.nn.DataParallel(model, device_ids=device_ids)
                feature_model = torch.nn.DataParallel(feature_model, device_ids=device_ids)
                reason_model = torch.nn.DataParallel(reason_model, device_ids=device_ids)
                head_model = torch.nn.DataParallel(head_model, device_ids=device_ids)

            # get function handles of loss and metrics
            criterion = getattr(module_loss, config['loss'])
            metrics = [getattr(module_metric, met) for met in config['metrics']]

            # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
            if 'adapter' in config['trainer']:
                trainable_params = list(feature_model.parameters()) + list(reason_model.parameters()) + list(head_model.parameters())
            else:
                trainable_params = filter(lambda p: p.requires_grad, list(feature_model.parameters()) + list(reason_model.parameters()) + list(head_model.parameters()))
            optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
            params = config['optuna']['params']
            cur_params = {}
            print("``Hyperparameter Groups")
            for name, scale in params.items():
                cur_params[name] = trial.suggest_loguniform(name, scale['min'], scale['max'])
                print(f"{name}: {cur_params[name]}")
            param_groups = optimizer.param_groups

            # TODO: only support explicit params
            for param in param_groups:
                param['lr'] = cur_params['learning_rate']
                param['weight_decay'] = cur_params['weight_decay']
            trainer = Trainer(feature_model, reason_model, head_model,
                        criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)
            return trainer.train_optuna()
        monitor = config['optuna']['monitor']
        mnt_mode, mnt_metric = monitor.split()
        assert mnt_mode in ['min', 'max']
        if mnt_mode == 'min':
            mnt_mode = 'minimize'
        else:
            mnt_mode = 'maximize'
        study = optuna.create_study(study_name=config['optuna']['study_name'],
                                    direction=mnt_mode,
                                    storage=config['optuna']['storage'],
                                    load_if_exists=config['optuna']['load_if_exists'])
        study.optimize(objective, n_trials=config['optuna']['n_trials'])

        best_params = study.best_params
        best_value = study.best_value

        print("Best params:", best_params)
        print("Best value:", best_value)
        
    else:
        if is_valid('wandb', config['trainer']):
            wandb.init(project=config['name'], entity='reasoner', config={"x-axis": "epoch"})
        feature_model = config.init_obj('feature_model', module_feature)
        reason_model = config.init_obj('reason_model', module_reason)
        head_model = config.init_obj('head_model', module_head)

        if 'pretrained' in config['feature_model']:
            if config['feature_model']['pretrained']['cotrain']:
                checkpoint = torch.load(config['feature_model']['pretrained']['path'])['feature_state_dicts']
            else:
                checkpoint = torch.load(config['feature_model']['pretrained']['path'])['feature_state_dict']
            if list(checkpoint.keys())[0].startswith('module.') and config['n_gpu'] == 1:
                checkpoint = {k[7:]: v for k, v in checkpoint.items()}
            if config['n_gpu'] > 1:
                feature_model = torch.nn.DataParallel(feature_model)
                if not list(checkpoint.keys())[0].startswith('module.'):
                    checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
            feature_model.load_state_dict(checkpoint)
        if 'pretrained' in config['reason_model']:
            checkpoint = torch.load(config['reason_model']['pretrained']['path'])['reason_state_dict']
            if list(checkpoint.keys())[0].startswith('module.') and config['n_gpu'] == 1:
                checkpoint = {k[7:]: v for k, v in checkpoint.items()}
            if config['n_gpu'] > 1:
                reason_model = torch.nn.DataParallel(reason_model)
                if not list(checkpoint.keys())[0].startswith('module.'):
                    checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
            reason_model.load_state_dict(checkpoint)
        if 'pretrained' in config['head_model']:
            if config['head_model']['pretrained']['cotrain']:
                checkpoint = torch.load(config['head_model']['pretrained']['path'])['head_state_dicts']
            else:
                checkpoint = torch.load(config['head_model']['pretrained']['path'])['head_state_dict']
            if list(checkpoint.keys())[0].startswith('module.') and config['n_gpu'] == 1:
                checkpoint = {k[7:]: v for k, v in checkpoint.items()}
            if config['n_gpu'] > 1:
                head_model = torch.nn.DataParallel(head_model)
                if not list(checkpoint.keys())[0].startswith('module.'):
                    checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
            head_model.load_state_dict(checkpoint)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        # model = model.to(device)
        feature_model = feature_model.to(device)
        reason_model = reason_model.to(device)
        if type(reason_model).__name__ == 'GNN':
            reason_model.set_device(device)
        head_model = head_model.to(device)
        if len(device_ids) > 1:
            # model = torch.nn.DataParallel(model, device_ids=device_ids)
            feature_model = torch.nn.DataParallel(feature_model, device_ids=device_ids)
            reason_model = torch.nn.DataParallel(reason_model, device_ids=device_ids)
            head_model = torch.nn.DataParallel(head_model, device_ids=device_ids)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        if 'adapter' in config['trainer']:
            trainable_params = list(feature_model.parameters()) + list(reason_model.parameters()) + list(head_model.parameters())
        else:
            trainable_params = filter(lambda p: p.requires_grad, list(feature_model.parameters()) + list(reason_model.parameters()) + list(head_model.parameters()))
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(feature_model, reason_model, head_model,
                      criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
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
