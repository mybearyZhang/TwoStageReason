import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.feature as module_feature
import model.reasoning as module_reason
import model.head as module_head
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    data_loaders = {}
    for key, _ in config['data_loaders'].items():
        # setup data_loader instances
        config['data_loaders'][key]['args']['split'] = "test"
        config['data_loaders'][key]['args']['shuffle'] = False
        if 'CVR' in key:
            config['data_loaders'][key]['args']['n_samples'] = -1
        data_loader = config.init_obj_deep2('data_loaders', key, module_data)
        data_loaders[key] = data_loader

    feature_models = {}
    head_models = {}
    for key, _ in config['feature_models'].items():
        feature_models[key] = config.init_obj_deep2('feature_models', key, module_feature)
    reason_model = config.init_obj('reason_model', module_reason)
    for key, _ in config['head_models'].items():
        head_models[key] = config.init_obj_deep2('head_models', key, module_head)

    # # build model architecture
    # logger.info(feature_model)
    # logger.info(reason_model)
    # logger.info(head_model)

    # get function handles of loss and metrics
    criterions = {}
    for key, _ in config['losses'].items():
        criterions[key] = getattr(module_loss, config['losses'][key])
    metric_fns = {}
    for key, _ in config['metrics'].items():
        metric_fns[key] = [getattr(module_metric, met) for met in config['metrics'][key]]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    logger.info('Loaded epoch: {}'.format(checkpoint['epoch']))
    feature_state_dicts = checkpoint['feature_state_dicts']
    reason_state_dict = checkpoint['reason_state_dict']
    head_state_dicts = checkpoint['head_state_dicts']
    if list(list(feature_state_dicts.values())[0].keys())[0].startswith('module.') and config['n_gpu'] == 1:
        for _, feature_state_dict in feature_state_dicts.items():
            feature_state_dict = {k[7:]: v for k, v in feature_state_dict.items()}
    if list(reason_state_dict.keys())[0].startswith('module.') and config['n_gpu'] == 1:
        reason_state_dict = {k[7:]: v for k, v in reason_state_dict.items()}
    if list(list(head_state_dicts.values())[0].keys())[0].startswith('module.') and config['n_gpu'] == 1:
        for _, head_state_dict in head_state_dicts.items():
            head_state_dict = {k[7:]: v for k, v in head_state_dict.items()}   
    if config['n_gpu'] > 1:
        for _, feature_model in feature_models.items():
            feature_model = torch.nn.DataParallel(feature_model)
        reason_model = torch.nn.DataParallel(reason_model)
        for _, head_model in head_models.items():
            head_model = torch.nn.DataParallel(head_model)
        if not list(list(feature_state_dicts.values())[0].keys())[0].startswith('module.'):
            for _, feature_state_dict in feature_state_dicts.items():
                feature_state_dict = {'module.' + k: v for k, v in feature_state_dict.items()}
        if not list(reason_state_dict.keys())[0].startswith('module.'):
            reason_state_dict = {'module.' + k: v for k, v in reason_state_dict.items()}
        if not list(list(head_state_dicts.values())[0].keys())[0].startswith('module.'):
            for _, head_state_dict in head_state_dicts.items():
                head_state_dict = {'module.' + k: v for k, v in head_state_dict.items()}       
    for key, feature_model in feature_models.items():
        feature_model.load_state_dict(feature_state_dicts[key])
    reason_model.load_state_dict(reason_state_dict)
    for key, head_model in head_models.items():
        head_model.load_state_dict(head_state_dicts[key])

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for _, feature_model in feature_models.items():
        feature_model = feature_model.to(device)
    reason_model = reason_model.to(device)
    if type(reason_model).__name__ == 'GNN':
        reason_model.set_device(device)
    for _, head_model in head_models.items():
        head_model = head_model.to(device)
    for _, feature_model in feature_models.items():
        feature_model.eval()
    reason_model.eval()
    for _, head_model in head_models.items():
        head_model.eval()

    with torch.no_grad():
        for name, data_loader in data_loaders.items():
            total_loss = 0.0
            total_metrics = torch.zeros(len(metric_fns[name]))
            for i, (data, target) in enumerate(tqdm(data_loader)):
                if isinstance(data, tuple):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                else:
                    data, target = data.to(device), target.to(device)
                # TODO special for CVR
                if 'Cvr' in type(data_loader).__name__:
                    x_size = data.shape
                    perms = torch.stack([torch.randperm(4, device=device) for _ in range(x_size[0])], 0)
                    target = perms.argmax(1)
                    perms = perms + torch.arange(x_size[0], device=device)[:,None]*4
                    perms = perms.flatten()
                    data = data.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]])

                # output = self.model(data)
                feature = feature_models[name](data)
                concept = reason_model(feature)
                output = head_models[name](concept)
                loss = criterions[name](output, target)

                
                #
                # save sample images, or do something with output here
                #

                # computing loss, metrics on test set
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns[name]):
                    total_metrics[i] += metric(output, target) * batch_size

            n_samples = len(data_loader.sampler)
            log = {'name': name, 'loss': total_loss / n_samples}
            log.update({
                met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns[name])
            })
            logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
