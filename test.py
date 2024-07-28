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

torch.backends.cudnn.enabled = False

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    config['data_loader']['args']['split'] = "test"
    config['data_loader']['args']['shuffle'] = False
    if 'Cvr' in config['data_loader']['type']:
        config['data_loader']['args']['n_samples'] = -1
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture
    feature_model = config.init_obj('feature_model', module_feature)
    reason_model = config.init_obj('reason_model', module_reason)
    head_model = config.init_obj('head_model', module_head)
    logger.info(feature_model)
    logger.info(reason_model)
    logger.info(head_model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    
    if config.resume is None:
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
    else:
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        logger.info('Loaded epoch: {}'.format(checkpoint['epoch']))
        feature_state_dict = checkpoint['feature_state_dict']
        reason_state_dict = checkpoint['reason_state_dict']
        head_state_dict = checkpoint['head_state_dict']
        if list(feature_state_dict.keys())[0].startswith('module.') and config['n_gpu'] == 1:
            feature_state_dict = {k[7:]: v for k, v in checkpoint['feature_state_dict'].items()}
            reason_state_dict = {k[7:]: v for k, v in checkpoint['reason_state_dict'].items()}
            head_state_dict = {k[7:]: v for k, v in checkpoint['head_state_dict'].items()}    
        if config['n_gpu'] > 1:
            feature_model = torch.nn.DataParallel(feature_model)
            reason_model = torch.nn.DataParallel(reason_model)
            head_model = torch.nn.DataParallel(head_model)
            if not list(feature_state_dict.keys())[0].startswith('module.'):
                feature_state_dict = {'module.' + k: v for k, v in checkpoint['feature_state_dict'].items()}
                reason_state_dict = {'module.' + k: v for k, v in checkpoint['reason_state_dict'].items()}
                head_state_dict = {'module.' + k: v for k, v in checkpoint['head_state_dict'].items()}    
        feature_model.load_state_dict(feature_state_dict)
        reason_model.load_state_dict(reason_state_dict)
        head_model.load_state_dict(head_state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_model = feature_model.to(device)
    reason_model = reason_model.to(device)
    if type(reason_model).__name__ == 'GNN':
        reason_model.set_device(device)
    head_model = head_model.to(device)
    feature_model.eval()
    reason_model.eval()
    head_model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            if 'Cvr' in type(data_loader).__name__:
                x_size = data.shape
                perms = torch.stack([torch.randperm(4, device=device) for _ in range(x_size[0])], 0)
                target = perms.argmax(1)
                perms = perms + torch.arange(x_size[0], device=device)[:,None]*4
                perms = perms.flatten()
                data = data.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]])

            feature = feature_model(data)
            if 'LLM' in reason_model.__class__.__name__:
                # answers = target.cpu().numpy().astype(str)
                answers = ['\n'] * data.shape[0]
                concept = reason_model(feature, answers)
            else:
                concept = reason_model(feature)
            # concept = reason_model(feature)
            output = head_model(concept)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
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
