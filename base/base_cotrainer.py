import torch
import wandb
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils import is_valid

class BaseCotrainer:
    """
    Base class for all trainers
    """
    def __init__(self, feature_models, reason_model, head_models, criterions, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # self.model = model
        self.feature_models = feature_models
        self.reason_model = reason_model
        self.head_models = head_models
        self.criterions = criterions
        self.metric_ftns = metric_ftns
        self.multi_optimizer = isinstance(optimizer, dict)
        if self.multi_optimizer:
            self.optimizers = {}
            for name, opt in optimizer.items():
                self.optimizers[name] = opt
        else:
            self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        best_total_val_acc = 0.0
        best_epoch = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            if is_valid('wandb', self.config['trainer']):
                for name, logs in result.items():
                    for key, value in logs.items():
                        wandb.log({name + ': ' + key: value}, step=epoch)

            total_val_acc = 1

            # for key, value in result.items():
            #     total_val_acc *= value['val_accuracy']
            # if is_valid('wandb', self.config['trainer']):
            #     wandb.log({"total_val_acc": total_val_acc, "best_total_val_acc": best_total_val_acc}, step=epoch)
            # if total_val_acc > best_total_val_acc:
            #     best_total_val_acc = total_val_acc
            #     best_epoch = epoch

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        return best_total_val_acc, best_epoch

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        feature_archs = {}
        for name, feature_model in self.feature_models.items():
            feature_archs[name] = type(feature_model).__name__
        reason_arch = type(self.reason_model).__name__
        head_archs = {}
        for name, head_model in self.head_models.items():
            head_archs[name] = type(head_model).__name__

        feature_state_dicts = {}
        for name, feature_model in self.feature_models.items():
            feature_state_dicts[name] = feature_model.state_dict()
        head_state_dicts = {}
        for name, head_model in self.head_models.items():
            head_state_dicts[name] = head_model.state_dict()
        state = {
            'feature_arch': feature_archs,
            'reason_arch': reason_arch,
            'head_arch': head_archs,
            'epoch': epoch,
            'feature_state_dicts': feature_state_dicts,
            'reason_state_dict': self.reason_model.state_dict(),
            'head_state_dicts': head_state_dicts,
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if self.multi_optimizer:
            optimizers_state_dicts = {}
            for name, optimizer in self.optimizers.items():
                optimizers_state_dicts[name] = optimizer.state_dict()
            state['optimizers'] = optimizers_state_dicts
        else:
            optimizer_state_dicts = self.optimizer.state_dict()
            state['optimizer'] = optimizer_state_dicts
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        # if checkpoint['config']['arch'] != self.config['arch']:
        #     self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
        #                         "checkpoint. This may yield an exception while state_dict is being loaded.")
        if list(feature_state_dict.keys())[0].startswith('module.') and self.config['n_gpu'] == 1:
            feature_state_dicts = {}
            for name, feature_state_dict in checkpoint['feature_state_dicts'].items():
                feature_state_dicts[name] = {k[7:]: v for k, v in feature_state_dict.items()}
            reason_state_dict = {k[7:]: v for k, v in checkpoint['reason_state_dict'].items()}
            head_state_dicts = {}
            for name, head_state_dict in checkpoint['head_state_dicts'].items():
                head_state_dicts[name] = {k[7:]: v for k, v in head_state_dict.items()}
            for name in checkpoint['feature_state_dicts'].keys():
                self.feature_models[name].load_state_dict(feature_state_dicts[name])
            self.reason_model.load_state_dict(reason_state_dict)
            for name in checkpoint['head_state_dicts'].keys():
                self.head_models[name].load_state_dict(head_state_dicts[name])
        elif self.config['n_gpu'] > 1 and not list(feature_state_dict.keys())[0].startswith('module.'):
            feature_state_dicts = {}
            for name, feature_state_dict in checkpoint['feature_state_dicts'].items():
                feature_state_dicts[name] = {'module.' + k: v for k, v in feature_state_dict.items()}
            reason_state_dict = {'module.' + k: v for k, v in checkpoint['reason_state_dict'].items()}
            head_state_dicts = {}
            for name, head_state_dict in checkpoint['head_state_dicts'].items():
                head_state_dicts[name] = {'module.' + k: v for k, v in head_state_dict.items()}
            for name in checkpoint['feature_state_dicts'].keys():
                self.feature_models[name].load_state_dict(feature_state_dicts[name])
            self.reason_model.load_state_dict(reason_state_dict)
            for name in checkpoint['head_state_dicts'].keys():
                self.head_models[name].load_state_dict(head_state_dicts[name])
        else:
            for name in checkpoint['feature_state_dicts'].keys():
                self.feature_models[name].load_state_dict(checkpoint['feature_state_dicts'][name])
            self.reason_model.load_state_dict(checkpoint['reason_state_dict'])
            for name in checkpoint['head_state_dicts'].keys():
                self.head_models[name].load_state_dict(checkpoint['head_state_dicts'][name])

        if checkpoint['config']['data_loaders'] != self.config['data_loaders']:
            self.logger.warning("Warning: Dataloaders given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        # load optimizer state from checkpoint only when optimizer type is not changed.
        elif checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
