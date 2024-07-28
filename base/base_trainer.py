import torch
import wandb
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils import is_valid

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, feature_model, reason_model, head_model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # self.model = model
        self.feature_model = feature_model
        self.reason_model = reason_model
        self.head_model = head_model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
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
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            if is_valid('wandb', self.config['trainer']):
                for key, value in log.items():
                    wandb.log({key: value}, step=epoch)

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

    def train_optuna(self):
        """
        Full training logic
        """
        not_improved_count = 0
        monitor = self.config['optuna']['monitor']
        mnt_mode, mnt_metric = monitor.split()
        assert mnt_mode in ['min', 'max']
        mnt_best = inf if mnt_mode == 'min' else -inf
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            res = result[mnt_metric]
            if mnt_mode == 'min' and res < mnt_best:
                mnt_best = res
            elif mnt_mode == 'max' and res > mnt_best:
                mnt_best = res

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            if is_valid('wandb', self.config['trainer']):
                for key, value in log.items():
                    wandb.log({key: value}, step=epoch)

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
        return mnt_best           # For Optuna Fine-tuning

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        feature_arch = type(self.feature_model).__name__
        reason_arch = type(self.reason_model).__name__
        head_arch = type(self.head_model).__name__
        state = {
            'feature_arch': feature_arch,
            'reason_arch': reason_arch,
            'head_arch': head_arch,
            'epoch': epoch,
            'feature_state_dict': self.feature_model.state_dict(),
            'reason_state_dict': self.reason_model.state_dict(),
            'head_state_dict': self.head_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
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
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        if list(feature_state_dict.keys())[0].startswith('module.') and self.config['n_gpu'] == 1:
            feature_state_dict = {k[7:]: v for k, v in checkpoint['feature_state_dict'].items()}
            reason_state_dict = {k[7:]: v for k, v in checkpoint['reason_state_dict'].items()}
            head_state_dict = {k[7:]: v for k, v in checkpoint['head_state_dict'].items()}    
            self.feature_model.load_state_dict(feature_state_dict)
            self.reason_model.load_state_dict(reason_state_dict)
            self.head_model.load_state_dict(head_state_dict)
        elif self.config['n_gpu'] > 1 and not list(feature_state_dict.keys())[0].startswith('module.'):
            feature_state_dict = {'module.' + k: v for k, v in checkpoint['feature_state_dict'].items()}
            reason_state_dict = {'module.' + k: v for k, v in checkpoint['reason_state_dict'].items()}
            head_state_dict = {'module.' + k: v for k, v in checkpoint['head_state_dict'].items()}
            self.feature_model.load_state_dict(feature_state_dict)
            self.reason_model.load_state_dict(reason_state_dict)
            self.head_model.load_state_dict(head_state_dict)
        else:
            self.feature_model.load_state_dict(checkpoint['feature_state_dict'])
            self.reason_model.load_state_dict(checkpoint['reason_state_dict'])
            self.head_model.load_state_dict(checkpoint['head_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
