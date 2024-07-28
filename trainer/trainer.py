import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, feature_model, reason_model, head_model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(feature_model, reason_model, head_model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
        self.adapt_flag = 'adapter' in config['trainer']
        
        if self.adapt_flag:
            cfg = config['trainer']['adapter']
            self.adapter = nn.Linear(cfg['input_size'], cfg['output_size']).to(device=device)
            self.adapter_optimizer = torch.optim.Adam(self.adapter.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.feature_model.train()
        self.reason_model.train()
        self.head_model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f'Epoch {epoch}')):
            if isinstance(data, tuple) or isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].to(self.device)
            else:
                data = data.to(self.device)
            if isinstance(target, tuple) or isinstance(target, list):
                for i in range(len(target)):
                    if not isinstance(target[i], list):
                        target[i] = target[i].to(self.device)
            else:
                target = target.to(self.device)

            # TODO special for CVR
            if 'Cvr' in type(self.data_loader).__name__:
                x_size = data.shape
                perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0)
                target = perms.argmax(1)
                perms = perms + torch.arange(x_size[0], device=self.device)[:,None]*4
                perms = perms.flatten()
                data = data.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]])
            self.optimizer.zero_grad()
            # output = self.model(data)
            feature = self.feature_model(data)
            if self.adapt_flag:
                feature = self.adapter(feature)
            # TODO for EBM
            if 'EBM' in self.reason_model.__class__.__name__:
                if 'Raven' in type(self.data_loader).__name__:
                    num_classes = 8
                elif 'Cvr' in type(self.data_loader).__name__:
                    num_classes = 4
                elif 'VQA' in type(self.data_loader).__name__:
                    num_classes = 10
                    target[0] = target[0] % 10
                    target_tuple = target 
                    target = target[0]
                else:
                    num_classes = 2
                energy_value = self.reason_model(feature, target)
                all_value = torch.zeros((len(target), num_classes)).to(self.device)
                for i in range(num_classes):
                    all_target = torch.Tensor([i]).repeat(len(target))
                    all_target = all_target.to(self.device)
                    all_value[:, i] = self.reason_model(feature, all_target)
                min_value, _ = torch.min(all_value, axis=1)
                loss = torch.sum(energy_value - min_value.to(energy_value.dtype))
                output = -all_value
            # elif 'LLM' in self.reason_model.__class__.__name__:
            #     if 'Raven' in type(self.data_loader).__name__:
            #         num_classes = 8
            #     elif 'Cvr' in type(self.data_loader).__name__:
            #         num_classes = 4
            #     elif 'VQA' in type(self.data_loader).__name__:
            #         num_classes = 10
            #         target[0] = target[0] % 10
            #         target_tuple = target 
            #         target = target[0]
            #     else:
            #         num_classes = 2
            #     # answer_pool = [str(i) for i in range(num_classes)]
            #     # for answer in answer_pool:
            #     #     answers = [answer] * len(target)
            #     #     output = self.reason_model(feature, answers)
            #     answers = target.cpu().numpy().astype(str)
            #     loss = self.reason_model(feature, answers)
            else:
                if 'LLM' in self.reason_model.__class__.__name__:
                    answers = target.cpu().numpy().astype(str)
                    # answers = ['\n'] * target.shape[0]
                    concept = self.reason_model(feature, answers)
                else:
                    concept = self.reason_model(feature)
                output = self.head_model(concept)
                if isinstance(target[1], list):
                    loss = self.criterion(output, target[0])
                else:
                    loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if self.adapt_flag:
                self.adapter_optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            # if 'LLM' not in self.reason_model.__class__.__name__:
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # if batch_idx % self.log_step == 0:
                # self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                #     epoch,
                #     self._progress(batch_idx),
                #     loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.feature_model.eval()
        self.reason_model.eval()
        self.head_model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                if isinstance(data, tuple) or isinstance(data, list):
                    for i in range(len(data)):
                        data[i] = data[i].to(self.device)
                else:
                    data = data.to(self.device)
                if isinstance(target, tuple) or isinstance(target, list):
                    for i in range(len(target)):
                        target[i] = target[i].to(self.device)
                else:
                    target = target.to(self.device)

                # TODO special for CVR
                if 'Cvr' in type(self.data_loader).__name__:
                    x_size = data.shape
                    perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0)
                    target = perms.argmax(1)
                    perms = perms + torch.arange(x_size[0], device=self.device)[:,None]*4
                    perms = perms.flatten()
                    data = data.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]])

                # output = self.model(data)
                feature = self.feature_model(data)
                if self.adapt_flag:
                    feature = self.adapter(feature)
                if 'EBM' in self.reason_model.__class__.__name__:
                    if 'Raven' in type(self.data_loader).__name__:
                        num_classes = 8
                    elif 'Cvr' in type(self.data_loader).__name__:
                        num_classes = 4
                    elif 'VQA' in type(self.data_loader).__name__:
                        num_classes = 10
                        target[0] = target[0] % 10
                        target_tuple = target 
                        target = target[0]
                    else:
                        num_classes = 2
                    energy_value = self.reason_model(feature, target)
                    all_value = torch.zeros((len(target), num_classes)).to(self.device)
                    for i in range(num_classes):
                        all_target = torch.Tensor([i]).repeat(len(target))
                        all_target = all_target.to(self.device)
                        all_value[:, i] = self.reason_model(feature, all_target)
                    min_value, _ = torch.min(all_value, axis=1)
                    loss = torch.sum(energy_value - min_value.to(energy_value.dtype))
                    output = -all_value
                # elif 'LLM' in self.reason_model.__class__.__name__:
                #     if 'Raven' in type(self.data_loader).__name__:
                #         num_classes = 8
                #     elif 'Cvr' in type(self.data_loader).__name__:
                #         num_classes = 4
                #     elif 'VQA' in type(self.data_loader).__name__:
                #         num_classes = 10
                #         target[0] = target[0] % 10
                #         target_tuple = target 
                #         target = target[0]
                #     else:
                #         num_classes = 2
                #     answer_pool = [str(i) for i in range(num_classes)]
                #     output = torch.zeros(num_classes, len(target)).to(self.device)
                #     for i, answer in enumerate(answer_pool):
                #         answers = [answer] * len(target)
                #         output[i] = self.reason_model(feature, answers)
                #     with torch.no_grad():
                #         pred = torch.argmin(output, dim=0)
                #         assert pred.shape[0] == len(target)
                #         correct = 0
                #         correct += torch.sum(pred == target).item()
                #     self.valid_metrics.update('accuracy', correct / len(target))
                    
                else:
                    
                    if 'LLM' in self.reason_model.__class__.__name__:
                        # answers = target.cpu().numpy().astype(str)
                        answers = ['\n'] * target.shape[0]
                        concept = self.reason_model(feature, answers)
                    else:
                        concept = self.reason_model(feature)
                    output = self.head_model(concept)
                    loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # if 'LLM' not in self.reason_model.__class__.__name__:
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        if 'optuna' in self.config.config:
            pass
        else:
            # add histogram of model parameters to the tensorboard
            for name, p in self.feature_model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
            for name, p in self.reason_model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
            for name, p in self.head_model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
