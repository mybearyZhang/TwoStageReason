import numpy as np
import torch
import random
from torchvision.utils import make_grid
from base import BaseCotrainer
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from utils import MultipleTqdm

class Cotrainer(BaseCotrainer):
    """
    Trainer class
    """
    def __init__(self, feature_models, reason_model, head_models, criterions, metric_ftns, optimizer, config, device,
                 data_loaders, valid_data_loaders=None, lr_scheduler=None, len_epoch=None):
        super().__init__(feature_models, reason_model, head_models, criterions, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loaders = data_loaders
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = 0
            for data_loader in self.data_loaders.values():
                self.len_epoch += len(data_loader)
        else:
            # iteration-based training
            self.data_loaders = inf_loop(data_loaders)
            self.len_epoch = len_epoch
        self.valid_data_loaders = valid_data_loaders
        self.do_validation = self.valid_data_loaders is not None
        if self.multi_optimizer:
            self.lr_schedulers = {}
            for name, opt in lr_scheduler.items():
                self.lr_schedulers[name] = opt
        else:
            self.lr_scheduler = lr_scheduler
        self.log_steps = {}
        for key, data_loader in data_loaders.items():
            self.log_steps[key] = int(np.sqrt(data_loader.batch_size))
        # self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metricses = {}
        self.valid_metricses = {}
        for name in data_loaders.keys():
            self.train_metricses[name] = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns[name]], writer=self.writer)
            self.valid_metricses[name] = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns[name]], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        for _, feature_model in self.feature_models.items():
            feature_model.train()
        self.reason_model.train()
        for _, head_model in self.head_models.items():
            head_model.train()
        for name in self.data_loaders.keys():
            self.train_metricses[name].reset()

        if self.config['trainer']['pick_prob']:
            # TODO Co-Training
            iters = {}
            batch_idxs = {}
            len_list = []
            name_list = []
            total_len = 0
            max_batches = 0
            for name, dataloader in self.data_loaders.items():
                if self.config['trainer']['entry_epochs'][name] >= epoch:
                    continue
                iters[name] = iter(dataloader)
                batch_idxs[name] = 0
                length = len(dataloader)
                name_list.append(name)
                len_list.append(length)
                max_batches = max(max_batches, length)
                total_len += length
            weights = [l / total_len for l in len_list]
            
            with tqdm(total=total_len, desc=f'Epoch {epoch}') as pbar:
                with MultipleTqdm(name_list=name_list, total_list=len_list) as tqdm_objs:
                    for idx in range(total_len):
                        # TODO config losses
                        losses = 0
                        name = random.choices(name_list, weights=weights)[0]
                        # tqdm.set_postfix(batch=name)
                        while batch_idxs[name] >= len(self.data_loaders[name]):
                            name = random.choices(name_list, weights=weights)[0]
                        if batch_idxs[name] < len(self.data_loaders[name]):
                            batch_idx = batch_idxs[name]
                            data, target = next(iters[name])
                            # TODO special for CVR
                            if 'Cvr' in type(self.data_loaders[name]).__name__:
                                x_size = data.shape
                                perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0)
                                target = perms.argmax(1)
                                perms = perms + torch.arange(x_size[0], device=self.device)[:,None]*4
                                perms = perms.flatten()
                                perms = perms.to(data.device)
                                data = data.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]])
                            
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
                            if self.multi_optimizer:
                                self.optimizers[name].zero_grad()
                                self.optimizers['reasoning'].zero_grad()
                            else:
                                self.optimizer.zero_grad()
                            feature = self.feature_models[name](data)
                            concept = self.reason_model(feature)
                            output = self.head_models[name](concept)
                            loss = self.criterions[name](output, target)
                            loss.backward()
                            if self.multi_optimizer:
                                self.optimizers[name].step()
                                self.optimizers['reasoning'].step()
                            else:
                                self.optimizer.step()

                            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                            self.train_metricses[name].update('loss', loss.item())
                            for met in self.metric_ftns[name]:
                                self.train_metricses[name].update(met.__name__, met(output, target))

                            # if batch_idx % self.log_steps[name] == 0:
                            #     self.logger.debug('Train Epoch: {} {} Loss_{}: {:.6f}'.format(
                            #         epoch,
                            #         self._progress(batch_idx, name),
                            #         name,
                            #         loss.item()))
                                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                            if batch_idx == self.len_epoch:
                                break
                            batch_idxs[name] += 1
                            tqdm_objs[name].update(1)
                            pbar.update(1)
                        else:
                            loss = 0
                        # losses.backward()
                
        else:
            # TODO Co-Training
            iters = {}
            for name, dataloader in self.data_loaders.items():
                iters[name] = iter(dataloader)
            name_list = list(self.data_loaders.keys())
            max_batches = max(len(data_loader) for data_loader in self.data_loaders.values())
            for batch_idx in range(max_batches):
                # TODO config losses
                losses = 0
                random.shuffle(name_list)
                for name in name_list:
                    if batch_idx < len(self.data_loaders[name]):
                        data, target = next(iters[name])
                        # TODO special for CVR
                        if 'Cvr' in type(self.data_loaders[name]).__name__:
                            x_size = data.shape
                            perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0)
                            target = perms.argmax(1)
                            perms = perms + torch.arange(x_size[0], device=self.device)[:,None]*4
                            perms = perms.flatten()
                            perms = perms.to(data.device)
                            data = data.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]])
                        
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
                        if self.multi_optimizer:
                            self.optimizers[name].zero_grad()
                        else:
                            self.optimizer.zero_grad()
                        # output = self.model(data)
                        feature = self.feature_models[name](data)
                        concept = self.reason_model(feature)
                        output = self.head_models[name](concept)
                        loss = self.criterions[name](output, target)
                        losses += loss
                        # loss.backward()
                        # self.optimizer.step()

                        # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                        self.train_metricses[name].update('loss', loss.item())
                        for met in self.metric_ftns[name]:
                            self.train_metricses[name].update(met.__name__, met(output, target))

                        if batch_idx % self.log_steps[name] == 0:
                            self.logger.debug('Train Epoch: {} {} Loss_{}: {:.6f}'.format(
                                epoch,
                                self._progress(batch_idx, name),
                                name,
                                loss.item()))
                            # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))\

                        if batch_idx == self.len_epoch:
                            break
                    else:
                        loss = 0
                losses.backward()
                if self.multi_optimizer:
                    for name in name_list:
                        self.optimizers[name].step()
                    self.optimizers['reasoning'].step()
                else:
                    self.optimizer.step()
        # TODO logs
        logs = {}
        for name in self.data_loaders.keys():
            logs[name] = self.train_metricses[name].result()
        if self.do_validation:
            val_logs = self._valid_epoch(epoch)
            for name, val_log in val_logs.items():
                logs[name].update(**{'val_'+k : v for k, v in val_log.items()})
        if self.multi_optimizer:
            for name, lr_scheduler in self.lr_schedulers.items():
                lr_scheduler.step()
        else:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        return logs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for _, feature_model in self.feature_models.items():
            feature_model.eval()
        self.reason_model.eval()
        for _, head_model in self.head_models.items():
            head_model.eval()
        for name in self.data_loaders.keys():
            self.valid_metricses[name].reset()
        # TODO
        with torch.no_grad():
            for name, valid_data_loader in self.valid_data_loaders.items():
                for batch_idx, (data, target) in enumerate(valid_data_loader):
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
                    if 'Cvr' in type(valid_data_loader).__name__:
                        x_size = data.shape
                        perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0)
                        target = perms.argmax(1)
                        perms = perms + torch.arange(x_size[0], device=self.device)[:,None]*4
                        perms = perms.flatten()
                        data = data.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]])

                    # output = self.model(data)
                    feature = self.feature_models[name](data)
                    concept = self.reason_model(feature)
                    output = self.head_models[name](concept)
                    loss = self.criterions[name](output, target)

                    self.writer.set_step((epoch - 1) * len(valid_data_loader) + batch_idx, 'valid')
                    self.valid_metricses[name].update('loss', loss.item())
                    for met in self.metric_ftns[name]:
                        self.valid_metricses[name].update(met.__name__, met(output, target))

        # add histogram of model parameters to the tensorboard
        for _, feature_model in self.feature_models.items():
            for name, p in feature_model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.reason_model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        for _, head_model in self.head_models.items():
            for name, p in head_model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        results = {}
        for name in self.data_loaders.keys():
            results[name] = self.valid_metricses[name].result() 
        return results

    def _progress(self, batch_idx, name):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loaders[name], 'n_samples'):
            current = batch_idx * self.data_loaders[name].batch_size
            total = self.data_loaders[name].n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
