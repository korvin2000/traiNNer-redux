import torch
from collections import OrderedDict

from ..archs import build_network
from ..losses import build_loss
from ..utils import get_root_logger
from ..utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast

@MODEL_REGISTRY.register()
class SRGANModel(SRModel):
    """SRGAN model for single image super-resolution."""

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('contextual_opt'):
            self.cri_contextual = build_loss(train_opt['contextual_opt']).to(self.device)
        else:
            self.cri_contextual = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None

        if train_opt.get('avg_opt'):
            self.cri_avg = build_loss(train_opt['avg_opt']).to(self.device)
        else:
            self.cri_avg = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        autoamp_d=False
        autoamp_g=False
        autoamp_d=self.opt['try_autoamp_d']
        autoamp_g=self.opt['try_autoamp_g']
        if autoamp_g==True:
            with autocast():
                self.output = self.net_g(self.lq)
                l_g_total = 0
                loss_dict = OrderedDict()
                if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
                    # pixel loss
                    if self.cri_pix:
                        l_g_pix = self.cri_pix(self.output, self.gt)
                        l_g_total += l_g_pix
                        loss_dict['l_g_pix'] = l_g_pix
                    # perceptual loss
                    if self.cri_perceptual:
                        l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                        if l_g_percep is not None:
                            l_g_total += l_g_percep
                            loss_dict['l_g_percep'] = l_g_percep
                        if l_g_style is not None:
                            l_g_total += l_g_style
                            loss_dict['l_g_style'] = l_g_style
                    # contextual loss
                    if self.cri_contextual:
                        l_g_contextual = self.cri_contextual(self.output, self.gt)
                        l_g_total += l_g_contextual
                        loss_dict['l_g_contextual'] = l_g_contextual
                    if self.cri_color:
                        l_g_color = self.cri_color(self.output, self.gt)
                        l_g_total += l_g_color
                        loss_dict['l_g_color'] = l_g_color
                    if self.cri_avg:
                        l_g_avg = self.cri_avg(self.output, self.gt)
                        l_g_total += l_g_avg
                        loss_dict['l_g_avg'] = l_g_avg
                    # gan loss
                    fake_g_pred = self.net_d(self.output)
                    if isinstance (fake_g_pred,list) == False:
                        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                        l_g_total += l_g_gan
                        loss_dict['l_g_gan'] = l_g_gan

                        scaler.scale(l_g_total).backward()

                        scaler.step(self.optimizer_g)

                        scaler.update()
                    else:
                        fake_g_preds = fake_g_pred
                        loss_dict['l_g_gan'] = 0
                        for fake_g_pred in fake_g_preds:
                            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                            l_g_total += l_g_gan
                            loss_dict['l_g_gan'] += l_g_gan

                        scaler.scale(l_g_total).backward()

                        scaler.step(self.optimizer_g)

                        scaler.update()

        else:
            self.output = self.net_g(self.lq)
            l_g_total = 0
            loss_dict = OrderedDict()
            if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
                # pixel loss
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix
                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style
                # contextual loss
                if self.cri_contextual:
                    l_g_contextual = self.cri_contextual(self.output, self.gt)
                    l_g_total += l_g_contextual
                    loss_dict['l_g_contextual'] = l_g_contextual
                if self.cri_color:
                    l_g_color = self.cri_color(self.output, self.gt)
                    l_g_total += l_g_color
                    loss_dict['l_g_color'] = l_g_color
                if self.cri_avg:
                    l_g_avg = self.cri_avg(self.output, self.gt)
                    l_g_total += l_g_avg
                    loss_dict['l_g_avg'] = l_g_avg
                # gan loss
                fake_g_pred = self.net_d(self.output)
                if isinstance (fake_g_pred,list) == False:
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    loss_dict['l_g_gan'] = l_g_gan

                    l_g_total.backward()
                    self.optimizer_g.step()

                else:
                    fake_g_preds = fake_g_pred
                    loss_dict['l_g_gan'] = 0
                    for fake_g_pred in fake_g_preds:
                        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                        l_g_total += l_g_gan
                        loss_dict['l_g_gan'] += l_g_gan

                    l_g_total.backward()
                    self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        if autoamp_d==False:
            if autoamp_g==True:
                self.output=self.output.float()
            real_d_pred = self.net_d(self.gt)
            if isinstance (real_d_pred,list) == False:
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                loss_dict['l_d_real'] = l_d_real
                loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                l_d_real.backward()
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_fake.backward()
            else:
                # real
                real_d_preds = real_d_pred
                loss_dict['l_d_real'] = 0
                loss_dict['out_d_real'] = 0
                l_d_real_tot = 0
                for real_d_pred in real_d_preds:
                    l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                    l_d_real_tot += l_d_real
                    loss_dict['l_d_real'] += l_d_real
                    loss_dict['out_d_real'] += torch.mean(real_d_pred.detach())
                l_d_real_tot.backward()
                # fake
                loss_dict['l_d_fake'] = 0
                loss_dict['out_d_fake'] = 0
                l_d_fake_tot = 0
                fake_d_preds = self.net_d(self.output.detach().clone())  # clone for pt1.9
                for fake_d_pred in fake_d_preds:
                    l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                    l_d_fake_tot += l_d_fake
                    loss_dict['l_d_fake'] += l_d_fake
                    loss_dict['out_d_fake'] += torch.mean(fake_d_pred.detach())
                l_d_fake_tot.backward()

            self.optimizer_d.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)
        else:
            with autocast():

                real_d_pred = self.net_d(self.gt)
                if isinstance (real_d_pred,list) == False:
                    l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                    loss_dict['l_d_real'] = l_d_real
                    loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                    scaler.scale(l_d_real).backward()
                    # fake
                    fake_d_pred = self.net_d(self.output.detach())
                    l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                    loss_dict['l_d_fake'] = l_d_fake
                    loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                    scaler.scale(l_d_fake).backward()

                else:
                    # real
                    real_d_preds = real_d_pred
                    loss_dict['l_d_real'] = 0
                    loss_dict['out_d_real'] = 0
                    l_d_real_tot = 0
                    for real_d_pred in real_d_preds:
                        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                        l_d_real_tot += l_d_real
                        loss_dict['l_d_real'] += l_d_real
                        loss_dict['out_d_real'] += torch.mean(real_d_pred.detach())
                    scaler.scale(l_d_real_tot).backward()
                    # fake
                    loss_dict['l_d_fake'] = 0
                    loss_dict['out_d_fake'] = 0
                    l_d_fake_tot = 0
                    fake_d_preds = self.net_d(self.output.detach().clone())  # clone for pt1.9
                    for fake_d_pred in fake_d_preds:
                        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                        l_d_fake_tot += l_d_fake
                        loss_dict['l_d_fake'] += l_d_fake
                        loss_dict['out_d_fake'] += torch.mean(fake_d_pred.detach())
                    scaler.scale(l_d_fake_tot).backward()

                scaler.step(self.optimizer_d)
                scaler.update()
                self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
