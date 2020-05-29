import argparse
import os
import json
import datetime
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer
from ignite.metrics import RunningAverage

from models import Encoder
from models import Decoder
from models.utils import apply_spectral_norm
from utils import load_json
from utils import check_manual_seed
from utils import norm
from utils import denorm
from utils import get_output_dir_path
from dataio import get_data_loader
from dataio.settings import TRAIN_PATIENT_IDS
from dataio.settings import TEST_PATIENT_IDS
import functions.pytorch_ssim as pytorch_ssim


def calc_latent_dim(config):
    return (
        config.dataset.batch_size,
        config.model.z_dim,
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters))),
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters)))
    )


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    if config.dataset.patient_ids == 'TRAIN_PATIENT_IDS':
        patient_ids = TRAIN_PATIENT_IDS
    elif config.dataset.patient_ids == 'TEST_PATIENT_IDS':
        patient_ids = TEST_PATIENT_IDS
    else:
        raise NotImplementedError

    data_loader = get_data_loader(mode=config.dataset.mode,
                                  dataset_name=config.dataset.name,
                                  patient_ids=patient_ids,
                                  root_dir_path=config.dataset.root_dir_path,
                                  use_augmentation=config.dataset.use_augmentation,
                                  batch_size=config.dataset.batch_size,
                                  num_workers=config.dataset.num_workers,
                                  image_size=config.dataset.image_size)

    E = Encoder(input_dim=config.model.input_dim,
                z_dim=config.model.z_dim,
                filters=config.model.enc_filters,
                activation=config.model.enc_activation).float()

    D = Decoder(input_dim=config.model.input_dim,
                z_dim=config.model.z_dim,
                filters=config.model.dec_filters,
                activation=config.model.dec_activation,
                final_activation=config.model.dec_final_activation).float()

    if config.model.enc_spectral_norm:
        apply_spectral_norm(E)

    if config.model.dec_spectral_norm:
        apply_spectral_norm(D)

    if config.training.use_cuda:
        E.cuda()
        D.cuda()
        E = nn.DataParallel(E)
        D = nn.DataParallel(D)

    if config.model.saved_E:
        print(config.model.saved_E)
        E.load_state_dict(torch.load(config.model.saved_E))

    if config.model.saved_D:
        print(config.model.saved_D)
        D.load_state_dict(torch.load(config.model.saved_D))

    print(E)
    print(D)

    e_optim = optim.Adam(filter(lambda p: p.requires_grad, E.parameters()),
                         config.optimizer.enc_lr, [0.9, 0.9999])

    d_optim = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()),
                         config.optimizer.dec_lr, [0.9, 0.9999])

    alpha = config.training.alpha
    beta = config.training.beta
    margin = config.training.margin

    batch_size = config.dataset.batch_size
    fixed_z = torch.randn(calc_latent_dim(config))

    if 'ssim' in config.training.loss:
        ssim_loss = pytorch_ssim.SSIM(window_size=11)

    def l_recon(recon: torch.Tensor, target: torch.Tensor):
        if config.training.loss == 'l2':
            loss = F.mse_loss(recon, target, reduction='sum')

        elif config.training.loss == 'l1':
            loss = F.l1_loss(recon, target, reduction='sum')

        elif config.training.loss == 'ssim':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon)

        elif config.training.loss == 'ssim+l1':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon) \
                 + F.l1_loss(recon, target, reduction='sum')

        elif config.training.loss == 'ssim+l2':
            loss = (1.0 - ssim_loss(recon, target)) * torch.numel(recon) \
                 + F.mse_loss(recon, target, reduction='sum')

        else:
            raise NotImplementedError

        return beta * loss / batch_size

    def l_reg(mu: torch.Tensor, log_var: torch.Tensor):
        loss = - 0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var))
        return loss / batch_size

    def update(engine, batch):
        E.train()
        D.train()

        image = norm(batch['image'])

        if config.training.use_cuda:
            image = image.cuda(non_blocking=True).float()
        else:
            image = image.float()

        e_optim.zero_grad()
        d_optim.zero_grad()

        z, z_mu, z_logvar = E(image)
        x_r = D(z)

        l_vae_reg = l_reg(z_mu, z_logvar)
        l_vae_recon = l_recon(x_r, image)
        l_vae_total = l_vae_reg + l_vae_recon

        l_vae_total.backward()

        e_optim.step()
        d_optim.step()

        if config.training.use_cuda:
            torch.cuda.synchronize()

        return {
            'TotalLoss': l_vae_total.item(),
            'EncodeLoss': l_vae_reg.item(),
            'ReconLoss': l_vae_recon.item(),
        }

    output_dir = get_output_dir_path(config)
    trainer = Engine(update)
    timer = Timer(average=True)

    monitoring_metrics = ['TotalLoss', 'EncodeLoss', 'ReconLoss']

    for metric in monitoring_metrics:
        RunningAverage(
            alpha=0.98,
            output_transform=partial(lambda x, metric: x[metric], metric=metric)
        ).attach(trainer, metric)

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    @trainer.on(Events.STARTED)
    def save_config(engine):
        config_to_save = defaultdict(dict)

        for key, child in config._asdict().items():
            for k, v in child._asdict().items():
                config_to_save[key][k] = v

        config_to_save['seed'] = seed
        config_to_save['output_dir'] = output_dir

        print('Training starts by the following configuration: ', config_to_save)

        if needs_save:
            save_path = os.path.join(output_dir, 'config.json')
            with open(save_path, 'w') as f:
                json.dump(config_to_save, f)

    @trainer.on(Events.ITERATION_COMPLETED)
    def show_logs(engine):
        if (engine.state.iteration - 1) % config.save.log_iter_interval == 0:
            columns = ['epoch', 'iteration'] + list(engine.state.metrics.keys())
            values = [str(engine.state.epoch), str(engine.state.iteration)] \
                   + [str(value) for value in engine.state.metrics.values()]

            message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                                  max_epoch=config.training.n_epochs,
                                                                  i=engine.state.iteration,
                                                                  max_i=len(data_loader))

            for name, value in zip(columns, values):
                message += ' | {name}: {value}'.format(name=name, value=value)

            pbar.log_message(message)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_logs(engine):
        if needs_save:
            fname = os.path.join(output_dir, 'logs.tsv')
            columns = ['epoch', 'iteration'] + list(engine.state.metrics.keys())
            values = [str(engine.state.epoch), str(engine.state.iteration)] \
                   + [str(value) for value in engine.state.metrics.values()]

            with open(fname, 'a') as f:
                if f.tell() == 0:
                    print('\t'.join(columns), file=f)
                print('\t'.join(values), file=f)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(
            engine.state.epoch, timer.value())
        )
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_images(engine):
        if needs_save:
            if engine.state.epoch % config.save.save_epoch_interval == 0:
                image = norm(engine.state.batch['image'])

                with torch.no_grad():
                    z, _, _ = E(image)
                    x_r = D(z)
                    x_p = D(fixed_z)

                image = denorm(image).detach().cpu()
                x_r = denorm(x_r).detach().cpu()
                x_p = denorm(x_p).detach().cpu()

                image = image[:config.save.n_save_images, ...]
                x_r = x_r[:config.save.n_save_images, ...]
                x_p = x_p[:config.save.n_save_images, ...]

                save_path = os.path.join(output_dir, 'result_{}.png'.format(engine.state.epoch))
                save_image(torch.cat([image, x_r, x_p]).data, save_path)

    if needs_save:
        checkpoint_handler = ModelCheckpoint(
            output_dir,
            config.save.study_name,
            save_interval=config.save.save_epoch_interval,
            n_saved=config.save.n_saved,
            create_dir=True,
        )
        trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                                  handler=checkpoint_handler,
                                  to_save={'E': E, 'D': D})

    timer.attach(trainer,
                 start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    print('Training starts: [max_epochs] {}, [max_iterations] {}'.format(
        config.training.n_epochs, config.training.n_epochs * len(data_loader))
    )

    trainer.run(data_loader, config.training.n_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)
