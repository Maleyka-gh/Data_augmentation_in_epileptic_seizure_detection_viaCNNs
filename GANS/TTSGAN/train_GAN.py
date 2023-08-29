from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
# import models_search
# import datasets
from dataLoader import *
from GANModels import *
from functions import train, train_d, validate, save_samples, LinearLrDecay, load_params, copy_params, cur_stages, \
    mse_calc, compute_cos_similarity, DTW_calculation
from utils.utils import set_log_dir, save_checkpoint, create_logger
# from utils.inception_score import _init_inception
# from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils import data
import os
import numpy as np
import torch.nn as nn
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random
import matplotlib.pyplot as plt
import warnings
import io
import PIL.Image
from torchvision.transforms import ToTensor


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()

    #     _init_inception()
    #     inception_path = check_or_download_inception(None)
    #     create_inception_graph(inception_path)

    if args.seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print('WE ARE', ngpus_per_node)
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        print('printing ')
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        #         elif classname.find('Linear') != -1:
        #             if args.init_type == 'normal':
        #                 nn.init.normal_(m.weight.data, 0.0, 0.02)
        #             elif args.init_type == 'orth':
        #                 nn.init.orthogonal_(m.weight.data)
        #             elif args.init_type == 'xavier_uniform':
        #                 nn.init.xavier_uniform(m.weight.data, 1.)
        #             else:
        #                 raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # import network
    # args.distributed= True
    gen_net = Generator(patch_size=args.patch_size, embed_dim=args.embed_dim,
                        latent_dim=args.latent_dim)
    print(gen_net)
    dis_net = Discriminator(patch_size=args.patch_size, emb_size=args.embed_dim)

    print(dis_net)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            #             gen_net = eval('models_search.'+args.gen_model+'.Generator')(args=args)
            #             dis_net = eval('models_search.'+args.dis_model+'.Discriminator')(args=args)

            gen_net.apply(weights_init)
            dis_net.apply(weights_init)
            gen_net.cuda(args.gpu)
            dis_net.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.dis_batch_size = int(args.dis_batch_size / ngpus_per_node)
            args.gen_batch_size = int(args.gen_batch_size / ngpus_per_node)
            args.batch_size = args.dis_batch_size

            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net, device_ids=[args.gpu],
                                                                find_unused_parameters=True)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net, device_ids=[args.gpu],
                                                                find_unused_parameters=True)
        else:
            gen_net.cuda()
            dis_net.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        gen_net.cuda(args.gpu)
        dis_net.cuda(args.gpu)
    else:
        print("THIS")
        gen_net = torch.nn.DataParallel(gen_net).cuda()
        dis_net = torch.nn.DataParallel(dis_net).cuda()
    print(dis_net) if args.rank == 0 else 0

    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                              args.g_lr, weight_decay=args.wd)

    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # fid stat
    #     if args.dataset.lower() == 'cifar10':
    #         fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    #     elif args.dataset.lower() == 'stl10':
    #         fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    #     elif args.fid_stat is not None:
    #         fid_stat = args.fid_stat
    #     else:
    #         raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    #     assert os.path.exists(fid_stat)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic

    #     dataset = datasets.ImageDataset(args, cur_img_size=8)
    #     train_loader = dataset.train
    #     train_sampler = dataset.train_sampler

    #     train_set = unimib_load_dataset(incl_xyz_accel = True, incl_rms_accel = False, incl_val_group = False, one_hot_encode = False, data_mode = 'Train')
    #     test_set = unimib_load_dataset(incl_xyz_accel = True, incl_rms_accel = False, incl_val_group = False, one_hot_encode = False, data_mode = 'Test')
    #     train_loader = data.DataLoader(train_set, batch_size=args.dis_batch_size, num_workers=args.num_workers, shuffle=True)
    #     test_loader = data.DataLoader(test_set, batch_size=args.dis_batch_size, num_workers=args.num_workers, shuffle=True)
    # (class_data , class_labels)
    train_set = Mond_data_loader(data_mode='Train', is_normalize=False,
                                 data_path='/data/Data_prep_for_gan/numpy_tts_seiz_numpy_0overlap')

    # print(train_set)
    # train_set = unimib_load_dataset(incl_xyz_accel = True, incl_rms_accel = False, incl_val_group = False, is_normalize = True, one_hot_encode = False, data_mode = 'Train', single_class = True, class_name = args.class_name, augment_times=args.augment_times)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    test_set = Mond_data_loader(data_mode='Test', is_normalize=False,
                                data_path='/data/Data_prep_for_gan/numpy_tts_seiz_numpy_0overlap')
    # test_set = unimib_load_dataset(incl_xyz_accel = True, incl_rms_accel = False, incl_val_group = False, is_normalize = True, one_hot_encode = False, data_mode = 'Test', single_class = True, class_name = args.class_name)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    print(len(train_loader))

    if args.max_iter:
        # args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))
        args.max_epoch = 500
    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # set writer
    writer = None
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']

        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])

        #         avg_gen_net = deepcopy(gen_net)
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']
        #         del avg_gen_net
        #         gen_avg_param = list(p.cuda().to(f"cuda:{args.gpu}") for p in gen_avg_param)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path']) if args.rank == 0 else None
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path']) if args.rank == 0 else None
        del checkpoint
    else:
        # create new log dir
        assert args.exp_name
        if args.rank == 0:
            args.path_helper = set_log_dir('logs', args.exp_name)
            logger = create_logger(args.path_helper['log_path'])
            writer = SummaryWriter(args.path_helper['log_path'])

    if args.rank == 0:
        logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    best_mse = 1000
    best_dtw = 1000
    best_cos_sim = 0

    # train loop
    epoch_mse, avg_max, total_max = 0, 0, 0
    mse_list = []
    cos_sim_list = []
    mean_similarity = 100000
    mean_dtw = 100000
    dtw_list = []
    gen_loss_list = []
    dis_loss_list = []
    for epoch in range(int(start_epoch), int(args.max_epoch)):
        #         train_sampler.set_epoch(epoch)
        print('EPOCHH')
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage)) if args.rank == 0 else 0
        print(f"path: {args.path_helper['prefix']}") if args.rank == 0 else 0

        #         if (epoch+1) % 3 == 0:
        #             # train discriminator and generator both
        #             train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,fixed_z, lr_schedulers)
        #         else:
        #             #only train discriminator
        #             train_d(args, gen_net, dis_net, dis_optimizer, train_loader, epoch, writer_dict,fixed_z, lr_schedulers)

        gen_loss, diss_loss = train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
                                    epoch, writer_dict, fixed_z, lr_schedulers)

        print(gen_loss, diss_loss)
        if args.rank == 0 and args.show:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            save_samples(args, fixed_z, fid_stat, epoch, gen_net, writer_dict)
            load_params(gen_net, backup_param, args)

        if epoch % 10 == 0:
            dtw_distance, mean_dtw = DTW_calculation(args, gen_net, test_set)

            x_similarity, y_similarity, z_similarity, mean_similarity = compute_cos_similarity(args, gen_net, test_set)

            epoch_mse, avg_max, total_max = mse_calc(args, gen_net, test_set)

        if epoch == int(args.max_epoch) - 1:
            dtw_distance, mean_dtw = DTW_calculation(args, gen_net, test_set)

            x_similarity, y_similarity, z_similarity, mean_similarity = compute_cos_similarity(args, gen_net, test_set)

            epoch_mse, avg_max, total_max = mse_calc(args, gen_net, test_set)

        # fid_stat is not defined  It doesn't make sense to use image evaluate matrics
        #         if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:
        #             backup_param = copy_params(gen_net)
        #             load_params(gen_net, gen_avg_param, args, mode="cpu")
        #             inception_score, fid_score = validate(args, fixed_z, fid_stat, epoch, gen_net, writer_dict)
        #             if args.rank==0:
        #                 logger.info(f'Inception score: {inception_score}, FID score: {fid_score} || @ epoch {epoch}.')
        #             load_params(gen_net, backup_param, args)
        #             if fid_score < best_fid:
        #                 best_fid = fid_score
        #                 is_best = True
        #             else:
        #                 is_best = False.
        #         else:
        #             is_best = False

        # TO DO: Validate add synthetic data plot in tensorboard
        # Plot synthetic data every 5 epochs
        #         if epoch and epoch % 1 == 0:
        gen_net.eval()

        plot_buf = gen_plot(gen_net, epoch, args.class_name, args.latent_dim)
        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        # writer = SummaryWriter(comment='synthetic signals')
        writer.add_image('Image', image[0], epoch)

        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
        #         if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #                 and args.rank == 0):
        # Add module in model saving code exp'gen_net.module.state_dict()' to solve the model loading unpaired name problem

        if epoch_mse < best_mse:
            best_mse = epoch_mse
            is_best1 = 0
        else:
            is_best1 = -1

            ## Comparing the DTW
        if mean_dtw < best_dtw:
            best_dtw = mean_dtw
            is_best2 = 1
        else:
            is_best2 = -1

        ## Comparing the Cos_similarity
        if mean_similarity >= best_cos_sim:
            best_cos_sim = mean_similarity
            is_best3 = 2
        else:
            is_best3 = -1

        # is_best = -1

        # print('#'*10)
        print('best_mse: ', best_mse)
        print('epoch_mse: ', epoch_mse)
        print('avg_max: ', avg_max)
        print('total_max: ', total_max)
        print('mean_similarity: ', mean_similarity)
        print('mean_dtw: ', mean_dtw)
        mse_list.append(epoch_mse)
        dtw_list.append(mean_dtw)
        cos_sim_list.append(mean_similarity)
        gen_loss_list.append(gen_loss)
        dis_loss_list.append(diss_loss)
        # print('aaa')
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
            'gen_state_dict': gen_net.module.state_dict(),
            'dis_state_dict': dis_net.module.state_dict(),
            'avg_gen_state_dict': avg_gen_net.module.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'path_helper': args.path_helper,
            'fixed_z': fixed_z,
            'generator_loss': gen_loss_list,
            'discriminator_loss': dis_loss_list,
            'best_mse': best_mse,
            'avg_max': avg_max,
            'total_max': total_max,
            'dtw_list': dtw_list,
            'cos_sim_list': cos_sim_list,
            'mse_list': mse_list,
            'mean_similarity': mean_similarity,
            'mean_dtw': mean_dtw,
            'epoch_mse': epoch_mse

        }, (is_best1, is_best2, is_best3), args.path_helper['ckpt_path'], filename="checkpoint")
        del avg_gen_net

        # break

    print(best_dtw, best_mse, best_cos_sim, file=open("output.txt", "w"))
    print('Hello')


def gen_plot(gen_net, epoch, class_name, lat_dim):
    """Create a pyplot plot and save to buffer."""
    synthetic_data = []

    for i in range(10):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, lat_dim)))
        fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
        synthetic_data.append(fake_sigs)

    fig, axs = plt.subplots(2, 5, figsize=(20, 5))
    fig.suptitle(f'Synthetic {class_name} at epoch {epoch}', fontsize=30)
    for i in range(2):
        for j in range(5):
            axs[i, j].plot(synthetic_data[i * 5 + j][0][0][0][:])
            axs[i, j].plot(synthetic_data[i * 5 + j][0][1][0][:])
            axs[i, j].plot(synthetic_data[i * 5 + j][0][2][0][:])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf


if __name__ == '__main__':
    main()

