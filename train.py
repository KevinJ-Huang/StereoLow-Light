import os
import math
import argparse
import random
import logging
import sys
sys.path.append('/ghome/huangjie/STEN_Mid')


import torch
import torch.multiprocessing as mp

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='/ghome/huangjie/STEN_Mid/options/train/train_Enhance.yml', help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings

    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')


    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        # if opt['use_tb_logger'] and 'debug' not in opt['name']:
        #     version = float(torch.__version__[0:3])
        #     if version >= 1.1:  # PyTorch 1.1
        #         from torch.utils.tensorboard import SummaryWriter
        #     else:
        #         logger.info(
        #             'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
        #         from tensorboardX import SummaryWriter
        #     tb_logger = SummaryWriter(log_dir='/output/tb_logger' + opt['name'])

    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')



    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    # dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(opt, dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))

            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(opt, dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    best_psnr = 0
    best_step = 0
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):

        # total_lv2 = 0
        total_psnr = 0
        total_lv4 = 0
        total_sr = 0
        total_loss = 0
        print_iter = 0
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                print_iter += 1  ############################################################## new
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '

                total_loss += logs['l_total']
                total_sr += logs['l_sr']
                # total_lv2 += logs['l_lv2']
                total_lv4 += logs['l_lv4']
                total_psnr += logs['psnr']
                mean_total = total_loss / print_iter
                mean_sr = total_sr / print_iter
                # mean_lv2 = total_lv2 / print_iter
                mean_lv4 = total_lv4 / print_iter
                mean_psnr = total_psnr / print_iter
                message += '{:s}: {:.4e} '.format('mean_total_loss', mean_total)
                message += '{:s}: {:.4e} '.format('mean_sr_loss', mean_sr)
                # message += '{:s}: {:.4e} '.format('mean_lv2_loss', mean_lv2)
                message += '{:s}: {:.4e} '.format('mean_lv4_loss', mean_lv4)
                message += '{:s}: {:} '.format('mesn_psnr', mean_psnr)
                # tensorboard logger
                # if opt['use_tb_logger'] and 'debug' not in opt['name']:
                #     if rank <= 0:
                #         tb_logger.add_scalar('mean_psnr', mean_psnr, current_step)
                if rank <= 0:
                    logger.info(message)


            ##### valid test
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                pbar = util.ProgressBar(len(val_loader))
                avg_psnr = 0.
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQleft_path'][0]))[0]
                    # img_dir = os.path.join(opt['path']['val_images'], img_name)
                    img_dir = opt['path']['val_images']
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    en_img = util.tensor2img(visuals['rlt'])  # uint8
                    gt_img = util.tensor2img(visuals['GTleft'])  # uint8

                    # Save SR images for reference
                    if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))

                        util.save_img(en_img, save_img_path)

                    # calculate PSNR
                    psnr_inst = util.calculate_psnr(en_img, gt_img)
                    if math.isinf(psnr_inst) or math.isnan(psnr_inst):
                        psnr_inst = 0
                        idx -= 1

                    avg_psnr = avg_psnr + psnr_inst
                    pbar.update('Test {}'.format(img_name))

                avg_psnr = avg_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.4e} Previous best PSNR: {:.4e} Previous best step: {}'.format(avg_psnr, best_psnr, best_step))
                # tensorboard logger
                # if opt['use_tb_logger'] and 'debug' not in opt['name']:
                #     tb_logger.add_scalar('valid_psnr', avg_psnr, current_step)

                if avg_psnr > best_psnr:
                    if rank <= 0:
                        best_psnr = avg_psnr
                        best_step = current_step
                        logger.info('Saving best models!!!!!!!The best psnr is:{:4e}'.format(best_psnr))
                        model.save_best()

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        # tb_logger.close()


if __name__ == '__main__':
    main()

