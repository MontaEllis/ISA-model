import os

from models.discriminator.model import DiscriminatorCycle_new
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
devicess = [0]

import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    ZNormalization,
)
from tqdm import tqdm
from torchvision import utils
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.fft as fft
#from loss_function import Binary_Loss,DiceLossss
from torch.nn.modules.loss import CrossEntropyLoss
#criterion_dice = DiceLossss(2).cuda()
criterion_ce = CrossEntropyLoss().cuda()
source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir


source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

# output_dir_test = hp.output_dir_test



def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file, help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')  
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')

    return parser



def train():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


    from data_function import MedData_train
    os.makedirs(args.output_dir, exist_ok=True)

    if hp.mode == '2d':
        #from models.two_d.unet import Unet
        #model = Unet(in_channels=hp.in_class, classes=hp.out_class)

        from models.two_d.miniseg import MiniSeg
        model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)

        #from models.two_d.fcn import FCN32s as fcn
        #model = fcn(in_class =hp.in_class,n_class=hp.out_class)

        # from models.two_d.segnet import SegNet
        # model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class)

        #from models.two_d.deeplab import DeepLabV3
        #model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class)

        #from models.two_d.unetpp import ResNet34UnetPlus
        #model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class)

        #from models.two_d.pspnet import PSPNet
        #model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class)

    elif hp.mode == '3d':
        from models.three_d.unet3d import UNet3D
        model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class+1, init_features=32)

        # from models.three_d.residual_unet3d import UNet
        # model = UNet(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=2)

        #from models.three_d.fcn3d import FCN_Net
        #model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class)

        #from models.three_d.highresnet import HighRes3DNet
        #model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class)

        #from models.three_d.densenet3d import SkipDenseNet3D
        #model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.vnet3d import VNet
        # model = VNet(in_channels=hp.in_class, classes=hp.out_class)

        #from models.three_d.unetr import UNETR
        #model = UNETR(img_shape=(hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class)

        # from models.three_d.denceunet import UNet3D  #CSR
        # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32)

    from models.discriminator.model import DiscriminatorCycle,test
    # discriminator = test()
    discriminator = DiscriminatorCycle(in_channels=2+2+1)

    model = torch.nn.DataParallel(model, device_ids=devicess)
    discriminator = torch.nn.DataParallel(discriminator, device_ids=devicess)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=0.001)


    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    scheduler_dis = StepLR(optimizer_dis, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        discriminator.load_state_dict(ckpt["model_dis"])
        optimizer.load_state_dict(ckpt["optim"])
        optimizer_dis.load_state_dict(ckpt["optim_dis"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()
    discriminator.cuda()

    from loss_function import Binary_Loss,DiceLossss
    criterion = Binary_Loss().cuda()
    criterion_dice = DiceLossss(2).cuda()
    criterion_ce = CrossEntropyLoss().cuda()


    writer = SummaryWriter(args.output_dir)



    train_dataset = MedData_train(source_train_dir,label_train_dir)
    train_loader = DataLoader(train_dataset.queue_dataset, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    model.train()
    discriminator.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)



    def lowpass_torch(input, limit):
        pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
        pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
        # pass1 = torch.fft.fftshift(fft.rfftfreq(input.shape[-1])) < limit
        # pass2 = torch.fft.fftshift(fft.fftfreq(input.shape[-2])) < limit
        kernel = torch.outer(pass2, pass1).cuda()
        
        fft_input = fft.rfftn(input)
        return fft.irfftn(fft_input * kernel, s=input.shape[-3:])

    def highpass_torch(input, limit):

        # temp0 = input.shape[-1]
        # tmp = fft.rfftfreq(input.shape[-1])

        # temp1 = fft.fftfreq(input.shape[-2])

        pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) > limit
        pass2 = torch.abs(fft.fftfreq(input.shape[-2])) > limit
        # pass1 = torch.fft.fftshift(fft.rfftfreq(input.shape[-1])) > limit
        # pass2 = torch.fft.fftshift(fft.fftfreq(input.shape[-2])) > limit
        kernel = torch.outer(pass2, pass1).cuda()
        
        fft_input = fft.rfftn(input)
        return fft.irfftn(fft_input * kernel, s=input.shape[-3:])



    for epoch in range(1, epochs + 1):
        print("epoch:"+str(epoch))
        epoch += elapsed_epochs

        num_iters = 0


        for i, batch in enumerate(train_loader):
            

            if hp.debug:
                if i >=1:
                    break

            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")

            optimizer.zero_grad()


            if (hp.in_class == 1) and (hp.out_class == 1) :
                x = batch['source']['data']
                y = batch['label']['data']

                #y[y!=0] = 1 

                # x = x.type(torch.FloatTensor).cuda()
                # y = y.type(torch.FloatTensor).cuda()
                #y[y!=0] = 1 
                y_back = torch.zeros_like(y)
                # y_back[(y==0) ^ (y_L_TL==0) ^ (y_R_TL==0)]=1
                y_back[(y==0)]=1


                x = x.type(torch.FloatTensor).cuda()
                y = torch.cat((y_back, y),1) 
                y = y.type(torch.FloatTensor).cuda()
            else:
                x = batch['source']['data']
                y_atery = batch['atery']['data']
                y_lung = batch['lung']['data']
                y_trachea = batch['trachea']['data']
                y_vein = batch['atery']['data']

                x = x.type(torch.FloatTensor).cuda()

                y = torch.cat((y_atery,y_lung,y_trachea,y_vein),1) 
                y = y.type(torch.FloatTensor).cuda()


            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)

            y[y!=0] = 1
                
                #print(y.max())

            

            low_x = lowpass_torch(x,0.1)
            high_x = highpass_torch(x,0.05)

            # TRAIN DISCRIMINATOR
            # y.requires_grad = True
            # x.requires_grad = True
            
            #11111
            temp_xy = torch.cat([x, y, y], dim=1)
            # temp_xy = torch.cat([x, y], dim=1)
            # temp_xy = x*y*y
            # temp_xy = x*y



            # temp_xy = x*y
            temp_xy.requires_grad = True
            y.requires_grad = True                  

            # r_preds = discriminator(temp_xy,y) # fixed

            #Cheng
            # outputs = model(x)
            # logits = torch.sigmoid(outputs)
            # scale = y.clone()

       
            # # scale = y.clone()        
            # scale[(scale == 1) & (logits<=0.8)] = 0.8
            # scale[(scale == 0) & (logits>0.2)] = 0.2
            # scale = torch.where((scale == 1) & (logits>0.8),logits, scale)
            # scale = torch.where((scale == 0) & (logits<=0.2),logits, scale)
            r_preds = discriminator(temp_xy) # fixed
          

            outputs1, outputs2 = model(x,low_x,high_x)
            # logits1 = torch.sigmoid(outputs1)
            # logits2 = torch.sigmoid(outputs2)
            if hp.r1_lambda > 0:
                # Gradient penalty
                # grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                # grad_real = torch.autograd.grad(outputs=r_preds.sum(), inputs=[temp_xy,scale], create_graph=True,allow_unused=True)[0]
                grad_real = torch.autograd.grad(outputs=r_preds.sum(), inputs=temp_xy, create_graph=True,allow_unused=True)[0]    
                # grad_real_1 = torch.autograd.grad(outputs=r_preds.sum(), inputs=scale, create_graph=True,allow_unused=True)[0]           
               
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                # grad_penalty_1 = (grad_real_1.view(grad_real_1.size(0), -1).norm(2, dim=1) ** 2).mean()
                # grad_penalty = 10 * grad_penalty+ 10*grad_penalty_1
                grad_penalty = 10*grad_penalty
            else:
                grad_penalty = 0

            # log_temp = logits1.clone()
            # log_temp[log_temp>0.5] =1
            # log_temp[log_temp<=0.5] =0

            # log_temp1 = logits1.clone()
            # log_temp1[log_temp1>0.5] =1
            # log_temp1[log_temp1<=0.5] =0

            # log_temp2 = logits2.clone()
            # log_temp2[log_temp2>0.5] =1
            # log_temp2[log_temp2<=0.5] =0

            logits1 = outputs1.argmax(dim=1)
            log_temp1 = torch.nn.functional.one_hot(logits1, num_classes=hp.out_class+1).permute(0,4,1,2,3)   


            logits2 = outputs2.argmax(dim=1)
            log_temp2 = torch.nn.functional.one_hot(logits2, num_classes=hp.out_class+1).permute(0,4,1,2,3)  


            # ## 0314
            # logits[y==1 & logits <0.8] = 0.8 
            # logits[y==0 & logits >0.2] = 0.2 
            # g_preds = discriminator(log_temp*x,logits)
            # g_preds = discriminator(log_temp*x,logits)

            #11111
            # g_preds = discriminator(torch.cat([x, logits1, logits2], dim=1))
            g_preds = discriminator(torch.cat([x, log_temp1, log_temp2], dim=1))
            # g_preds = discriminator(torch.cat([x, logits1], dim=1))
            # g_preds = discriminator(x*logits1*logits2)
            #12341234
            # g_preds = discriminator(x*logits1)
            # g_preds = discriminator(x*log_temp)


            d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty


            writer.add_scalar('D/D-Loss', d_loss.item(), iteration)
            optimizer_dis.zero_grad()
            d_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), hp.grad_clip)
            optimizer_dis.step()

            # TRAIN GENERATOR
            outputs1, outputs2 = model(x,low_x,high_x)
            # logits1 = torch.sigmoid(outputs1)
            # logits2 = torch.sigmoid(outputs2)

            # g_preds_ = discriminator(log_temp*x,logits)

            #11111
            # g_preds_ = discriminator(x*logits1*logits2)
            # g_preds_ = discriminator(x*logits1)
            # g_preds_ = discriminator(torch.cat([x, logits1, logits2], dim=1))
            logits1 = outputs1.argmax(dim=1)
            log_temp1 = torch.nn.functional.one_hot(logits1, num_classes=hp.out_class+1).permute(0,4,1,2,3)   


            logits2 = outputs2.argmax(dim=1)
            log_temp2 = torch.nn.functional.one_hot(logits2, num_classes=hp.out_class+1).permute(0,4,1,2,3)  


            g_preds_ = discriminator(torch.cat([x, log_temp1, log_temp2], dim=1))
            # g_preds_ = discriminator(torch.cat([x, logits1], dim=1))

            # g_preds_ = discriminator(log_temp*x,log_temp*x)

            g_preds_ = torch.topk(g_preds_,k=2,dim=0).values
            
########################################################################################################################################
            #weight = math.exp(-0.01*(20-epoch)**1.5)
            g_loss = torch.nn.functional.softplus(-g_preds_).mean() + criterion_ce(outputs1, y.argmax(dim=1)) + criterion_ce(outputs2, y.argmax(dim=1))
               
            # g_loss = torch.nn.functional.softplus(-g_preds_).mean() +  criterion_ce(outputs1, y.argmax(dim=1)) + criterion_ce(outputs2, y.argmax(dim=1)) + criterion_dice(outputs2, y.argmax(dim=1)) + criterion_dice(outputs1, y.argmax(dim=1)) 
             
            # criterion(outputs1, y) + criterion(outputs2, y)

            writer.add_scalar('G/G-Loss', g_loss.item(), iteration)
########################################################################################################################################

            optimizer.zero_grad()

            g_loss.backward()
            optimizer.step()

            # outputs1, outputs2 = model(x,low_x,high_x)
            # # for metrics
            # logits = torch.sigmoid(outputs2)           
            # labels = logits.clone()
            # labels[labels>0.5] = 1
            # labels[labels<=0.5] = 0

            # logits11 = torch.sigmoid(outputs1)      
            # labels11 = logits11.clone()
            # labels11[labels11>0.5] = 1
            # labels11[labels11<=0.5] = 0


            # loss = criterion(outputs, y)

            num_iters += 1
            # loss.backward()

            # optimizer.step()
            iteration += 1

            y_argmax = y.argmax(dim=1)
            y_one_hot = torch.nn.functional.one_hot(y_argmax, num_classes=hp.out_class+1).permute(0,4,1,2,3)


            false_positive_rate,false_negtive_rate,dice = metric(y_one_hot[:,1:,:,:].cpu(),log_temp1[:,1:,:,:].cpu())
            false_positive_rate1,false_negtive_rate1,dice1 = metric(y_one_hot[:,1:,:,:].cpu(),log_temp2[:,1:,:,:].cpu())

            ## log
            writer.add_scalar('Training/Loss', (criterion(outputs1, y.argmax(dim=1))+criterion(outputs2, y.argmax(dim=1))).item(),iteration)
            # writer.add_scalar('Training/false_positive_rate', false_positive_rate,iteration)
            # writer.add_scalar('Training/false_negtive_rate', false_negtive_rate,iteration)
            writer.add_scalar('Training/dice-output2', dice,iteration)

            # writer.add_scalar('Training/Loss', (criterion(outputs1, y)+criterion(outputs2, y)).item(),iteration)
            writer.add_scalar('Training/dice-output1', dice1,iteration)


            


            # print("loss:"+str(loss.item()))
            print('lr:'+str(scheduler._last_lr[0]))

            

        scheduler.step()


        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,
                "model_dis": discriminator.state_dict(),
                "optim_dis": optimizer_dis.state_dict(),

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )




        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:

            torch.save(
                {
                    
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "epoch": epoch,
                    "model_dis": discriminator.state_dict(),
                    "optim_dis": optimizer_dis.state_dict(),
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )
        


            
            with torch.no_grad():
                if hp.mode == '2d':
                    x = x.unsqueeze(4)
                    y = y.unsqueeze(4)
                    outputs = outputs.unsqueeze(4)
                    
                x = x[0].cpu().detach().numpy()
                y = y[0].cpu().detach().numpy()
                outputs = outputs2[0].cpu().detach().numpy()
                affine = batch['source']['affine'][0].numpy()




                if (hp.in_class == 1) and (hp.out_class == 1) :
                    source_image = torchio.ScalarImage(tensor=x, affine=affine)
                    source_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-source"+hp.save_arch))
                    # source_image.save(os.path.join(args.output_dir,("step-{}-source.mhd").format(epoch)))

                    label_image = torchio.ScalarImage(tensor=y, affine=affine)
                    label_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt"+hp.save_arch))

                    output_image = torchio.ScalarImage(tensor=outputs, affine=affine)
                    output_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict"+hp.save_arch))
                else:
                    y = np.expand_dims(y, axis=1)
                    outputs = np.expand_dims(outputs, axis=1)

                    source_image = torchio.ScalarImage(tensor=x, affine=affine)
                    source_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-source"+hp.save_arch))

                    label_image_artery = torchio.ScalarImage(tensor=y[0], affine=affine)
                    label_image_artery.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt_artery"+hp.save_arch))

                    output_image_artery = torchio.ScalarImage(tensor=outputs[0], affine=affine)
                    output_image_artery.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict_artery"+hp.save_arch))

                    label_image_lung = torchio.ScalarImage(tensor=y[1], affine=affine)
                    label_image_lung.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt_lung"+hp.save_arch))

                    output_image_lung = torchio.ScalarImage(tensor=outputs[1], affine=affine)
                    output_image_lung.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict_lung"+hp.save_arch))

                    label_image_trachea = torchio.ScalarImage(tensor=y[2], affine=affine)
                    label_image_trachea.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt_trachea"+hp.save_arch))

                    output_image_trachea = torchio.ScalarImage(tensor=outputs[2], affine=affine)
                    output_image_trachea.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict_trachea"+hp.save_arch))

                    label_image_vein = torchio.ScalarImage(tensor=y[3], affine=affine)
                    label_image_vein.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt_vein"+hp.save_arch))

                    output_image_vein = torchio.ScalarImage(tensor=outputs[3], affine=affine)
                    output_image_vein.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict_vein"+hp.save_arch))           


    writer.close()




if __name__ == '__main__':
    train()
