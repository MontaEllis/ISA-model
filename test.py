import os
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

def weights_init_normal(m):
    classname = m.__class__.__name__
    gain = 0.02
    init_type = hp.init_type

    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir


source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

# output_dir_test = hp.output_dir_test
output_int_dir = hp.output_int_dir
output_float_dir = hp.output_float_dir



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
        model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32)

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

    model.apply(weights_init_normal)

    model = torch.nn.DataParallel(model, device_ids=devicess)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)


    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()

    from loss_function import Binary_Loss,DiceLoss
    criterion = Binary_Loss().cuda()


    writer = SummaryWriter(args.output_dir)



    train_dataset = MedData_train(source_train_dir,label_train_dir)
    train_loader = DataLoader(train_dataset.queue_dataset, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)



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

                x = x.type(torch.FloatTensor).cuda()
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
                
            outputs = model(x)


            # for metrics
            logits = torch.sigmoid(outputs)
            labels = logits.clone()
            labels[labels>0.5] = 1
            labels[labels<=0.5] = 0


            loss = criterion(outputs, y)

            num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1



            false_positive_rate,false_negtive_rate,dice = metric(y.cpu(),labels.cpu())
            ## log
            writer.add_scalar('Training/zLoss', loss.item(),iteration)
            writer.add_scalar('Training/false_positive_rate', false_positive_rate,iteration)
            writer.add_scalar('Training/false_negtive_rate', false_negtive_rate,iteration)
            writer.add_scalar('Training/dice', dice,iteration)
            


            print("loss:"+str(loss.item()))
            print('lr:'+str(scheduler._last_lr[0]))

            

        scheduler.step()


        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )




        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:

            torch.save(
                {
                    
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
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
                outputs = outputs[0].cpu().detach().numpy()
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


def test():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_test

    os.makedirs(output_float_dir, exist_ok=True)

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

        # from models.three_d.fcn3d import FCN_Net
        # model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class)

        #from models.three_d.highresnet import HighRes3DNet
        #model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class)

        # from models.three_d.densenet3d import SkipDenseNet3D
        # model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.vnet3d import VNet
        # model = VNet(in_channels=hp.in_class, classes=hp.out_class)

        #from models.three_d.unetr import UNETR 
        #model = UNETR(img_shape=(hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class)

        # from models.three_d.denceunet import UNet3D  #CSR
        # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32)


    model = torch.nn.DataParallel(model, device_ids=devicess)


    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])


    model.cuda()
    model.eval()



    test_dataset = MedData_test(source_test_dir,label_test_dir)
    znorm = ZNormalization()

    if hp.mode == '3d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size
    elif hp.mode == '2d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size


    for i,subj in enumerate(test_dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(
                subj,
                patch_size,
                patch_overlap,
            )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=args.batch)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
        model.eval()
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):


                input_tensor = patches_batch['source'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]

                if hp.mode == '2d':
                    input_tensor = input_tensor.squeeze(4)

                low_x = lowpass_torch(input_tensor,0.1)
                high_x = highpass_torch(input_tensor,0.05)

                outputs1,outputs2 = model(input_tensor,low_x,high_x)

                if hp.mode == '2d':
                    outputs = outputs2.unsqueeze(4)


                logits1 = outputs1.argmax(dim=1)
                # log_temp1 = torch.nn.functional.one_hot(logits1, num_classes=hp.out_class+1).permute(0,4,1,2,3)   


                logits2 = outputs2.argmax(dim=1)
                # log_temp2 = torch.nn.functional.one_hot(logits2, num_classes=hp.out_class+1).permute(0,4,1,2,3)  

                # labels = logits.clone()
                # labels[labels>0.5] = 1
                # labels[labels<=0.5] = 0

                aggregator.add_batch(logits1.unsqueeze(1), locations)
                aggregator_1.add_batch(logits2.unsqueeze(1), locations)
        output_tensor = aggregator.get_output_tensor()
        output_tensor_1 = aggregator_1.get_output_tensor()




        affine = subj['source']['affine']
        if (hp.in_class == 1) and (hp.out_class == 1) :
            if i<9:
                label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
                label_image.save(os.path.join(output_float_dir,"0"+str(i+1)+".mhd"))
                output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
                output_image.save(os.path.join(output_int_dir,"0"+str(i+1)+".mhd"))
            else:
                label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
                label_image.save(os.path.join(output_float_dir,str(i+1)+".mhd"))
                output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
                output_image.save(os.path.join(output_int_dir,str(i+1)+".mhd"))
        else:
            output_tensor = output_tensor.unsqueeze(1)
            output_tensor_1= output_tensor_1.unsqueeze(1)

            output_image_artery_float = torchio.ScalarImage(tensor=output_tensor[0].numpy(), affine=affine)
            output_image_artery_float.save(os.path.join(output_int_dir,str(i)+"_result_float_artery.mhd"))

            output_image_artery_int = torchio.ScalarImage(tensor=output_tensor_1[0].numpy(), affine=affine)
            output_image_artery_int.save(os.path.join(output_int_dir,str(i)+"_result_int_artery.mhd"))

            output_image_lung_float = torchio.ScalarImage(tensor=output_tensor[1].numpy(), affine=affine)
            output_image_lung_float.save(os.path.join(output_int_dir,str(i)+"_result_float_lung.mhd"))

            output_image_lung_int = torchio.ScalarImage(tensor=output_tensor_1[1].numpy(), affine=affine)
            output_image_lung_int.save(os.path.join(output_int_dir,str(i)+"_result_int_lung.mhd"))

            output_image_trachea_float = torchio.ScalarImage(tensor=output_tensor[2].numpy(), affine=affine)
            output_image_trachea_float.save(os.path.join(output_int_dir,str(i)+"_result_float_trachea.mhd"))

            output_image_trachea_int = torchio.ScalarImage(tensor=output_tensor_1[2].numpy(), affine=affine)
            output_image_trachea_int.save(os.path.join(output_int_dir,str(i)+"_result_int_trachea.mhd"))

            output_image_vein_float = torchio.ScalarImage(tensor=output_tensor[3].numpy(), affine=affine)
            output_image_vein_float.save(os.path.join(output_int_dir,str(i)+"_result_float_veiny.mhd"))

            output_image_vein_int = torchio.ScalarImage(tensor=output_tensor_1[3].numpy(), affine=affine)
            output_image_vein_int.save(os.path.join(output_int_dir,str(i)+"_result_int_vein.mhd"))            


   

if __name__ == '__main__':
    # if hp.train_or_test == 'train':
        # train()
    # elif hp.train_or_test == 'test':
        test()
