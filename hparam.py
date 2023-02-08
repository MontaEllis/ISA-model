class hparams:

    train_or_test = 'train'
    output_dir = 'logs/your_program_name'
    # output_dir = 'logs/batch2'
    aug = False
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 100
    epochs_per_checkpoint = 20
    batch_size = 4
    ckpt = None
    init_lr = 0.01
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 1

    crop_or_pad_size = 64,64,64 # if 2D: 256,256,1
    patch_size = 64,64,64 # if 2D: 128,128,1 

    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0

    fold_arch = '*.mhd'

    save_arch = '.mhd'

    source_train_dir = '/data/cc/Ying-TOF/train/source'
    label_train_dir = '/data/cc/Ying-TOF/train/label1'
    source_test_dir = '/data/cc/Ying-TOF/test/source'
    label_test_dir = '/data/cc/Ying-TOF/test/label1'
    
    # source_train_dir = '/data/cc/TOF/GAN/train/source'
    # label_train_dir = '/data/cc/TOF/GAN/train/label'
    # source_test_dir = '/data/cc/TOF/GAN/test/source'
    # label_test_dir = '/data/cc/TOF/GAN/test/label'
	
	
	# source_train_dir = '/data/cc/Ying-TOF/train/source'
    # label_train_dir = '/data/cc/Ying-TOF/train/label1'
    # source_test_dir = '/data/cc/Ying-TOF/test/source'
    # label_test_dir = '/data/cc/Ying-TOF/test/label1'

    grad_clip = 0.3
    r1_lambda = 0.5

    output_int_dir = 'Results/out2'
    output_float_dir = 'Results/out1'

    init_type = 'xavier' # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]