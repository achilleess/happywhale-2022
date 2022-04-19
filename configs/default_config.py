class Config:
    dataset = dict(
        train_img_dir_fullbody="/root/autodl-tmp/happy-whale-fullbody/train_images/",
        test_img_dir_fullbody="/root/autodl-tmp/happy-whale-fullbody/test_images/",

        train_img_dir_backfin = "/root/autodl-tmp/happy-whale-fin/train_images/",
        test_img_dir_backfin = "/root/autodl-tmp/happy-whale-fin/test_images/",

        train_annos = '/root/autodl-tmp/happy-whale-fin/train.csv',
        test_annos = '/root/autodl-tmp/happy-whale-fin/test.csv',

        train_mode = 'mix', # options: backfin, fullbody, mix
        val_mode = 'backfin',
        test_mode = 'backfin',

        num_workers=11,
        batch_size = 30
    )

    train_procedure = dict(
        resume=False,
        resume_epoch=0,

        n_splits=5,
        special_split=True,

        use_wandb=True,
        device_name='cuda',
        fp16=True,
        
        model_save_dir='models',

        stage_epochs = [0, 4, 8],
        img_sizes = [768, 768, 768],

        accum_grad_steps = 4,
        epochs = 15
    )

    model = dict(
        model_name = 'tf_efficientnet_b6_ns',
        num_classes=15587,
        embedding_size = 512,
        dropout_rate=0.2
    )

    optimizer = dict(
        type='Adam',
        lr=0,
        weight_decay=0.0005
    )

    KNN = 100