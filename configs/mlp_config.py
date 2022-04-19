class Config:
    dataset = dict(
        sources=[
            'dolg_embed',
            'backfin_778', # my
            'backfin_800', # my
            'mix_fullbody_798', # fate
            'mix_fullbody_816', # fate
            #'nfent',
            'fullbody_812', # atom
            'fullbody_768', # my
            'row_660', # my,
        ],
        batch_size=256,
        num_workers=3,
        name_id_map='extra_data/name_to_id.json'
    )

    model = dict(
        num_classes=15587,
        embedding_size=1024,
        hidden_dim=1024,
        features_mapping={
            'backfin_778': (1024, 512),
            'fullbody_768': (2048, 512)
        },
        concat_features=dataset['sources'],
        input_dim=len(dataset['sources']) * 512,
    )

    optimizer = dict(
        type='Adam',
        lr=0.001,
        weight_decay=0.0001,
        amsgrad=False
    )

    training_procedure = dict(
        epochs=40,
        n_splits=5,
        lr_drop_epochs=[16, 23, 30],
        saving_model_dir='models',
        device_name='cuda',
    )

    KNN = 100