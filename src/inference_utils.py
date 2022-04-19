import pickle
from collections import defaultdict

import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import pandas as pd

from .dataset import get_loaders, get_test_loader
#from .model import get_model_optimizer_criterion
#from .utils import set_seed, to_device, get_ind_id_map


def collect_loader_features(config, loader, model, 
                    col_embs=True, col_targets=True, col_img_names=False):
    to_np = lambda x: x.detach().cpu().numpy()
    emmbedings = []
    targets = []
    img_names = []
    for batch_i, batch in enumerate(tqdm(loader)):
        # TODO: there is a leakage without targets conversion to the gpu and back
        images, cur_targets = to_device([batch['images'], batch['targets']], config)
        with torch.no_grad(), torch.cuda.amp.autocast():
            emb = model(images)
        if col_embs:
            emmbedings.append(to_np(emb))
        if col_targets:
            targets.append(to_np(cur_targets))
        if col_img_names:
            img_names.extend(batch['image_codes'])
        #if batch_i == 15:
        #    break
    emmbs = np.concatenate(emmbedings, axis=0) if col_embs else None
    targets = np.concatenate(targets, axis=0) if col_targets else None
    return emmbs, targets, img_names


def get_predictions(test_df, threshold=0.2):
    sample_list = ['938b7e931166', '5bf17305f073', '7593d2aee842', '7362d7a01d00','956562ff2888']
    predictions = defaultdict(lambda: [[], False])
    for i, row in tqdm(test_df.iterrows()):
        img_name = row.image
        if len(predictions[img_name][0]) == 5:
            continue
        elif row.confidence > threshold:
            predictions[img_name][0].append(row.target)
        else:
            new_flag = predictions[img_name][1]
            if not new_flag:
                if len(predictions[img_name][0]) < 4:
                    predictions[img_name][0].append('new_individual')
                    predictions[img_name][0].append(row.target)
                else:
                    predictions[img_name][0].append('new_individual')
                predictions[img_name][1] = True
            else:
                predictions[img_name][0].append(row.target)

    final_preds = {}
    for img_name in tqdm(predictions):
        preds, new_flag = predictions[img_name]

        if len(preds) < 5 and not new_flag:
            preds.append('new_individual')

        if len(preds) < 5:
            remaining = [y for y in sample_list if y not in preds]
            preds = preds + remaining
            preds = preds[:5]
        final_preds[img_name] = preds
    return final_preds


def map_per_image(label, predictions):
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def calc_score(val_targets_df, all_preds, th):
    for i, row in val_targets_df.iterrows():
        target = row.target
        preds = all_preds[row.image]
        val_targets_df.loc[i,th] = map_per_image(target, preds)
    cv = val_targets_df[th].mean()
    return cv


def find_best_threshold(config, train_embs, train_targets, target_encodings,
                        val_embs, val_targets, val_names):
    neigh_model = NearestNeighbors(n_neighbors=config.KNN, metric='cosine')
    neigh_model.fit(train_embs)

    neigh_dists, neigh_idxs = neigh_model.kneighbors(val_embs, config.KNN, return_distance=True)

    val_targets_df = pd.DataFrame(
        np.stack([val_names, val_targets], axis=1),
        columns=['image', 'target']
    )
    val_targets_df['target'] = val_targets_df['target'].astype(int).map(target_encodings)

    # fill unseen val targets with 'new_individual'
    allowed_targets = set([target_encodings[x] for x in np.unique(train_targets)])
    val_targets_df.loc[~val_targets_df.target.isin(allowed_targets), 'target'] = 'new_individual'
    val_targets_df.target.value_counts()

    val_df = []
    for i, val_name in tqdm(enumerate(val_names)):
        cur_neigh_targets = train_targets[neigh_idxs[i]]
        cur_neigh_distances = neigh_dists[i]
        subset_preds = pd.DataFrame(
            np.stack([cur_neigh_targets, cur_neigh_distances], axis=1),
            columns=['target', 'distances']
        )
        subset_preds['image'] = val_name
        val_df.append(subset_preds)

    val_df = pd.concat(val_df).reset_index(drop=True)
    val_df['confidence'] = 1 - val_df['distances']

    val_df = val_df.groupby(['image', 'target']).confidence.max().reset_index()
    val_df = val_df.sort_values('confidence', ascending=False).reset_index(drop=True)
    val_df['target'] = val_df['target'].map(target_encodings)

    ## Compute CV
    best_th = 0
    best_cv = 0
    ths = [0.1 * x for x in range(11)]
    ths = np.arange(0.44, 0.51, 0.01)
    print(ths)
    for th in ths:
        all_preds = get_predictions(val_df, threshold=th)
        cv = calc_score(val_targets_df, all_preds, th)
        
        print(f"CV at threshold {th}: {cv}")
        if cv > best_cv:
            best_th = th
            best_cv = cv

    print("Best threshold",best_th)
    print("Best cv",best_cv)

    ## Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
    val_targets_df['is_new_individual'] = val_targets_df.target == 'new_individual'
    print(val_targets_df.is_new_individual.value_counts().to_dict())

    val_scores = val_targets_df.groupby('is_new_individual').mean().T
    val_scores['adjusted_cv'] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    best_threshold_adjusted = val_scores['adjusted_cv'].idxmax()
    print("best_threshold", best_threshold_adjusted)
    return best_threshold_adjusted


def inference_on_test(config, train, test, skf_splits, args, ind_id_map):
    sample_list = ['938b7e931166', '5bf17305f073', '7593d2aee842', '7362d7a01d00','956562ff2888']
    target_encodings = {j: i for i, j in ind_id_map.items()}

    if config.device_name in ['cuda', 'cpu']:
        config.device = torch.device(config.device_name)
    elif config.device_name == 'tpu':
        assert 0, 'there is no option to inference on tpu'
    else:
        assert 0, 'unrecognized device'

    preload_emmbedings = args.preload
    #preload_tag = '_fullbody' if config.full_body else '_fin'
    preload_tag = ''

    fold_thresholds = []
    df_train_list = []
    df_test_list = []
    for fold, (train_i, valid_i) in enumerate(skf_splits):
        if not fold in [0]:
            continue

        print("~"*8, f"FOLD {fold}", "~"*8)
        
        if preload_emmbedings:
            print('loading train/test data...')
            df_train = pd.read_pickle(f'preload/train_data_fold_{fold}{preload_tag}.pkl')
            df_test = pd.read_pickle(f'preload/test_data_fold_{fold}{preload_tag}.pkl')

            df_train = df_train.sort_values('names')
            df_test = df_test.sort_values('names')

            print(len(df_train))
            print(len(df_test))

            train_loader, valid_loader = get_loaders(0, config, train, train_i, valid_i, ind_id_map)

            fold_thresholds.append(0.5)
            df_train_list.append(df_train)
            df_test_list.append(df_test)
        else:
            print('inferencing...')

            train_loader, valid_loader = get_loaders(0, config, train, train_i, valid_i, ind_id_map)
            test_loader = get_test_loader(config, test, ind_id_map)
        
            model, _, _ = get_model_optimizer_criterion(config)
            model.eval()
            model.load_state_dict(torch.load(f'models/fold_{fold}_best.pth'))

            train_embs, train_targets, train_names = \
                collect_loader_features(config, train_loader, model, col_img_names=True)
            
            val_embs, val_targets, val_names = \
                collect_loader_features(config, valid_loader, model, col_img_names=True)
            
            #best_th = find_best_threshold(config, train_embs, train_targets, target_encodings,
            #                                val_embs, val_targets, val_names)
            best_th = 0.5

            train_data = {
                'names': train_names + val_names,
                'embs': np.concatenate([train_embs, val_embs], axis=0).tolist(),
                'targets': np.concatenate([train_targets, val_targets], axis=0).tolist()
            }
            
            print(len(train_data['targets']))

            df_train = pd.DataFrame(train_data).sort_values('names')
            print(len(df_train))
            df_train.to_pickle(f'preload/train_data_fold_{fold}{preload_tag}.pkl')

            test_embs, _, test_names = \
                collect_loader_features(config, test_loader, model, col_img_names=True, col_targets=False)

            test_data = {
                'names': test_names,
                'embs': test_embs.tolist()
            }  
            df_test = pd.DataFrame(test_data).sort_values('names')
            df_test.to_pickle(f'preload/test_data_fold_{fold}{preload_tag}.pkl') 

            df_train_list.append(df_train)
            df_test_list.append(df_test)
            fold_thresholds.append(best_th)

    # check if columns are alignd
    aligned = []
    for i in range(1, len(df_train_list)):
        train_names1 = df_train_list[0]['names'].values
        train_names2 = df_train_list[1]['names'].values
        aligned.append(all(train_names1 == train_names2))

        test_names1 = df_test_list[0]['names'].values
        test_names2 = df_test_list[1]['names'].values
        aligned.append(all(test_names1 == test_names2))
    print(aligned)
    assert sum(aligned) == len(aligned)

    train_embeddings = []
    for df in df_train_list:
        train_embeddings.append(np.array(list(df['embs'].values)))
    train_embeddings = np.concatenate(train_embeddings, axis=1)
    train_targets = df_train_list[0]['targets'].values

    test_embeddings = []
    for df in df_test_list:
        test_embeddings.append(np.array(list(df['embs'].values)))
    test_embeddings = np.concatenate(test_embeddings, axis=1)
    test_names = df_test_list[0]['names'].values

    best_th = sum(fold_thresholds) / len(fold_thresholds)

    print('Fitting NearestNeighbors')
    neigh = NearestNeighbors(n_neighbors=config.KNN, metric='cosine')
    neigh.fit(train_embeddings)

    test_nn_distances, test_nn_idxs = neigh.kneighbors(
        test_embeddings,
        config.KNN,
        return_distance=True
    )

    with open('extra_data/test_predictions.pkl', 'wb') as f:
        pickle.dump([test_nn_distances, test_nn_idxs, test_names, train_targets], f)

    test_df = []
    for i in tqdm(range(len(test_names))):
        id_ = test_names[i]
        targets = train_targets[test_nn_idxs[i]]
        distances = test_nn_distances[i]
        subset_preds = pd.DataFrame(
            np.stack([targets,distances], axis=1),
            columns=['target', 'distances']
        )
        subset_preds['image'] = id_
        test_df.append(subset_preds)

    test_df = pd.concat(test_df).reset_index(drop=True)
    test_df['confidence'] = 1 - test_df['distances']
    test_df = test_df.groupby(['image','target']).confidence.max().reset_index()
    test_df = test_df.sort_values('confidence',ascending=False).reset_index(drop=True)
    test_df['target'] = test_df['target'].map(target_encodings)
    test_df.to_csv('test_neighbors.csv')
    test_df.image.value_counts().value_counts()

    # prepare final submission.csv file
    predictions = get_predictions(test_df, best_th)
                
    for x in tqdm(predictions):
        if len(predictions[x])<5:
            remaining = [y for y in sample_list if y not in predictions]
            predictions[x] = predictions[x]+remaining
            predictions[x] = predictions[x][:5]
        predictions[x] = ' '.join(predictions[x])
        
    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ['image','predictions']
    predictions.to_csv('submissions/fold_0.csv', index=False)



