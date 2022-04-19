import sys
import argparse
import json
import pickle

sys.path.append('/home/nikita/nikita/happawhale')

from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.inference_utils import find_best_threshold, get_predictions
from src.dataset import build_mlp_dataloaders, build_mlp_test_dataloader
from src.model import MLP
from configs.mlp_config import Config


def collect_loader_features(config, loader, model, 
                    col_embs=True, col_targets=True, col_img_names=False):
    to_np = lambda x: x.detach().cpu().numpy()
    emmbedings = []
    targets = []
    img_names = []
    for batch_i, batch in enumerate(tqdm(loader)):
        for source in config.dataset['sources']:
            batch[source] = batch[source].to(config.device)
        batch['targets'] = batch['targets'].to(config.device)

        with torch.no_grad():
            emb = model(batch)
        if col_embs:
            emmbedings.append(to_np(emb))
        if col_targets:
            cur_targets = batch['targets']
            targets.append(to_np(cur_targets))
        if col_img_names:
            img_names.extend(batch['image_codes'])
        #if batch_i == 15:
        #    break
    emmbs = np.concatenate(emmbedings, axis=0) if col_embs else None
    targets = np.concatenate(targets, axis=0) if col_targets else None
    return emmbs, targets, img_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fold_to_test', type=int)
    parser.add_argument('--preload', action='store_true')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    config = Config()
    config.device = torch.device(config.training_procedure['device_name'])

    args = parse_args()

    ind_id_map = json.load(open(config.dataset['name_id_map']))
    target_encodings = {j[1]: j[0] for i, j in ind_id_map.items()}

    fold_to_test = args.fold_to_test
    if args.preload:
        print('loading train/test data...')

        with open(f'preload/train_data_fold_{fold_to_test}_mlp.pkl', 'rb') as f:
            train_embeddings, train_targets, train_names, best_th = pickle.load(f)

        with open(f'preload/test_data_fold_{fold_to_test}_mlp.pkl', 'rb') as f:
            test_names, test_embeddings, best_th = pickle.load(f)

        print('Threshold: ', best_th)
    else:
        print('inferencing...')
        train_loader, val_loader = build_mlp_dataloaders(config, fold_to_test)
        test_loader = build_mlp_test_dataloader(config, fold_to_test)

        model = MLP(config).cuda().eval()
        model.load_state_dict(torch.load(f'models/fold_{fold_to_test}_mlp_best.pth'))

        train_embeddings, train_targets, train_names = \
            collect_loader_features(config, train_loader, model, col_img_names=True)
        
        val_embeddings, val_targets, val_names = \
            collect_loader_features(config, val_loader, model, col_img_names=True)
                
        test_embeddings, _, test_names = \
            collect_loader_features(config, test_loader, model, col_img_names=True, col_targets=False)

        best_th = find_best_threshold(config, train_embeddings, train_targets, target_encodings,
                                     val_embeddings, val_targets, val_names)

        train_embeddings = np.concatenate([train_embeddings, val_embeddings], axis=0)
        train_targets = np.concatenate([train_targets, val_targets], axis=0)
        train_names = train_names + val_names
        with open(f'preload/train_data_fold_{fold_to_test}_mlp.pkl', 'wb') as f:
            pickle.dump([train_embeddings, train_targets, train_names, best_th], f)

        with open(f'preload/test_data_fold_{fold_to_test}_mlp.pkl', 'wb') as f:
            pickle.dump([test_names, test_embeddings, best_th], f)

    print('Fitting NearestNeighbors')
    neigh = NearestNeighbors(n_neighbors=config.KNN, metric='cosine')
    neigh.fit(train_embeddings)

    print('Calculating test distances')
    test_nn_distances, test_nn_idxs = neigh.kneighbors(
        test_embeddings,
        config.KNN,
        return_distance=True
    )
    
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
    predictions.to_csv(f'submissions/fold_mlp_{fold_to_test}.csv', index=False)
    print(predictions.head())

    predictions = predictions['predictions']
    counter = 0
    for i in predictions:
        if i.startswith('new_i'):
            counter += 1
    print('Number of new_inds: ', counter / len(predictions))
    
    