'''
This script is to train a model based on specified configuration and 
'''

import numpy as np
from hpo import *
import pandas as pd
import pickle
from metrics import ACCLoss
from datetime import datetime


def run(config):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    net = FNO(
        in_channels=6,
        out_channels=5,
        decoder_layers=config["num_projs"],
        decoder_layer_size=config["proj_size"],
        decoder_activation_fn=config["proj_act"],
        dimension=2,
        latent_channels=config["latent_ch"],
        num_fno_layers=config["num_FNO"],
        num_fno_modes=int(config["num_modes"]),
        padding=config["padding"],
        padding_type=config["padding_type"],
        activation_fn=config["lift_act"],
        coord_features=config["coord_feat"],
    )

    trainset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "train", transform=True
    )
    valset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "val", transform=True
    )
    testSet = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "test", transform=True
    )


    # TrainLossFn = MSE_ACCLoss(alpha=config['alpha'])
    TrainLossFn = torch.nn.MSELoss()
    ValLossFn = torch.nn.MSELoss()

    trainer = Trainer(model=net, TrainLossFn=TrainLossFn, ValLossFn=ValLossFn)

    trainLoader = DataLoader(trainset, batch_size=int(config['batch_size']), shuffle=True)
    valLoader = DataLoader(valset, batch_size=valset.__len__())
    testLoader = DataLoader(testSet, batch_size=testSet.__len__())

    rolloutLoader = DataLoader(testSet, batch_size=1) 

    log = trainer.train(
        trainLoader=trainLoader,
        valLoader=valLoader,
        epochs=100,
        optimizer=config["optimizer"],
        learningRate=config["lr"],
        weight_decay=config["weight_decay"],
    )
    time_now =  datetime.now().strftime("%Y%m%d_%H-%M")


    config_name = 'bestMSEOnlytrainMSEval'
    with open(f'{time_now}_{config_name}_log.pkl', 'wb') as f:
        pickle.dump(log.logger, f)

    pred, true = trainer.predict(testLoader)
    np.savez(f'{time_now}_{config_name}Pred.npz', true=true, pred=pred)
    trainLoss = log.logger["TrainLoss"]
    valLoss = log.logger["ValLoss"]
    with open(f'{time_now}_{config_name}Log.pkl', 'wb') as f:
        pickle.dump({'train': trainLoss, 'val': valLoss}, f)

    rolloutPred = trainer.rollout(rolloutLoader)
    with open(f'{time_now}_{config_name}Rollout.pkl', 'wb') as f:
        pickle.dump(rolloutPred, f)
    return


def getBestConfig(df):
    # load the result.csv datafraem
    # remove 'F'
    df = df[df['objective']!='F']
    #df = df[df['objective_2']!='F']
    # take out the pareto front
    #df = df[df['pareto_efficient']==True]


    # convert to float and take the negative
    qt = QuantileTransformer()
    df['objective'] = -df['objective'].astype(float)

    # df['objective'] = qt.fit_transform(-df['objective'].astype(float).values.reshape(-1, 1))
    # df['objective_0'] = qt.fit_transform(-df['objective_0'].astype(float).values.reshape(-1, 1))
    # df['objective_1'] = qt.fit_transform(-df['objective_1'].astype(float).values.reshape(-1, 1))
    # df['objective_2'] = qt.fit_transform(-df['objective_2'].astype(float).values.reshape(-1, 1))

    # Pick the best objectives 
    # quantile transformation
    
    
    # min_row_index = (df['objective_0'] + df['objective_1']).idxmin()
    min_row_index = (df['objective']).idxmin()

    # min_row_index = (df['objective_0']).idxmin()
    df = df.rename(columns=lambda x: x.replace('p:', ''))
    return df.loc[min_row_index]


if __name__ == '__main__':
    config = problem.default_configuration
    # df = pd.read_csv('/home/iamyixuan/work/ImPACTs/HPO/2024022911_DoubleStop100Nodes-MSE_only/results.csv')
    # config = dict(getBestConfig(df))
    print(config)
    run(config)
