'''
This script is to train a model based on specified configuration and 
'''

import numpy as np
from hpo import *
import pandas as pd
import pickle


def run(config):
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

    trainer = Trainer(model=net, lossFn=torch.nn.MSELoss())

    trainLoader = DataLoader(trainset, batch_size=int(config['batch_size']), shuffle=True)
    valLoader = DataLoader(valset, batch_size=valset.__len__())
    testLoader = DataLoader(testSet, batch_size=testSet.__len__())

    rolloutLoader = DataLoader(valset, batch_size=1) 

    log = trainer.train(
        trainLoader=trainLoader,
        valLoader=valLoader,
        epochs=20,
        optimizer=config["optimizer"],
        learningRate=config["lr"],
        weight_decay=config["weight_decay"],
    )
    pred, true = trainer.predict(testLoader)
    np.savez('defaultPredictions-nmlzd.npz', true=true, pred=pred)
    trainLoss = log.logger["TrainLoss"]
    valLoss = log.logger["ValLoss"]
    with open('defaultLog-nmlzd.pkl', 'wb') as f:
        pickle.dump({'train': trainLoss, 'val': valLoss}, f)

    rolloutPred = trainer.rollout(rolloutLoader)
    with open('defaultRollout.pkl', 'wb') as f:
        pickle.dump(rolloutPred, f)
    return

if __name__ == '__main__':
    config = problem.default_configuration
    # df = pd.read_csv('/home/iamyixuan/work/ImPACTs/HPO/results-100nodes-2000evals.csv')
    # config = dict(getBestConfig(df))
    print(config)
    run(config)