import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from modulus.models.fno import FNO
from torch.utils.data import DataLoader
from data import SOMAdata
import time


def get_optimizer(name):
    if name == "Adadelta":
        optimizer = torch.optim.Adadelta
    elif name == "Adagrad":
        optimizer = torch.optim.Adagrad
    elif name == "Adam":
        optimizer = torch.optim.Adam
    elif name == "AdamW":
        optimizer = torch.optim.AdamW
    elif name == "RMSprop":
        optimizer = torch.optim.RMSprop
    elif name == "SGD":
        optimizer = torch.optim.SGD
    else:
        raise ValueError(f"Optimizer {name} does not exist.")
    return optimizer


def get_scheduler(name, optimizer):
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
    elif name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
    else:
        raise ValueError(f"Scheduler {name} does not exist.")
    return scheduler


class Logger:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger = {}

    def log(self, key, info):
        if key in self.logger.keys():
            self.logger[key].append(info)
        else:
            self.logger[key] = [info]

    def save(self, name):
        with open(self.path + f"log_{name}.pkl", "wb") as f:
            pickle.dump(self.logger, f)


class Trainer:
    def __init__(self, model, lossFn):
        self.model = model
        self.lossFn = lossFn

    def train(
        self,
        trainLoader,
        valLoader,
        epochs,
        optimizer,
        learningRate,
        # scheduler,
        weight_decay,
    ):
        optimizer = get_optimizer(optimizer)
        optimizer = optimizer(
            self.model.parameters(), lr=learningRate, weight_decay=weight_decay
        )
        # scheduler = get_scheduler(scheduler, optimizer)

        logger = Logger("./log/")
        bestVal = np.inf
        self.model.train()
        logger.log(
            "NumTrainableParams",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        for ep in tqdm(range(epochs)):
            runningLoss = []
            for xTrain, yTrain in trainLoader:
                optimizer.zero_grad()
                pred = self.model(xTrain)
                loss = self.lossFn(yTrain, pred)
                loss.backward()
                optimizer.step()
                runningLoss.append(loss.item())
            # scheduler.step()
            valLoss = self.val(valLoader)
            # if valLoss < bestVal:
            #     torch.save(self.model.state_dict(), './savedModel/bestStateDict')
            #     bestVal = valLoss
            #     logger.log('BestModelEp', ep)

            # logger.save('bsize_128_ep_100')

            #print(f"Epoch {ep}, Train loss {np.mean(runningLoss)}, Val loss {valLoss}")


            logger.log("Epoch", ep)
            logger.log("TrainLoss", np.mean(runningLoss))
            logger.log("ValLoss", valLoss.item())
            if ep >= 10 and valLoss.item() >= 10595.082: #terminate training if worse than the constant predictor after 10 epochs
                logger.log('Flag', 'F')
                break
    
        return logger

    def val(self, valLoader):
        self.model.eval()
        for xVal, yVal in valLoader:
            predVal = self.model(xVal)
            valLoss = self.lossFn(yVal, predVal)
        return valLoss

    def test(self, testLoader):
        self.model.eval()
        start_time = time.time()
        for xTest, yTest in testLoader:
            predTest = self.model(xTest)
            testLoss = self.lossFn(yTest, predTest)
        inferenceTime = time.time() - start_time
        return testLoss.item(), inferenceTime
    
    def predict(self, testLoader):
        self.model.eval()
        start_time = time.time()
        for xTest, yTest in testLoader:
            predTest = self.model(xTest)
        return predTest.detach().numpy(), yTest.numpy()
    
    def rollout(self, dataloader):
        self.model.eval()
        # for every 29 steps we save a sequence
        truePred = {'true': [] , 'pred': []}
        temp = []
        true = []
        for i, (x, y) in enumerate(dataloader):
            if i % 29 == 0 and i > 0:
                truePred['true'].append(true)
                truePred['pred'].append(temp)
                temp = [self.model(x)] # reset sequence
                true = [y]
            else:
                if len(temp) != 0:
                    oneStepPred = self.model(torch.cat([temp[-1], x[:,-1:, ...]], dim=1))
                else:
                    temp = [self.model(x)]
                    true = [y]
                    print(temp[0].shape, x.shape)
                    oneStepPred = self.model(torch.cat([temp[-1], x[:,-1:, ...]], dim=1))

                temp.append(oneStepPred)
                true.append(y)
        return truePred
                


        


if __name__ == "__main__":
    net = FNO(
        in_channels=6,
        out_channels=5,
        dimension=2,
    )
    trainset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "train"
    )
    valset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "val"
    )
    trainer = Trainer(
        model=net,
        lossFn=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        learningRate=1e-4,
    )
    trainLoader = DataLoader(trainset, batch_size=128, shuffle=True)
    valLoader = DataLoader(valset, batch_size=valset.__len__())

    trainer.train(trainLoader, valLoader, 100)
