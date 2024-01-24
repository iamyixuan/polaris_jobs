import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
mpi4py.rc.recv_mprobe = False


from modulus.models.fno import FNO
from deephyper.problem import HpProblem
from deephyper.evaluator import RunningJob, profile
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
import logging
import os
from model import *
from sklearn.preprocessing import QuantileTransformer

# Avoid some errors on some MPI implementations


log_dir = "/home/iamyixuan/work/ImPACTs/HPO/scale_log"

from mpi4py import MPI




problem = HpProblem()
activations = [
    "relu",
    "leaky_relu",
    "prelu",
    "relu6",
    "elu",
    "selu",
    "silu",
    "gelu",
    "sigmoid",
    "logsigmoid",
    "softplus",
    "softshrink",
    "softsign",
    "tanh",
    "tanhshrink",
    "threshold",
    "hardtanh",
    "identity",
    "squareplus",
]
optimizers = ["Adadelta", "Adagrad", "Adam", "AdamW", "RMSprop", "SGD"]
schedulers = ["cosine", "step"]

problem.add_hyperparameter((1, 16), "padding", default_value=8)
problem.add_hyperparameter(
    ["constant", "reflect", "replicate", "circular"],
    "padding_type",
    default_value="constant",
)
problem.add_hyperparameter([True, False], "coord_feat", default_value=True)
problem.add_hyperparameter(activations, "lift_act", default_value="gelu")
problem.add_hyperparameter((2, 32), "num_FNO", default_value=4)
problem.add_hyperparameter((2, 32), "num_modes", default_value=16)
problem.add_hyperparameter((2, 64), "latent_ch", default_value=32)
problem.add_hyperparameter((1, 16), "num_projs", default_value=1)
problem.add_hyperparameter((2, 32), "proj_size", default_value=32)
problem.add_hyperparameter(activations, "proj_act", default_value="silu")

problem.add_hyperparameter(optimizers, "optimizer", default_value="Adam")
problem.add_hyperparameter((1e-6, 1e-2), "lr", default_value=1e-3)
problem.add_hyperparameter((0.0, 0.1), "weight_decay", default_value=0)
# problem.add_hyperparameter(schedulers, "scheduler", default_value='cosine')
problem.add_hyperparameter((8, 256), "batch_size", default_value=128)
#problem.add_hyperparameter((20, 100), 'epochs', default_value=20)



def rollout_mse_avg(true, pred):
    '''
    True and pred are from a single rollout - single sequence
    '''
    avg_mse = []
    for i in range(len(true)):
        MSE = []
        for t in range(29):
            MSE.append(np.mean((true[i][t].detach().numpy() - pred[i][t].detach().numpy())**2))
        avg_mse.append(MSE)
    
    avg_mse = np.array(avg_mse).mean(axis=0)
        
    return avg_mse


@profile
def run(job: RunningJob):
    config = job.parameters.copy()

    net = FNO(
        in_channels=6,
        out_channels=5,
        decoder_layers=config["num_projs"],
        decoder_layer_size=config["proj_size"],
        decoder_activation_fn=config["proj_act"],
        dimension=2,
        latent_channels=config["latent_ch"],
        num_fno_layers=config["num_FNO"],
        num_fno_modes=config["num_modes"],
        padding=config["padding"],
        padding_type=config["padding_type"],
        activation_fn=config["lift_act"],
        coord_features=config["coord_feat"],
    )

    trainset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "train", transform=True,
    )
    valset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "val", transform=True,
    )
    testSet = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "test", transform=True
    )

    trainer = Trainer(model=net, lossFn=torch.nn.MSELoss())

    trainLoader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    valLoader = DataLoader(valset, batch_size=valset.__len__())
    testLoader = DataLoader(testSet, batch_size=testSet.__len__())

    rollout_val_data = DataLoader(valset, batch_size=1)
    rollout_test_data = DataLoader(testSet, batch_size=1)

    log = trainer.train(
        trainLoader=trainLoader,
        valLoader=valLoader,
        epochs=20,
        optimizer=config["optimizer"],
        learningRate=config["lr"],
        # scheduler=config['scheduler'],
        weight_decay=config["weight_decay"],
    )

    testLoss, inferenceTime = trainer.test(testLoader)

    #get rollout objectives
    rollout_val = trainer.rollout(rollout_val_data)
    rollout_test = trainer.rollout(rollout_test_data)
    
    val_rollout_mse = rollout_mse_avg(rollout_val['true'], rollout_val['pred'])[14]
    test_rollout_mse = rollout_mse_avg(rollout_test['true'], rollout_test['pred'])[14]



    trainLoss = log.logger["TrainLoss"]
    valLoss = log.logger["ValLoss"]
    NumTrainableParams = log.logger["NumTrainableParams"]

    objective_0 = (
        "F" if np.isnan(valLoss).any() or np.isinf(valLoss).any() or ('Flag' in log.keys()) else -valLoss[-1]
    )

    objective_1 = -inferenceTime  # lower the better so taking the negative to maximize

    objective_2 = -val_rollout_mse # improve rollout performance when training for one step forward

    objective = [objective_0, objective_1, objective_2]

    return {
        "objective": objective,
        "metadata": {"TrainLoss": trainLoss, "valLoss": valLoss, "testLoss": testLoss, 'testRollout' :test_rollout_mse},
    }


def trainWithBest(config):
    net = FNO(
        in_channels=6,
        out_channels=5,
        decoder_layers=config["p:num_projs"],
        decoder_layer_size=config["p:proj_size"],
        decoder_activation_fn=config["p:proj_act"],
        dimension=2,
        latent_channels=config["p:latent_ch"],
        num_fno_layers=config["p:num_FNO"],
        num_fno_modes=config["p:num_modes"],
        padding=config["p:padding"],
        padding_type=config["p:padding_type"],
        activation_fn=config["p:lift_act"],
        coord_features=config["p:coord_feat"],
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

    trainLoader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    valLoader = DataLoader(valset, batch_size=valset.__len__())
    testLoader = DataLoader(testSet, batch_size=testSet.__len__())

    log = trainer.train(
        trainLoader=trainLoader,
        valLoader=valLoader,
        epochs=20,
        optimizer=config["optimizer"],
        learningRate=config["lr"],
        # scheduler=config['scheduler'],
        weight_decay=config["weight_decay"],
    )

    pred, true = trainer.predict(testLoader)

    return {'true': true, 'pred': pred}


def getBestConfig(df):
    # load the result.csv datafraem
    # remove 'F'
    df = df[df['objective_0']!='F']
    # take out the pareto front
    df = df[df['pareto_efficient']==True]


    # convert to float and take the negative
    qt = QuantileTransformer()

    df['objective_0'] = qt.fit_transform(-df['objective_0'].astype(float).values.reshape(-1, 1))
    df['objective_1'] = qt.fit_transform(-df['objective_1'].astype(float).values.reshape(-1, 1))

    # Pick the best objectives 
    # quantile transformation
    
    
    min_row_index = (df['objective_0'] + df['objective_1']).idxmin()
    df = df.rename(columns=lambda x: x.replace('p:', ''))
    return df.loc[min_row_index]



    # get hyperparmeter configs
    


if __name__ == "__main__":
    # if not MPI.Is_initialized():
    #     MPI.Init_thread()
    #     rank = MPI.COMM_WORLD.Get_rank()
    #     print(f"{rank=}hahahahah")


    if not MPI.Is_initialized():
        MPI.Init_thread()

        if MPI.COMM_WORLD.Get_rank() == 0:
            # Only the root rank will create the directory
            print("creating logging................")
            os.makedirs(log_dir, exist_ok=True)

        MPI.COMM_WORLD.barrier()  # Synchronize all processes
        logging.basicConfig(
            filename=os.path.join(log_dir, f"deephyper.{MPI.COMM_WORLD.Get_rank()}.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )


    with Evaluator.create(
        run,
        method="mpicomm",
    ) as evaluator:
        if evaluator is not None:
            search = CBO(
                problem,
                evaluator, 
                moo_scalarization_strategy="Chebyshev",
                moo_scalarization_weight="random",
                objective_scaler="quantile-uniform",
                acq_func="UCB", 
                multi_point_strategy="qUCB", 
                n_jobs=8,
                verbose=1,
                initial_points=[problem.default_configuration]
            )
            results = search.search(max_evals=2000)
            results.to_csv("results-100nodes-stopper.csv")
