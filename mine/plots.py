import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import MultivariateNormalDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = 1 if device=='cuda' else 0
print(device)

from models.mine import MutualInformationEstimator
from pytorch_lightning import Trainer
import logging
logging.getLogger().setLevel(logging.ERROR)

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('dim', type=int)
args = parser.parse_args()

dim = args.dim
N = 3000
lr = 1e-4
epochs = 200
batch_size = 500

steps = 15
rhos = np.array([-0.99,-0.9,-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7,0.9,0.99])
loss_type = ['mine_biased']

results_dict = dict()

for loss in loss_type:
    results = []
    for rho in rhos:
        train_loader = torch.utils.data.DataLoader(
        MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
        MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)

        
        true_mi = train_loader.dataset.true_mi

        kwargs = {
            'lr': lr,
            'batch_size': batch_size,
            'train_loader': train_loader,
            'test_loader': test_loader,
            'alpha': 1.0
        }

        model = MutualInformationEstimator(
            dim, dim, loss=loss, **kwargs).to(device)
        
        trainer = Trainer(max_epochs=epochs, early_stop_callback=False, gpus=num_gpus)
        trainer.fit(model)
        trainer.test()

        print("True_mi {}".format(true_mi))
        print("MINE {}".format(model.avg_test_mi))
        results.append((rho, model.avg_test_mi, true_mi))

    results = np.array(results)
    results_dict[loss] = results


fig, axs = plt.subplots(1, len(loss_type), sharex = True, figsize = (6,4))
plots = []
for ix, loss in enumerate(loss_type):
    results = results_dict[loss]
    plots += axs.plot(results[:,0], results[:,1], label='MINE')
    plots += axs.plot(results[:,0], results[:,2], linestyle='--', label='True MI')
    axs.set_xlabel('correlation')
    axs.set_ylabel('mi')
    #axs.title.set_text(f"{loss} for {dim} dimensional inputs")
    
fig.legend(plots[0:2], labels = ['MINE', 'True MI'], loc='upper right')
fig.savefig('figures/mi_estimation.png')
