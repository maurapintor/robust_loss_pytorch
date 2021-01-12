import foolbox as fb
import numpy as np
import torch.utils.data
from models import ConvNet
from utils import H5Dataset
import matplotlib.pyplot as plt

classes = (0, 1, 2)
ds = H5Dataset(path='data/kws10_16x32.hdf5', classes=classes)
num_samples = len(ds)
train_size = 0.8
idxs = np.arange(num_samples)
np.random.shuffle(idxs)
train_size = int(num_samples*train_size)
test_idx = idxs[train_size:]

ts_sampler = torch.utils.data.SubsetRandomSampler(indices=test_idx)
test_loader = torch.utils.data.DataLoader(ds, batch_size=20, sampler=ts_sampler)


plt.rcParams.update({
    "text.usetex": True})

num_eps = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epsilons = np.linspace(0.0, 3, num=num_eps)
num_batches = 10
model = ConvNet(out_dim=len(classes))
model = model.to(device)
lambda_consts = [0, 4]
for lambda_const in lambda_consts:
    print("Security evaluation for lambda = ", lambda_const)
    lambda_str = str(lambda_const).replace('.', '-')
    model.load_state_dict(torch.load(f'pretrained/lambda_{lambda_str}.pt'))
    model.eval()
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=None)
    fmodel = fmodel.transform_bounds((0, 1))

    attack = fb.attacks.L2PGD(steps=50)
    noise = fb.attacks.L2AdditiveUniformNoiseAttack()
    robust_accuracy = torch.zeros(num_eps)
    noise_accuracy = torch.zeros(num_eps)

    for batch_idx, (data, target) in enumerate(test_loader):
        criterion = fb.criteria.Misclassification(target)
        raw, clipped, is_adv = attack(fmodel, data, criterion, epsilons=epsilons)
        raw_noise, clipped_noise, is_adv_noise = noise(fmodel, data, criterion, epsilons=epsilons)
        robust_accuracy += (1 - is_adv.float()).mean(axis=-1)/num_batches
        noise_accuracy += (1 - is_adv_noise.float()).mean(axis=-1)/num_batches
        if batch_idx == num_batches - 1:
            break
    c = ['tab:red', 'tab:cyan', 'tab:purple', 'tab:green', 'tab:orange', 'tab:blue']
    md = 'robust model' if lambda_const > 0 else 'standard model'
    c = 'tab:blue' if lambda_const > 0 else 'tab:red'
    plt.plot(epsilons, (robust_accuracy).numpy(), label=md, c=c)
    plt.plot(epsilons, (noise_accuracy).numpy(), c=c, linestyle='-.')

plt.legend()
# plt.xlabel(r'Perturbation Strength ($L_{\infty}$)')
plt.xlabel(r'Perturbation Strength ($L_{2}$)')
plt.ylabel('Model Accuracy')
plt.savefig('sec_eval_kws.pdf', format='pdf')
