import argparse
from termcolor import colored


import numpy as np
import torch
from torch import optim
import random

from dataset import SimDataset
from model_main import IVAE, MLP

class Log:
    def __init__(self, filename):
        self.filename = filename
    def log(self, content):
        with open(self.filename, "a") as f:
            f.write(content + '\n')
            f.close()
        print(content)

def RMSE(pre, target):
    mse = np.mean(np.square(pre - target))
    return np.sqrt(mse)


parser = argparse.ArgumentParser()
parser.add_argument("-l", type=float, default=0.001, help="learning rate")
parser.add_argument("-decay", type=float, default=0.000)
parser.add_argument("--latdim", type=int, default=5, help="Latent Dimension")
parser.add_argument("--obsm", type=int, default=0, help="Observed Dimension")
parser.add_argument("--ycof", type=float, default=0.5, help="Y cof") 
parser.add_argument("--mask", type=int, default=0, help="Mask ObsX")
parser.add_argument("--ylayer", type=int, default=50, help="Y Layer Dimension")
parser.add_argument("--nlayer", type=int, default=50, help="N Layer Dimension")
parser.add_argument("--proxyn", type=float, default=0.5, help="Proxy Noise Std")

args = parser.parse_args()
lr = args.l
latent_dim = args.latdim
y_cof = args.ycof

n = 10000  
m = 5 
p = 20 
noise_dim = 10 
new_data = True
obs_idx = list(range(args.obsm))

name = "Obs_confounder5_n10_t20_cor02_logit08_Linear" 
data = SimDataset(n, m, p, obs_idx, noise_dim, new_data, name)


x, t, y, obs_x = data.getTrainData(args.proxyn)
t_test, y_test = data.getInTestData()  # Insample的Test data
x_out_test, t_out_test, y_out_test, obs_x_out = data.getOutTestData(args.proxyn)


file_name = name + "_obs" + str(args.obsm) + "_proxy"
filelog = Log("result_inout_{}.txt".format(file_name)) #!!!
filelog.log("Experiment Start!")
filelog.log(str(args))
filelog.log("Y Mean %f, Std %f " % (np.mean(y), np.std(y)))
filelog.log("Test Y Mean %f, Std %f " % (np.mean(y_test), np.std(y_test)))
filelog.log("Y Out Test Mean %f, Std %f" % (np.mean(y_out_test), np.std(y_out_test)))
filelog.log('Observe confounder %d, Noise %d dimension' % (args.obsm, noise_dim))

obsm = obs_x.shape[1]
obs_x = obs_x[:, :obsm-args.mask]
obs_x_out = obs_x_out[:, :obsm-args.mask]
obsm = obs_x.shape[1]
filelog.log("Learning Rate %f" % (lr))

hidden_size = args.nlayer
n_layers = 3
y_layers = 3
y_hidden = args.ylayer
epochs = 3000
batch_size = 1024
rep_times = 10
m1 = []
m1_out = []
m4 = []
mt = []
mp = []
mt2 = []

rec_conf_dwp = []
rec_noise_dwp = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


for rep in range(rep_times):
    manual_seed(rep)
    filelog.log(colored("========== repeat time {} ==========".format(rep + 1), "red"))
    i_vae = IVAE(
        obsm,
        latent_dim,
        p,
        hidden_size,
        n_layers,
        y_layers,
        y_hidden,
        learning_rate=lr,
        weight_decay=args.decay,
        y_cof=y_cof,
    )

    i_vae = i_vae.to(device)

    filelog.log(colored("== Training all ==".format(rep + 1), "blue"))

    last_loss = 100000
    last_epoch = -1
    for ep in range(epochs):
        idx = np.random.permutation(n)
        rec_loss_s = []
        KL_loss_s = []
        loss_s = []
        t_loss_s = []
        y_loss_s = []
        for j in range(0, n, batch_size):
            op, ed = j, min(j + batch_size, n)
            obsx_batch = torch.FloatTensor(obs_x[idx[op:ed]]).to(device)
            t_batch = torch.FloatTensor(t[idx[op:ed]]).to(device)
            y_batch = torch.FloatTensor(y[idx[op:ed]]).view(-1, 1).to(device)

            loss, rec_loss, KL_loss, t_loss, y_loss = i_vae.optimize(
                obsx_batch, t_batch, y_batch
            )
            loss_s.append(loss * (ed - op))
            rec_loss_s.append(rec_loss * (ed - op))
            KL_loss_s.append(KL_loss * (ed - op))
            t_loss_s.append(t_loss * (ed - op))
            y_loss_s.append(y_loss * (ed - op))
        if (ep + 1) % 50 == 0:
            filelog.log("Epoch %d " % (ep))
            filelog.log("Overall Loss: %f" % (sum(loss_s) / n))
            filelog.log("Rec Loss: %f" % (sum(rec_loss_s) / n))
            filelog.log("KL Loss: %f" % (sum(KL_loss_s) / n))
            filelog.log("Y Loss: %f" % (sum(y_loss_s) / n))
            filelog.log("T Loss: %f" % (sum(t_loss_s) / n))
        current_loss = sum(loss_s) / n

    filelog.log(colored("== Reconstructing confounder ==".format(rep + 1), "blue"))
    rec_net = MLP(latent_dim, x.shape[1], 20, 3).to(device)
    optimizer = optim.Adam(rec_net.parameters(), lr=0.005)
    euc = torch.nn.MSELoss(reduction="none")
    last_loss = 100000
    for ep in range(epochs):
        idx = np.random.permutation(n)
        rec_loss_s = []
        for j in range(0, n, batch_size):
            op, ed = j, min(j + batch_size, n)
            obsx_batch = torch.FloatTensor(obs_x[idx[op:ed]]).to(device)
            t_batch = torch.FloatTensor(t[idx[op:ed]]).to(device)
            y_batch = torch.FloatTensor(y[idx[op:ed]]).view(-1, 1).to(device)
            z_batch = i_vae.infer_post(obsx_batch, t_batch, y_batch, ifnoise=False)
            x_batch = torch.FloatTensor(x[idx[op:ed]]).to(device)
            pre_x = rec_net(z_batch)
            loss = euc(pre_x, x_batch).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rec_loss_s.append(loss.item() * (ed - op))
        if (ep + 1) % 50 == 0:
            filelog.log("Epoch %d" % (ep))
            filelog.log("Rec Loss: %f" % (sum(rec_loss_s) / n))
            if (ep + 1) % 50 == 0:
                if sum(rec_loss_s) / n >= last_loss * 1.00:
                    break
                else:
                    last_loss = sum(rec_loss_s) / n
    rec_conf_dwp.append(last_loss)

    filelog.log(colored("== Reconstructing noise ==".format(rep + 1), "blue"))
    rec_net = MLP(latent_dim, obs_x.shape[1] - args.obsm, 20, 3).to(device)
    optimizer = optim.Adam(rec_net.parameters(), lr=0.005)
    euc = torch.nn.MSELoss(reduction="none")
    last_loss = 100000
    for ep in range(epochs):
        idx = np.random.permutation(n)
        rec_loss_s = []
        for j in range(0, n, batch_size):
            op, ed = j, min(j + batch_size, n)
            obsx_batch = torch.FloatTensor(obs_x[idx[op:ed]]).to(device)
            t_batch = torch.FloatTensor(t[idx[op:ed]]).to(device)
            y_batch = torch.FloatTensor(y[idx[op:ed]]).view(-1, 1).to(device)
            z_batch = i_vae.infer_post(obsx_batch, t_batch, y_batch, ifnoise=False)

            pre_x = rec_net(z_batch)
            loss = euc(pre_x, obsx_batch[:, args.obsm :]).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rec_loss_s.append(loss.item() * (ed - op))
        if (ep + 1) % 50 == 0:
            filelog.log("Epoch %d" % (ep))
            filelog.log("Rec Loss: %f" % (sum(rec_loss_s) / n))
            if (ep + 1) % 50 == 0:
                if sum(rec_loss_s) / n >= last_loss * 1.00:
                    break
                else:
                    last_loss = sum(rec_loss_s) / n
    rec_noise_dwp.append(last_loss)

    filelog.log(colored("== Testing in sample performance ==".format(rep + 1), "blue"))
    Train_y = np.zeros(n)
    for i in range(0, n, batch_size):
        op, ed = i, min(i + batch_size, n)
        obsx_batch = torch.FloatTensor(obs_x[op:ed]).to(device)
        t_batch = torch.FloatTensor(t[op:ed]).to(device)
        y_batch = torch.FloatTensor(y[op:ed]).view(-1, 1).to(device)
        Train_y[op:ed] = i_vae.predict_post(
            obsx_batch, t_batch, y_batch, t_batch, ifexp=False
        )

    # In-sample
    Insample_y = np.zeros(n)
    for i in range(0, n, batch_size):
        op, ed = i, min(i + batch_size, n)
        obsx_batch = torch.FloatTensor(obs_x[op:ed]).to(device)
        t_batch = torch.FloatTensor(t[op:ed]).to(device)
        y_batch = torch.FloatTensor(y[op:ed]).view(-1, 1).to(device)
        t_new_batch = torch.FloatTensor(t_test[op:ed]).to(device)
        Insample_y[op:ed] = i_vae.predict_post(
            obsx_batch, t_batch, y_batch, t_new_batch, ifexp=False
        )
    # Out-sample
    Outsample_y = np.zeros(n)
    for i in range(0, n, batch_size):
        op, ed = i, min(i + batch_size, n)
        obsx_batch = torch.FloatTensor(obs_x_out[op:ed]).to(device)
        t_new_batch = torch.FloatTensor(t_out_test[op:ed]).to(device)
        Outsample_y[op:ed] = i_vae.predict_prior(obsx_batch, t_new_batch, ifexp=False)
    filelog.log("Train Error: %f" % (RMSE(Train_y, y)))
    filelog.log("Insample Error: %f" % (RMSE(Insample_y, y_test)))
    filelog.log("Outsample Error: %f" % (RMSE(Outsample_y, y_out_test)))
    m1.append(RMSE(Insample_y, y_test))
    m1_out.append(RMSE(Outsample_y, y_out_test))
    mt.append(RMSE(Train_y, y))

m1 = np.array(m1)
m1_out = np.array(m1_out)

mt = np.array(mt)
rec_conf_dwp = np.array(rec_conf_dwp)
rec_conf_dwp = rec_conf_dwp[~np.isnan(rec_conf_dwp)]
rec_noise_dwp = np.array(rec_noise_dwp)
rec_noise_dwp = rec_noise_dwp[~np.isnan(rec_noise_dwp)]


output = ""
output += "Train, RMSE mean %.4f std %.4f\n" % (mt.mean(), mt.std())
try:
    output += "Ours, In-sample RMSE mean %.4f std %.4f, reconstruct confounder %.4f (%.4f) noise %.4f (%.4f)\n" % (
        m1.mean(),
        m1.std(),
        np.mean(rec_conf_dwp),
        np.std(rec_conf_dwp),
        np.mean(rec_noise_dwp),
        np.std(rec_noise_dwp),
    )
    output += "Ours, Out-sample RMSE mean %.4f std %.4f\n" % (m1_out.mean(), m1_out.std())

except:
    pass

filelog.log(output)
