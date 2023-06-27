import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight.data, gain=0.3)
        nn.init.constant_(m.bias.data, 0)


def BinaryCE(pre, tar):
    return -(tar * torch.log(pre + 1e-4) + (1 - tar) * torch.log(1 - pre + 1e-4))


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, sig=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(
                nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim)
            )
        self.fc = nn.ModuleList(_fc_list)
        self.sig = sig
        self.apply(weights_init)

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = F.elu(self.fc[c](h))
        if self.sig:
            h = torch.sigmoid(h)
        return h


class IVAE(nn.Module):
    def __init__(
        self,
        obsx_dim,
        latent_dim,
        treat_dim,
        hidden_dim,
        n_layers,
        y_layers,
        y_hidden,
        learning_rate=0.001,
        weight_decay=0.001,
        y_cof=2.0,
    ):
        super().__init__()
        self.hid_prior_mean = MLP(obsx_dim, latent_dim, hidden_dim, n_layers)
        self.hid_prior_logv = MLP(obsx_dim, latent_dim, hidden_dim, n_layers)
        self.encoder_mean = MLP(
            obsx_dim + treat_dim + 1, latent_dim, hidden_dim, n_layers
        )
        self.encoder_logv = MLP(
            obsx_dim + treat_dim + 1, latent_dim, hidden_dim, n_layers
        )
        self.decoder_t = MLP(latent_dim, treat_dim, hidden_dim, n_layers, True)
        self.decoder_y = MLP(latent_dim + treat_dim, 1, y_hidden, y_layers)
        models = [
            self.hid_prior_mean,
            self.hid_prior_logv,
            self.encoder_mean,
            self.encoder_logv,
            self.decoder_t,
            self.decoder_y,
        ]
        self.bceloss = nn.BCELoss(reduction="none")
        self.mseloss = nn.MSELoss(reduction="none")
        parameters = []
        for model in models:
            parameters.extend(list(model.parameters()))
        self.optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        self.y_cof = y_cof
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def neg_elbo(self, obsx, t, y):
        prior_m = self.hid_prior_mean(obsx)
        prior_lv = self.hid_prior_logv(obsx)
        z_mean = self.encoder_mean(torch.cat((obsx, t, y), dim=1))
        z_logv = self.encoder_logv(torch.cat((obsx, t, y), dim=1))

        std_z = torch.randn(size=z_mean.size()).to(self.device)
        sample_z = std_z * torch.exp(z_logv) + z_mean

        rec_t = self.decoder_t(sample_z)
        rec_y = self.decoder_y(torch.cat((sample_z, t), dim=1))
        t_loss = self.bceloss(rec_t, t).sum(1)
        y_loss = self.mseloss(rec_y, y).sum(1)
        KL_divergence = 0.5 * (
            (prior_lv - z_logv) * 2
            - 1
            + torch.exp(2 * (z_logv - prior_lv))
            + (z_mean - prior_m) * (z_mean - prior_m) * torch.exp(-2 * prior_lv)
        ).sum(1)
        rec_loss = t_loss + y_loss * self.y_cof

        return (
            (KL_divergence + rec_loss).mean(),
            KL_divergence.mean(),
            rec_loss.mean(),
            t_loss.mean(),
            y_loss.mean(),
        )

    def optimize(self, obsx, t, y):
        self.optimizer.zero_grad()
        loss, kl, rec, t_loss, y_loss = self.neg_elbo(obsx, t, y)
        loss.backward()
        self.optimizer.step()
        return loss.item(), rec.item(), kl.item(), t_loss.item(), y_loss.item()

    def infer_post(self, obsx, t, y, ifnoise):
        if not ifnoise:
            ret = self.encoder_mean(torch.cat((obsx, t, y), dim=1))
        else:
            ret = self.encoder_mean(torch.cat((obsx, t, y), dim=1))
            ret += torch.exp(
                self.encoder_logv(torch.cat((obsx, t, y), dim=1))
            ) * torch.randn(size=ret.size()).to(self.device)
        return ret

    def predict_post(self, obsx, t, y, tnew, ifexp=True):
        if ifexp:
            z = self.infer_post(obsx, t, y, False)
            pre_y = self.decoder_y(torch.cat((z, tnew), dim=1))
            pre_y = pre_y.detach().cpu().numpy().squeeze()
        else:
            pre_y = np.zeros(obsx.shape[0])
            for i in range(500):
                z = self.infer_post(obsx, t, y, True)
                tmp = self.decoder_y(torch.cat((z, tnew), dim=1))
                tmp = tmp.detach().cpu().numpy().squeeze()
                pre_y = pre_y + tmp
            pre_y /= 500
        return pre_y

    def infer_prior(self, obsx, ifnoise):
        if not ifnoise:
            ret = self.hid_prior_mean(obsx)
        else:
            ret = self.hid_prior_mean(obsx)
            ret += torch.exp(self.hid_prior_logv(obsx)) * torch.randn(size=ret.size()).to(self.device)
        return ret

    def predict_prior(self, obsx, tnew, ifexp=True):
        if ifexp:
            z = self.infer_prior(obsx, False)
            pre_y = self.decoder_y(torch.cat((z, tnew), dim=1))
            pre_y = pre_y.detach().cpu().numpy().squeeze()
        else:
            pre_y = np.zeros(obsx.shape[0])
            for i in range(500):
                z = self.infer_prior(obsx, True)
                tmp = self.decoder_y(torch.cat((z, tnew), dim=1))
                tmp = tmp.detach().cpu().numpy().squeeze()
                pre_y = pre_y + tmp
            pre_y /= 500
        return pre_y
