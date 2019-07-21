import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class MLP(nn.Module):
    def __init__(self, in_d, mid_d, out_d):
        super(MLP, self).__init__()
        self.a_fc = nn.Linear(in_d, mid_d)
        self.b_fc = nn.Linear(mid_d, out_d)

    def forward(self, x):
        return self.b_fc(F.softplus(self.a_fc(x)))


class Extractor(nn.Module):
    def __init__(self, net_arch):
        super(Extractor, self).__init__()
        self.net_arch = net_arch

        # encoder for head words
        self.en1_fc_1 = nn.Linear(net_arch.num_input, net_arch.l1_units)  # V -> 100
        self.en1_fc_2 = nn.Linear(net_arch.l1_units, net_arch.l2_units)  # 100 -> 100
        self.en1_drop = nn.Dropout(0.2)
        # encoder for extracted features
        self.en2_fc_1 = nn.Linear(net_arch.feature_dim + 1, net_arch.l1_units)  # D + 1 -> 100
        self.en2_fc_2 = nn.Linear(net_arch.l1_units, net_arch.l2_units)  # 100 -> 100
        self.en2_drop = nn.Dropout(0.2)
        # encoder for the variational parameters of potential event types $t$
        self.mean_fc = nn.Linear(2 * net_arch.l2_units, net_arch.feature_dim)  # 200 -> D
        self.mean_bn = nn.BatchNorm1d(net_arch.feature_dim)  # bn for mean
        self.logvar_fc = nn.Linear(2 * net_arch.l2_units, net_arch.feature_dim)  # 200 -> D
        self.logvar_bn = nn.BatchNorm1d(net_arch.feature_dim)  # bn for logvar
        self.t_drop = nn.Dropout(0.2)

        # mlp for t
        # self.s_fc = MLP(net_arch.feature_dim, net_arch.feature_dim, net_arch.num_topic)  # D -> K
        self.s_fc = nn.Linear(net_arch.feature_dim, net_arch.num_topic)  # D -> K
        # decoder for head words
        self.decoder_h = nn.Linear(net_arch.num_topic, net_arch.num_input)  # K -> V
        self.decoder_h_bn = nn.BatchNorm1d(net_arch.num_input)  # bn for decoder_h
        # decoder for extracted features
        self.decoder_f_mean = nn.Linear(net_arch.num_topic, net_arch.feature_dim + 1)  # K -> D + 1
        self.decoder_f_mean_bn = nn.BatchNorm1d(net_arch.feature_dim + 1)  # bn for decoder_f_mean
        self.decoder_f_logvar = nn.Linear(net_arch.num_topic, net_arch.feature_dim + 1)  # K -> D + 1
        self.decoder_f_logvar_bn = nn.BatchNorm1d(net_arch.feature_dim + 1)  # bn for decoder_f_logvar
        if net_arch.init_mult != 0:
            self.decoder_h.weight.data.uniform_(0, net_arch.init_mult)
            self.decoder_f_mean.weight.data.uniform_(0, net_arch.init_mult)
            self.decoder_f_logvar.weight.data.uniform_(0, net_arch.init_mult)

        # prior mean and variance
        self.prior_mean = torch.Tensor(1, net_arch.feature_dim).fill_(0)
        self.prior_var = torch.Tensor(1, net_arch.feature_dim).fill_(net_arch.variance)
        if not net_arch.nogpu:
            self.prior_mean = self.prior_mean.cuda()
            self.prior_var = self.prior_var.cuda()
        self.prior_logvar = self.prior_var.log()

    def get_unnormalized_phi(self):
        return self.decoder_h.weight.data.cpu().numpy().T

    def get_beta_mean(self):
        return self.decoder_f_mean.weight.data.cpu().numpy().T

    def get_beta_logvar(self):
        return self.decoder_f_logvar.weight.data.cpu().numpy().T

    def forward(self, hcounts, feas, mask, compute_loss=False, avg_loss=True):
        '''
        :param hcounts: [batch_size, V]
        :param feas: [batch_size, seq_len, D + 1]
        :param mask: [batch_size, seq_len]
        '''
        # compute posterior
        en1 = F.softplus(self.en1_fc_1(hcounts))  # encoder for word counts
        en1 = F.softplus(self.en1_fc_2(en1))
        en1 = self.en1_drop(en1)  # [batch_size, 100] for head words
        pooled_fs = (feas * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)  # [batch_size, D + 1]
        en2 = F.softplus(self.en2_fc_1(pooled_fs))
        en2 = F.softplus(self.en2_fc_2(en2))
        en2 = self.en2_drop(en2)  # [batch_size, 100] for extracted features
        en = torch.cat([en1, en2], dim=1)  # [batch_size, 200] for data
        posterior_mean = self.mean_bn(self.mean_fc(en))  # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en))  # posterior log variance
        posterior_var = posterior_logvar.exp()  # posterior variance
        # take sample
        eps = hcounts.data.new().resize_as_(posterior_mean.data).normal_()  # noise
        event_type = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
        event_type = self.t_drop(event_type)
        # do reconstruction
        p = F.softmax(self.s_fc(event_type), dim=-1)  # [batch_size, K] mixture probability
        assert torch.isnan(p).sum().item() == 0
        recon_hcounts = F.softmax(self.decoder_h_bn(self.decoder_h(p)), dim=-1)  # reconstructed dist over vocabulary
        assert torch.isnan(recon_hcounts).sum().item() == 0
        recon_fs_mean = self.decoder_f_mean_bn(self.decoder_f_mean(p))  # reconstructed means of features
        assert torch.isnan(recon_fs_mean).sum().item() == 0
        recon_fs_logvar = self.decoder_f_logvar_bn(self.decoder_f_logvar(p))  # reconstructed logvariances of features
        assert torch.isnan(recon_fs_logvar).sum().item() == 0

        if compute_loss:
            loss = self.loss(hcounts, feas, mask, recon_hcounts, recon_fs_mean, recon_fs_logvar,
                             posterior_mean, posterior_logvar, posterior_var, avg_loss)
            return event_type, posterior_mean, posterior_var, loss
        else:
            return event_type, posterior_mean, posterior_var

    def loss(self, hcounts, feas, mask, recon_hcounts, recon_fs_mean, recon_fs_logvar,
             posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL1 = -(hcounts * (recon_hcounts + 1e-10).log()).sum(1)  # cross entropy loss
        dist = MultivariateNormal(loc=recon_fs_mean, covariance_matrix=torch.diag_embed(recon_fs_logvar.exp().sqrt()))
        NL2 = (-dist.log_prob(feas.transpose(0, 1)).transpose(0, 1) * mask).sum(1)
        # put NL together
        NL = NL1 + NL2
        # KLD
        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.net_arch.feature_dim)
        # loss
        loss = NL + KLD
        # in training mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss

    def save_cpu_model(self, path):
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()
        torch.save(state_dict, path)
        print("Saving model in %s." % path)

    def load_cpu_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("Loading model from %s." % path)
