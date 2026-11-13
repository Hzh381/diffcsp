import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)

        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5

    def forward(self, batch):

        # ===== 1. 采样扩散时间步 =====
        # batch 是 PyG 的 Batch 对象: B 个晶体拼成的"大图"。
        # 约定形状: B = 晶体数(=batch_size), N = 总原子数, E = 总边数。
        batch_size = batch.num_graphs                                                      # 标量 B
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)             # (B,) 每个晶体随机抽一个时间步 t
        time_emb = self.time_embedding(times)                                             # (B, time_dim) 时间步正弦嵌入

        # ===== 2. 取出时间步对应的噪声调度系数 =====
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]                        # (B,) 晶格 DDPM 的 α̅_t
        beta = self.beta_scheduler.betas[times]                                           # (B,) 晶格 β_t (训练阶段未真正用到)

        c0 = torch.sqrt(alphas_cumprod)                                                   # (B,) DDPM 前向: 信号系数 √α̅_t
        c1 = torch.sqrt(1. - alphas_cumprod)                                              # (B,) DDPM 前向: 噪声系数 √(1-α̅_t)

        sigmas = self.sigma_scheduler.sigmas[times]                                       # (B,) 坐标 SDE 的噪声尺度 σ_t
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]                             # (B,) 归一化 σ (用于尺度对齐)

        # ===== 3. 准备晶格矩阵与分数坐标 =====
        # batch.lengths / batch.angles: (B, 3) → 每行[a,b,c] / [α,β,γ]
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)            # (B, 3, 3) 晶格矩阵(行向量为晶格基矢)
        frac_coords = batch.frac_coords                                                   # (N, 3) 全部原子的分数坐标(已拼接)

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)        # (B,3,3) / (N,3) 标准高斯噪声 ε_l, ε_x

        # ===== 4. 对晶格和分数坐标分别加噪 =====
        # (c0/c1 用 [:,None,None] 升维到 (B,1,1) 以便 broadcast 到 (B,3,3))
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l         # (B,3,3) 晶格走 DDPM 线性插值加噪
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]              # (N,1) 把图级 σ_t 按原子数广播到每个原子
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]    # (N,1) 同上(归一化 σ)
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.                 # (N,3) 坐标加噪; %1. 做周期边界 wrap


        # ===== 5. 条件/消融开关: 对应模态 cost 为 0 时不参与扩散, 保持原始值 =====
        if self.keep_coords:
            input_frac_coords = frac_coords                                               # 坐标不扩散
        if self.keep_lattice:
            input_lattice = lattices                                                      # 晶格不扩散

        # ===== 6. 网络前向: 预测噪声/得分 =====
        # decoder = CSPNet(等变GNN)。输入:
        #   atom_types (N,), input_frac_coords (N,3), input_lattice (B,3,3),
        #   num_atoms (B,), batch.batch (N,) 节点→图映射
        # 输出: pred_l (B,3,3) 晶格噪声估计, pred_x (N,3) 坐标得分估计
        # 原 CSPNet 调用（保留备查）
        # pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        # 新：self.decoder 已替换为 EquiformerCSPDecoder（接口一致，参数不变）
        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        # ===== 7. 计算目标与损失 =====
        # 坐标目标 tar_x: 环形空间 wrapped normal 分布的得分(对数密度梯度) ∇log p(x), 再除 √σ_norm 归一化
        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)   # (N,3)

        loss_lattice = F.mse_loss(pred_l, rand_l)                                         # 晶格: 直接回归噪声 ε_l (DDPM)
        loss_coord = F.mse_loss(pred_x, tar_x)                                            # 坐标: 得分匹配(score matching)

        loss = (                                                                          # 加权总损失(标量)
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict, loss

