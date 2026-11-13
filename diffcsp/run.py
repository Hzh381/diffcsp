"""
DiffCSP 训练主入口。

该文件是整个项目的"装配层"：负责用 Hydra 读取配置、把数据（datamodule）
和模型（model）实例化出来，再交给 PyTorch Lightning 的 Trainer 去训练/测试。

注意：模块之间的串联不是靠 import，而是靠 conf/*.yaml 里的 _target_ / defaults
配置，配合 hydra.utils.instantiate 动态拼接完成的。
"""

from pathlib import Path
from typing import List
import sys
sys.path.append('.')  # 把项目根目录加入模块搜索路径，便于 import diffcsp
import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from diffcsp.common.utils import log_hyperparameters, PROJECT_ROOT

def build_callbacks(cfg: DictConfig) -> List[Callback]:
    """根据配置组装训练回调函数（学习率监控、早停、模型保存）。"""
    callbacks: List[Callback] = []

    # 学习率监控：记录每一步/每一轮的学习率变化
    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    # 早停：验证指标连续多轮不下降则提前终止训练
    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    # 模型保存：保存验证指标最优的若干个 checkpoint
    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),  # 保存到本次 Hydra 运行目录
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    通用训练主流程。

    :param cfg: 由 Hydra 从 /conf 目录解析得到的运行配置
    """
    # 若开启确定性训练，固定随机种子以保证可复现
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    # 调试模式：强制用 CPU、单进程、wandb 离线，方便在断点里调试
    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0
        cfg.logging.wandb.mode = "offline"

    # 本次 Hydra 运行目录（存放 checkpoint、hparams、scaler 等输出）
    hydra_dir = Path(HydraConfig.get().run.dir)

    # 1. 实例化数据模块（读 CSV → 建 dataset/Dataloader，并计算归一化 scaler）
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # 2. 实例化模型；这里的具体类是 conf/model/*.yaml 里 _target_ 决定的
    #    （默认是 diffusion.CSPDiffusion）
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # 3. 把数据模块算好的 scaler 传给模型（用于反归一化预测值），并落盘保存
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    if datamodule.scaler is not None:
        model.lattice_scaler = datamodule.lattice_scaler.copy()
        model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')

    # 4. 组装训练回调函数
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # 5. 配置日志记录器。这里改用本地 CSVLogger，避免 Windows 上 WandB
    #    的 fork 启动方式以及联网登录要求。
    train_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <CSVLogger>")
        train_logger = CSVLogger(save_dir=str(hydra_dir), name="csv_logs")

    # 6. 把最终生效的完整配置存成 hparams.yaml，方便复现与加载
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # 7. 若运行目录里已有 checkpoint，则选 epoch 最大的那个用于续训
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None

    # 8. 实例化 Trainer（训练引擎），可能有旧版 API，对应 pytorch_lightning 1.x
    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=train_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )

    # 记录超参数到 logger
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    # 9. 开始训练，训练完自动做一次测试
    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    """Hydra 装饰器加载 conf/default.yaml 及其 defaults 引用的子配置后进入 run。"""
    run(cfg)


if __name__ == "__main__":
    main()
