import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import model as M
import nnue_dataset      # C++版ローダーに必要
import nnue_bin_dataset  # Python版 (予備)
import features

def data_loader_cc(train_filename, val_filename, feature_set, num_workers, batch_size, filtered, random_fen_skipping, main_device, epoch_size):
    # Epoch and validation sizes are arbitrary
    val_size = 1000000
    features_name = feature_set.name
    
    # C++実装のデータセットを呼び出し
    train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filename, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device)
    val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, device=main_device)
    
    # Lightningで使用するために固定のバッチ数でラップする
    train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
    val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
    return train, val

def data_loader_py(train_filename, val_filename, feature_set, batch_size, num_workers):
    train = DataLoader(
        nnue_bin_dataset.NNUEBinData(train_filename, feature_set),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val = DataLoader(
        nnue_bin_dataset.NNUEBinData(val_filename, feature_set),
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train, val

def main():
    # Tensorコアを有効化 (RTX 5060 Ti用)
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="NNUE Trainer (DDP safe)")

    # positional
    parser.add_argument("train", help="Training data (.bin)")
    parser.add_argument("val", help="Validation data (.bin)")

    # Lightning Trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # training options
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-size", type=int, default=100000000) # エポックごとのサンプル数

    # NNUE options
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--label-smoothing-eps", type=float, default=0.0)
    parser.add_argument("--num-batches-warmup", type=int, default=10000)
    parser.add_argument("--newbob-decay", type=float, default=0.5)
    parser.add_argument("--num-epochs-to-adjust-lr", type=int, default=50)
    parser.add_argument("--score-scaling", type=float, default=361)
    parser.add_argument("--min-newbob-scale", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.0)

    features.add_argparse_args(parser)

    args = parser.parse_args()

    # ---- seed ----
    pl.seed_everything(args.seed, workers=True)

    # ---- feature set ----
    feature_set = features.get_feature_set_from_name(args.features)

    # ---- model ----
    model = M.NNUE(
        feature_set=feature_set,
        lambda_=[args.lambda_],
        lr=[args.lr],
        label_smoothing_eps=args.label_smoothing_eps,
        num_batches_warmup=args.num_batches_warmup,
        newbob_decay=args.newbob_decay,
        num_epochs_to_adjust_lr=args.num_epochs_to_adjust_lr,
        score_scaling=args.score_scaling,
        min_newbob_scale=args.min_newbob_scale,
        momentum=args.momentum,
    )

    # ---- data (C++ Loader) ----
    # 以前の data_loader_py から変更
    train_loader, val_loader = data_loader_cc(
        train_filename=args.train,
        val_filename=args.val,
        feature_set=feature_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        filtered=True,              # 適宜変更してください
        random_fen_skipping=False,  # 適宜変更してください
        main_device="cuda",
        epoch_size=args.epoch_size
    )

    # ---- logger & checkpoint ----
    logger = pl_loggers.TensorBoardLogger(
        save_dir=args.default_root_dir or "logs",
        name="nnue",
    )

    checkpoint_cb = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=args.network_save_period if hasattr(args, "network_save_period") else 10,
        filename="{epoch}",
    )

    # ---- trainer ----
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[checkpoint_cb],
    )

    # ---- train ----
    trainer.fit(model, train_loader, val_loader)

    # ---- final checkpoint ----
    trainer.save_checkpoint(os.path.join(logger.log_dir, "final.ckpt"))

if __name__ == "__main__":
    main()
