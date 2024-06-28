import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
# from src.models import BasicConvClassifier
from src.utils import set_seed

###新しくモデルをインポート
# from src.models import AudioInspiredClassifier
# from src.models import Wav2Vec2Classifier
# from src.models import Wav2Vec2ConvClassifier
from transformers import AutoConfig, Wav2Vec2Model
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
# from src.models import Wav2Vec2MEGClassifier
# from src.models import Wav2Vec2MEGClassifierWithAttention
# from src.models import SpatialAttention
# from src.models import MEGClassifier
# from src.models import ImprovedMEGClassifier
# from src.models import FurtherImprovedMEGClassifier
from src.models import PatchEmbedding
from src.models import MultiHeadAttention
from src.models import ResidualAdd
from src.models import FeedForwardBlock
from src.models import TransformerEncoderBlock
from src.models import TransformerEncoder
from src.models import ClassificationHead
from src.models import EEGConformer



@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # deviceの定義を追加 EEDConformer用に
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")
    ##ここまで
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # #前処理済みのデータを保存する関数を自分で作った
    # # データの保存関数
    # def save_preprocessed_data(dataset, split):
    #     save_path = os.path.join(args.data_dir, f"{split}_X_preprocessed.pt")
    #     torch.save(dataset.X, save_path)
    #     print(f"Preprocessed data for {split} split saved to {save_path}")

    # # 各データセットの前処理済みデータを保存
    # save_preprocessed_data(train_set, "train")
    # save_preprocessed_data(val_set, "val")
    # save_preprocessed_data(test_set, "test")

    # ------------------
    #       Model
    # ------------------

    # ## Basicモデルを使うパターン
    # model = BasicConvClassifier(
    #     train_set.num_classes, train_set.seq_len, train_set.num_channels
    # ).to(args.device)

    ##
    model = EEGConformer(
        emb_size=40,
        depth=6,
        n_classes=train_set.num_classes
    ).to(device)
    
    # ## カスタムモデル2 MEGClassifier
    # model = ImprovedMEGClassifier(
    #     num_classes=train_set.num_classes,
    #     seq_len=train_set.seq_len,
    #     in_channels=train_set.num_channels,
    #     hid_dim=args.hid_dim,
    #     num_heads=8,  # Transformer用のヘッド数
    #     num_layers=2  # Transformerレイヤーの数
    # ).to(args.device)

    ## カスタムモデル3
    # model = FurtherImprovedMEGClassifier(
    #     num_classes=train_set.num_classes,
    #     seq_len=train_set.seq_len,
    #     in_channels=train_set.num_channels,
    #     hid_dim=args.hid_dim,
    #     num_heads=args.transformer_heads,
    #     num_layers=args.transformer_layers,
    #     dropout=args.dropout
    # ).to(args.device)

    # ## Wav2Vec2
    # model = Wav2Vec2ConvClassifier(
    #     num_classes=train_set.num_classes,
    #     seq_len=train_set.seq_len,
    #     in_channels=train_set.num_channels,
    #     hid_dim=args.hid_dim
    # ).to(args.device)

    # # Wav2Vec2+3D畳み込みバージョン
    # model = Wav2Vec2MEGClassifier(
    #     num_classes=train_set.num_classes,
    #     num_channels=train_set.num_channels
    # ).to(args.device)

    # # Wav2Vec2+空間的注意機構バージョン
    # model = Wav2Vec2MEGClassifierWithAttention(
    #     num_classes=train_set.num_classes,
    #     num_channels=train_set.num_channels
    # ).to(args.device)

    

    # ------------------
    #     Optimizer
    # ------------------

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # オプティマイザの変更
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay  # L2正則化の強さ
    )


    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            # ##テンソル構造を知りたいので追加した
            # print(f"Batch shape from dataloader (train): X: {X.shape}, y: {y.shape}")
            # ##ここまで

            X, y = X.to(args.device), y.to(args.device)
            # ##テンソル構造を知りたいので追加した
            # print(f"Batch shape on device (train): X: {X.shape}, y: {y.shape}")
            # ##ここまで

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc),  "learning_rate": optimizer.param_groups[0]['lr']})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
