import os
import time
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from bioreason.models.dna_only import DNAClassifierModel
from bioreason.dataset.utils import truncate_dna
from bioreason.dataset.kegg import dna_collate_fn
from bioreason.dataset.variant_effect import clean_variant_effect_example
from bioreason.models.evo2_tokenizer import Evo2Tokenizer, register_evo2_tokenizer
from bioreason.models.hyenaDNA_tokenizer import register_hyena_tokenizer
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

register_evo2_tokenizer()
register_hyena_tokenizer()

class DNAClassifierModelTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training the DNA classifier.
    """

    def __init__(self, args):
        """
        Initialize the DNAClassifierModelTrainer.

        Args:
            args: Command line arguments
        """
        super().__init__()
        self.save_hyperparameters(args)

        # Load dataset and labels
        self.dataset, self.labels = self.load_dataset()
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        # Load model
        self.dna_model = DNAClassifierModel(
            dna_model_name=self.hparams.dna_model_name,
            cache_dir=self.hparams.cache_dir,
            max_length_dna=self.hparams.max_length_dna,
            num_classes=len(self.labels),
            dna_is_evo2=self.hparams.dna_is_evo2,
            dna_embedding_layer=self.hparams.dna_embedding_layer,
            train_just_classifier=self.hparams.train_just_classifier,
        )
        self.dna_tokenizer = self.dna_model.dna_tokenizer

        # Set the training mode for the classifier and pooler
        self.dna_model.pooler.train()
        self.dna_model.classifier.train()

        # Freeze the DNA model parameters
        if self.hparams.dna_is_evo2:
            self.dna_model_params = self.dna_model.dna_model.model.parameters()
        else:
            self.dna_model_params = self.dna_model.dna_model.parameters()

        if self.hparams.train_just_classifier:
            for param in self.dna_model_params:
                param.requires_grad = False
                
        # For calculating the metrics
        self.val_preds = []
        self.val_labels = []
        self.train_preds = []
        self.train_labels = []
        self.test_preds = []
        self.test_labels = []

    def _step(self, prefix, batch_idx, batch):
        """
        Performs a single training/validation step.

        Args:
            batch: Dictionary containing the batch data
            prefix: String indicating the step type ('train' or 'val')

        Returns:
            torch.Tensor: The computed loss for this batch
        """
        ref_ids = batch["ref_ids"].to(self.device)
        alt_ids = batch["alt_ids"].to(self.device)
        ref_attention_mask = batch["ref_attention_mask"].to(self.device)
        alt_attention_mask = batch["alt_attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        logits = self.dna_model(ref_ids=ref_ids, alt_ids=alt_ids, ref_attention_mask=ref_attention_mask, alt_attention_mask=alt_attention_mask)

        # Calculate loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        preds_np = preds.cpu()
        labels_np = labels.cpu()
        
        if prefix == "train":
            self.train_preds.append(preds_np)
            self.train_labels.append(labels_np)
        elif prefix == "val":
            self.val_preds.append(preds_np)
            self.val_labels.append(labels_np)
        elif prefix == "test":
            self.test_preds.append(preds_np)
            self.test_labels.append(labels_np)

        # Logging metrics
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{prefix}_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if (prefix == "test") or (prefix == "train" and (self.global_step % 1000 == 0)) or (prefix == "val" and (batch_idx % 100 == 0)):
            pred_label = self.labels[preds[0]]
            true_label = self.labels[labels[0]]
            print(f"Example {prefix} {batch_idx} {self.global_step}: Prediction: {pred_label}, Target: {true_label}")

        return loss

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        return self._step(prefix="train", batch_idx=batch_idx, batch=batch)

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        return self._step(prefix="val", batch_idx=batch_idx, batch=batch)
    
    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        return self._step(prefix="test", batch_idx=batch_idx, batch=batch)
    
    def on_validation_epoch_end(self):
        """Actions to perform at the end of each validation epoch."""
        preds = torch.cat(self.val_preds, dim=0).numpy()
        labels = torch.cat(self.val_labels, dim=0).numpy()
        self.val_preds.clear()
        self.val_labels.clear()
        acc = accuracy_score(labels, preds)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        if self.hparams.dataset_type == "kegg":
            # For KEGG dataset, we use macro metrics
            precision = precision_score(labels, preds, average='weighted', zero_division=0)
            recall = recall_score(labels, preds, average='weighted', zero_division=0)
            f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        elif self.hparams.dataset_type in ["variant_effect_coding", "variant_effect_non_snv","clinvar"]:
            precision = precision_score(labels, preds, average='binary', zero_division=0)
            recall = recall_score(labels, preds, average='binary', zero_division=0)
            f1 = f1_score(labels, preds, average='binary', zero_division=0)
        
            # For other datasets, we use weighted metrics
        self.log("val_acc_epoch", acc, prog_bar=True, logger=True)
        self.log("val_precision_epoch", precision,  prog_bar=True, logger=True)
        self.log("val_recall_epoch", recall,prog_bar=True, logger=True)
        self.log("val_f1_epoch", f1, prog_bar=True, logger=True)
        self.log("val_precision_macro_epoch", precision_macro,  prog_bar=True, logger=True)
        self.log("val_recall_macro_epoch", recall_macro,prog_bar=True, logger=True)
        self.log("val_f1_macro_epoch", f1_macro, prog_bar=True, logger=True)
        # Log the metrics to console
        print(f"{self.current_epoch} Validation - Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        print(f"{self.current_epoch} Validation - Macro Precision: {precision_macro}, Macro Recall: {recall_macro}, Macro F1: {f1_macro}")
    
    def on_train_epoch_end(self):
        """Actions to perform at the end of each training epoch."""
        preds = torch.cat(self.train_preds, dim=0).numpy()
        labels = torch.cat(self.train_labels, dim=0).numpy()
        self.train_preds.clear()
        self.train_labels.clear()
        acc = accuracy_score(labels, preds)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        if self.hparams.dataset_type == "kegg":
            # For KEGG dataset, we use macro metrics
            precision = precision_score(labels, preds, average='weighted', zero_division=0)
            recall = recall_score(labels, preds, average='weighted', zero_division=0)
            f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        elif self.hparams.dataset_type in ["variant_effect_coding", "variant_effect_non_snv","clinvar"]:
            precision = precision_score(labels, preds, average='binary', zero_division=0)
            recall = recall_score(labels, preds, average='binary', zero_division=0)
            f1 = f1_score(labels, preds, average='binary', zero_division=0)
        
        self.log("train_acc_epoch", acc, prog_bar=True, logger=True)
        self.log("train_precision_epoch", precision,  prog_bar=True, logger=True)
        self.log("train_recall_epoch", recall,prog_bar=True, logger=True)
        self.log("train_f1_epoch", f1, prog_bar=True, logger=True)
        self.log("train_precision_macro_epoch", precision_macro,  prog_bar=True, logger=True)
        self.log("train_recall_macro_epoch", recall_macro,prog_bar=True, logger=True)
        self.log("train_f1_macro_epoch", f1_macro, prog_bar=True, logger=True)
        # Log the metrics to console
        print(f"{self.current_epoch} Training - Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        print(f"{self.current_epoch} Training - Macro Precision: {precision_macro}, Macro Recall: {recall_macro}, Macro F1: {f1_macro}")
        
    def on_test_epoch_end(self):
        """Actions to perform at the end of each test epoch."""
        preds = torch.cat(self.test_preds, dim=0).numpy()
        labels = torch.cat(self.test_labels, dim=0).numpy()
        self.test_preds.clear()
        self.test_labels.clear()
        acc = accuracy_score(labels, preds)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        if self.hparams.dataset_type == "kegg":
            # For KEGG dataset, we use macro metrics
            precision = precision_score(labels, preds, average='weighted', zero_division=0)
            recall = recall_score(labels, preds, average='weighted', zero_division=0)
            f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        elif self.hparams.dataset_type in ["variant_effect_coding", "variant_effect_non_snv","clinvar"]:
            precision = precision_score(labels, preds, average='binary', zero_division=0)
            recall = recall_score(labels, preds, average='binary', zero_division=0)
            f1 = f1_score(labels, preds, average='binary', zero_division=0)
        self.log("test_acc_epoch", acc, prog_bar=True, logger=True)
        self.log("test_precision_epoch", precision,  prog_bar=True, logger=True)
        self.log("test_recall_epoch", recall,prog_bar=True, logger=True)
        self.log("test_f1_epoch", f1, prog_bar=True, logger=True)
        self.log("test_precision_macro_epoch", precision_macro,  prog_bar=True, logger=True)
        self.log("test_recall_macro_epoch", recall_macro,prog_bar=True, logger=True)
        self.log("test_f1_macro_epoch", f1_macro, prog_bar=True, logger=True)
        # Log the metrics to console
        print(f"{self.current_epoch} Test - Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        print(f"{self.current_epoch} Test - Macro Precision: {precision_macro}, Macro Recall: {recall_macro}, Macro F1: {f1_macro}")
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Only include parameters that require gradients
        classifier_params = [
            {
                "params": self.dna_model.classifier.parameters(),
                "lr": self.hparams.learning_rate,
            },
            {
                "params": self.dna_model.pooler.parameters(),
                "lr": self.hparams.learning_rate,
            }
        ]
        dna_model_params = [
            {
                "params": self.dna_model_params,
                "lr": self.hparams.learning_rate * 0.1,
            },
        ]

        if self.hparams.train_just_classifier:
            # Only train classifier parameters
            optimizer = AdamW(
                classifier_params,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            # Train both DNA model and classifier with different learning rates
            optimizer = AdamW(
                classifier_params + dna_model_params,
                weight_decay=self.hparams.weight_decay,
            )

        # Get total steps from trainer's estimated stepping batches
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        # Create scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def load_dataset(self):
        """Load the dataset based on the dataset type."""
        if self.hparams.dataset_type == "kegg":
            dataset_type = self.hparams.dataset_type
            dataset = load_dataset('json', data_files={
                "train": f"data/{dataset_type}/train.json",
                "val": f"data/{dataset_type}/val.json",
                "test": f"data/{dataset_type}/test.json"
            })
            
            if self.hparams.truncate_dna_per_side:
                dataset = dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

            labels = []
            for split, data in dataset.items():
                labels.extend(data["answer"])
            labels = list(set(labels))
        
        elif self.hparams.dataset_type == "variant_effect_coding" or self.hparams.dataset_type == "clinvar":
            # dataset = load_dataset("wanglab/bioR_tasks", "variant_effect_coding")
            if self.hparams.dataset_type == "variant_effect_coding":
                dataset = load_dataset("json", data_files={
                    "train": "data/vep/ve_coding_train.json",
                    "test": "data/vep/ve_coding_test.json"})
            else:
                dataset = load_dataset("json", data_files={
                    "train": "data/clinvar/train.json",
                    "test": "data/clinvar/test.json"})
            dataset = dataset.map(clean_variant_effect_example)

            if self.hparams.truncate_dna_per_side:
                dataset = dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

            labels = []
            for split, data in dataset.items():
                labels.extend(data["answer"])
            labels = sorted(list(set(labels)))
        
        elif self.hparams.dataset_type == "variant_effect_non_snv":
            # dataset = load_dataset("wanglab/bioR_tasks", "task5_variant_effect_non_snv")
            dataset = load_dataset("json", data_files={
                "train": "data/vep/ve_non_snv_train.json",
                "test": "data/vep/ve_non_snv_test.json"})
            
            dataset = dataset.rename_column("mutated_sequence", "variant_sequence")
            dataset = dataset.map(clean_variant_effect_example)

            if self.hparams.truncate_dna_per_side:
                dataset = dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

            labels = []
            for split, data in dataset.items():
                labels.extend(data["answer"])
            labels = sorted(list(set(labels)))

        else:
            raise ValueError(f"Invalid dataset type: {self.hparams.dataset_type}")

        print(f"Dataset:\n{dataset}\nLabels:\n{labels}\nNumber of labels:{len(labels)}")
        return dataset, labels

    def train_dataloader(self):
        """Create and return the training DataLoader."""
        if self.hparams.dataset_type == "kegg":
            train_dataset = self.dataset["train"]
            collate_fn = lambda b: dna_collate_fn(b, dna_tokenizer=self.dna_tokenizer, label2id=self.label2id, max_length=self.hparams.max_length_dna)
        
        elif self.hparams.dataset_type == "variant_effect_coding":
            train_dataset = self.dataset["train"]
            collate_fn = lambda b: dna_collate_fn(b, dna_tokenizer=self.dna_tokenizer, label2id=self.label2id, max_length=self.hparams.max_length_dna)
        
        elif self.hparams.dataset_type == "variant_effect_non_snv":
            train_dataset = self.dataset["train"]
            collate_fn = lambda b: dna_collate_fn(b, dna_tokenizer=self.dna_tokenizer, label2id=self.label2id, max_length=self.hparams.max_length_dna)
        elif self.hparams.dataset_type == "clinvar":
            train_dataset = self.dataset["train"]
            collate_fn = lambda b: dna_collate_fn(b, dna_tokenizer=self.dna_tokenizer, label2id=self.label2id, max_length=self.hparams.max_length_dna)
        else:
            raise ValueError(f"Invalid dataset type: {self.hparams.dataset_type}")

        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Create and return the training DataLoader."""
        if self.hparams.dataset_type == "kegg":

            if self.hparams.merge_val_test_set:
                val_dataset = concatenate_datasets([self.dataset['test'], self.dataset['val']])
            else:
                val_dataset = self.dataset["val"]

            collate_fn = lambda b: dna_collate_fn(b, dna_tokenizer=self.dna_tokenizer, label2id=self.label2id, max_length=self.hparams.max_length_dna)
        
        elif self.hparams.dataset_type == "variant_effect_coding":
            val_dataset = self.dataset["test"]
            collate_fn = lambda b: dna_collate_fn(b, dna_tokenizer=self.dna_tokenizer, label2id=self.label2id, max_length=self.hparams.max_length_dna)
        
        elif self.hparams.dataset_type == "variant_effect_non_snv":
            val_dataset = self.dataset["test"]
            collate_fn = lambda b: dna_collate_fn(b, dna_tokenizer=self.dna_tokenizer, label2id=self.label2id, max_length=self.hparams.max_length_dna)
        
        elif self.hparams.dataset_type == "clinvar":
            val_dataset = self.dataset["test"]
            collate_fn = lambda b: dna_collate_fn(b, dna_tokenizer=self.dna_tokenizer, label2id=self.label2id, max_length=self.hparams.max_length_dna)
        else:
            raise ValueError(f"Invalid dataset type: {self.hparams.dataset_type}")

        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        """Create and return the test DataLoader."""
        return self.val_dataloader()


def main(args):
    """Main function to run the training process."""
    # Set random seed and environment variables
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Initialize model
    model = DNAClassifierModelTrainer(args)

    # Setup directories
    run_name = f"{args.project_name}-{args.dataset_type}-{args.dna_model_name.split('/')[-1]}"
    args.checkpoint_dir = f"{args.checkpoint_dir}/{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    args.output_dir = f"{args.output_dir}/{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{run_name}-" + "{epoch:02d}-{val_loss_epoch:.4f}",
            save_top_k=2,
            monitor="val_acc_epoch",
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.project_name,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=(
            "ddp"
            if args.strategy == "ddp"
            else DeepSpeedStrategy(stage=2, offload_optimizer=False, allgather_bucket_size=5e8, reduce_bucket_size=5e8)
        ),
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        val_check_interval=1 / 3,
    )

    # Train model
    trainer.fit(model, ckpt_path=args.ckpt_path)
    trainer.test(model, ckpt_path=args.ckpt_path if args.ckpt_path else "best")

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    torch.save(model.dna_model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description="Train DNA Classifier")

    # Model parameters
    parser.add_argument(
        "--dna_model_name",
        type=str,
        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_length_dna", type=int, default=1024)
    parser.add_argument("--dna_is_evo2", type=bool, default=False)
    parser.add_argument("--dna_embedding_layer", type=str, default=None)

    # Training parameters
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--train_just_classifier", type=bool, default=True)
    parser.add_argument("--dataset_type", type=str, choices=["kegg", "variant_effect_coding", "variant_effect_non_snv","clinvar"], default="kegg")
    parser.add_argument("--kegg_data_dir_huggingface", type=str, default="wanglab/kegg")
    parser.add_argument("--truncate_dna_per_side", type=int, default=0)

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="dna_classifier_output")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints"
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="tb_logs")
    parser.add_argument("--project_name", type=str, default="dna-only-nt-500m")
    parser.add_argument("--merge_val_test_set", type=bool, default=True)

    # Other parameters
    parser.add_argument("--seed", type=int, default=23)

    args = parser.parse_args()
    main(args)
