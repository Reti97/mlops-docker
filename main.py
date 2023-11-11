import argparse
from datetime import datetime
import torch
from pytorch_lightning import Trainer, seed_everything
from glue_module import GLUEDataModule, GLUETransformer

def main(args):
    # Set random seed for reproducibility
    seed_everything(42)

    # Initialize LightningDataModule
    data_module = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size
    )

    # Initialize LightningModule
    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=data_module.num_labels,
        task_name=args.task_name,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size
        #eval_splits=data_module.eval_splits
    )

    # Set up Trainer
    trainer = Trainer(
        default_root_dir=args.checkpoint_dir,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=args.max_epochs,
        #progress_bar_refresh_rate=1,
    )

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLUE Transformer model.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Pretrained model name or path.")
    parser.add_argument("--task_name", type=str, default="mrpc", help="GLUE task name.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum number of training epochs.")
    parser.add_argument("--checkpoint_dir", type=str, default="models", help="Directory to save checkpoints.")

    args = parser.parse_args()
    main(args)
