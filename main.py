from datetime import datetime
import torch
from pytorch_lightning import Trainer, seed_everything
from glue_module import GLUEDataModule, GLUETransformer
import argparse
def main(args):
    seed_everything(42)

    data_module = GLUEDataModule(
        model_name_or_path='distilbert-base-uncased',
        task_name="mrpc",
        max_seq_length=128,
        train_batch_size=32,
        eval_batch_size=32
    )

    model = GLUETransformer(
        model_name_or_path='distilbert-base-uncased',
        num_labels=2,
        task_name="mrpc",
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        train_batch_size=32,
        eval_batch_size=32
    )

    trainer = Trainer(
        default_root_dir="models",
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=3,
    )
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLUE Transformer model.")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)

    args = parser.parse_args()
    main(args)