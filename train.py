import argparse

from mbednet import make_config, run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or f"checkpoints/fold{args.fold}"
    kwargs = dict(checkpoint_dir=checkpoint_dir, fold=args.fold,
                  epochs=args.epochs, num_workers=args.num_workers)
    if args.data_path:
        kwargs["data_path"] = args.data_path
    cfg = make_config(**kwargs)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
