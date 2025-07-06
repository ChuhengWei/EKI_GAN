from torch.utils.data import DataLoader

from sgan.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    max_pedestrians = getattr(args, 'max_pedestrians', 5)

    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        max_pedestrians=max_pedestrians 
    )

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)

    return dset, loader