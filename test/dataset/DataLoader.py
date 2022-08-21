from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lib.dataset.DataLoader import DataLoader
from lib.random.wrapper import seed2key

if __name__ == '__main__':
    key = seed2key(42)

    data_loader = DataLoader(key=key, n_workers=24, max_files=2)
    for n_batches, (src, dst, mask_enc, mask_dec, mask_dec_enc, labels) in enumerate(data_loader):
        pass
    data_loader.close()
