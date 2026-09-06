"""Dump JLD2-saved TCI.TensorTrain objects to .npz (Julia axis order restored)."""
import sys, os, re, glob
import h5py, numpy as np

def load_tt(path):
    with h5py.File(path, "r") as f:
        refs = f[f["single_stored_object"][()]["sitetensors"]][()]
        out = []
        for r in refs:
            a = f[r][()]                      # HDF5 sees Julia dims reversed
            c = a["re"] + 1j * a["im"]
            out.append(np.ascontiguousarray(c.transpose(*range(c.ndim - 1, -1, -1))))
    return out

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(src, "psi_mps_*.jld2")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    store = {}
    for p in files:
        step = int(re.search(r"(\d+)", os.path.basename(p)).group(1))
        tt = load_tt(p)
        for i, t in enumerate(tt):
            store[f"step{step}_core{i}"] = t
        store[f"step{step}_n"] = np.array(len(tt))
        print(f"step {step:>3}: {len(tt)} cores, bonds "
              f"{[t.shape[0] for t in tt] + [tt[-1].shape[-1]]}")
    np.savez_compressed(dst, **store)
    print("wrote", dst)
