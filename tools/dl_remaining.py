import sys
from huggingface_hub import hf_hub_download
DEST = "/nobackup/proj/disk/bloom/personal/shenghui/probe/third_party/CORAL/datasets/metas"
files = [
    "libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5",
    "libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5",
]
for f in files:
    print("downloading", f, flush=True)
    p = hf_hub_download(repo_id="yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
                        filename=f, local_dir=DEST)
    print("OK", p, flush=True)
print("ALL DONE")
