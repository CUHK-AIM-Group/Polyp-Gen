# Polyp-Gen

# 🛠Setup

```bash
git clone https://github.com/Saint-lsy/Polyp-Gen.git
cd Polyp-Gen
conda create -n PolypGen python=3.10
conda activate PolypGen
pip install -r requirements.txt
```

## Data Preparation
This model was trained by [LDPolypVideo](https://github.com/dashishi/LDPolypVideo-Benchmark) dataset.

We filtered out some low-quality images with blurry, reflective, and ghosting effects, and finally select 55,883 samples including 29,640 polyp frames and 26,243 non-polyp frames. 
## Checkpoint
You can download the chekpoints of our Polyp_Gen on [HuggingFace](https://huggingface.co/Saint-lsy/Polyp-Gen-sd2-inpainting/tree/main)  

## Sampling with Specified Mask
```
python sample_one_image.py
```

## Sampling with Mask Proposer
The first step is building database and Global Retrieval.
```bash
python GlobalRetrieval.py --data_path /path/of/non-polyp/images --database_path /path/to/build/database --image_path /path/of/query/image/
```
The second step is Local Matching for query image.
```bash
python LocalMatching.py --ref_image /path/ref/image --ref_mask /path/ref/mask --query_image /path/query/image --mask_proposal /path/to/save/mask
```
One Demo of LocalMatching
```bash
python LocalMatching.py --ref_image demos/img_1513_neg.jpg --ref_mask  demos/mask_1513.jpg --query_image  demos/img_1592_neg.jpg --mask_proposal gen_mask.jpg
```

The third step is using the generated Mask to sample.
