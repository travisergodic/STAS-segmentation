## 安裝
```python
git clone https://github.com/travisergodic/STAS-segmentation.git
pip install segmentation-models-pytorch
pip install patchify
pip install transformers
pip install einops

cd /content/STAS-segmentation
mkdir models
git clone https://github.com/davda54/sam.git
```

## 使用
1. 訓練
```python
python main.py --mode train
```
2. 評估
```python
python main.py --mode "evaluate" --model_path "./models/model_path.pt" --multiscale "416, 320, 352, 384"
```

3. 預測
```python
python main.py --mode "make_prediction" --model_path "./models/model_v2.pt" --target_dir "./data/Public_Image/" \
                --mask_mode "color" --do_tta "True" --multiscale "416, 352, 384"
```
