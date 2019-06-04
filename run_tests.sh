<<tmp
for f in ./configs/fusion/*/*.yml; do
    echo $f
    CUDA_VISIBLE_DEVICES=0 python -W ignore train.py --config $f
done

for f in ./configs/recalibration/*/*.yml; do
    echo $f
    CUDA_VISIBLE_DEVICES=0 python -W ignore train.py --config $f
done
tmp


for f in ./configs/tempScaling/*/*.yml; do
    echo $f
    CUDA_VISIBLE_DEVICES=0 python -W ignore train.py --config $f
done