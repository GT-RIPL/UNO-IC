# configs/train/gatedFusion/VarianceScaling/
# configs/train/gatedFusion/CAF/

yml=*.yml
files=$1$yml

echo "$files"

for f in $files; do
    echo $f
    CUDA_VISIBLE_DEVICES=$2 python -W ignore train.py --config $f
done