# configs/train/gatedFusion/VarianceScaling/
# configs/train/gatedFusion/CAF/

echo $1

for n in $(seq 1 $3); do
    echo $n
    CUDA_VISIBLE_DEVICES=$2 python -W ignore train.py --config $1 --tag $n
done