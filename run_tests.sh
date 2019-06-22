
for f in configs/experiments/fusion/WeightedVariance/*.yml; do
    echo $f
    CUDA_VISIBLE_DEVICES=1 python -W ignore train.py --config $f >> "WeightedVariance.log"
done