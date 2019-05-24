for f in ./configs/recalibration/*.yml; do
    # do some stuff here with "$f"
    # remember to quote it or spaces may misbehave
    CUDA_VISIBLE_DEVICES=1 python train.py --config $f
done
