export CUDA_VISIBLE_DEVICES=0
for VAR in {75,70,65,60,55}
do
	python validate.py --config ./configs/rgbd_synthia.yml --beta $VAR --id "SoftmaxMultiply_${VAR}"
done