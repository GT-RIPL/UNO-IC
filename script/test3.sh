export CUDA_VISIBLE_DEVICES=1
for VAR in {50,45,40,35,30}
do
	python validate.py --config ./configs/rgbd_synthia.yml --beta $VAR --id "SoftmaxMultiply_${VAR}"
	
done