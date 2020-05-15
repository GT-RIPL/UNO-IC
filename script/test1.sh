export CUDA_VISIBLE_DEVICES=0
for VAR in {100,95,90,85,80}
do
	 python validate.py --config ./configs/rgbd_synthia.yml --beta $VAR --id "SoftmaxMultiply_${VAR}"
	
done








