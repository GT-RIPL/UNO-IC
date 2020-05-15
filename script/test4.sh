export CUDA_VISIBLE_DEVICES=1
for VAR in {25,20,15,10,5,0}
do
	python validate.py --config ./configs/rgbd_synthia.yml --beta $VAR --id "SoftmaxMultiply_${VAR}"
	
done