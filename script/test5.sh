export CUDA_VISIBLE_DEVICES=1
for VAR in {60,50,40}
do
	python validate.py --config ./configs/rgbd_synthia.yml --beta $VAR --id "MixedGMM_${VAR}"
	
done
