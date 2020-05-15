export CUDA_VISIBLE_DEVICES=0
for VAR in {30,20,10,0}
do
	python validate.py --config ./configs/rgbd_synthia.yml --beta $VAR --id "MixedGMM_${VAR}"
	
done
