atc --framework=1 \
	--model=unet_hw960_bs1.air \
	--output=unet_hw960_bs1_AIPP \
	--insert_op_conf=./aipp_unet_simple_opencv.cfg \
	--input_format=NCHW \
	--input_shape="x:1,3,960,960" \
	--soc_version=Ascend310 \
	--log=error

#atc --framework=1 \
#	--model=unet_hw960_bs1.air \
#	--output=unet_hw960_bs1_noAIPP \
#	--input_format=NCHW \
#	--input_shape="x:1,3,960,960" \
#	--soc_version=Ascend310 \
#	--log=error


#atc --framework=1 \
#	--model=fcn-4.air \
#	--output=fcn-4 \
#	--dynamic_batch_size="1,2,4,8,10,12,16" \
#	--input_shape="-1,1,96,1366" \
#	--input_format=NCHW \
#	--soc_version=Ascend310 \
#	--log=error
