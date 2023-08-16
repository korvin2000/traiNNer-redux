#!/bin/bash
SVR_NAME=Jupiter

# LABEL=train_EDSR_Lx4
# LABEL=train_RRDBNet_PSNR_x4_LRGBL1D2KC5
# LABEL=train_RRDBNet_PSNR_x4_LRGBL1D2KC30
LABEL=train_RRDBNet_PSNR_x4_LRGBL1FULL
OPTION=options/train/LRGB/$LABEL.yml

# LABEL=test_RRDBNet_PSNR_x4
# OPTION=options/test/LRGB/$LABEL.yml
# CUDA_VISIBLE_DEVICES=2 python basicsr/test.py -opt $OPTION

# CUDA_VISIBLE_DEVICES=2 python basicsr/train.py -opt $OPTION

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 \
    basicsr/train.py -opt $OPTION --launcher pytorch

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4323 \
#     basicsr/train.py -opt $OPTION --launcher pytorch


# Send finish message.
SMS_MSG=${LABEL}' @ '${SVR_NAME}' is over.'
curl 'https://api.twilio.com/2010-04-01/Accounts/'${TWILIO_ACCOUNT_SID}'/Messages.json' -X POST \
    --data-urlencode 'To='${MY_PHONE_NUMBER} \
    --data-urlencode 'From='${TWILIO_PHONE_NUMBER} \
    --data-urlencode 'Body='"${SMS_MSG}" \
    -u ${TWILIO_ACCOUNT_SID}:${TWILIO_AUTHTOKEN}
