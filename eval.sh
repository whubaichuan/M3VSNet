#!/usr/bin/env bash
#DTU_TESTING="./dtu/"
DTU_TESTING="/data/mvsnet/dtu_point/"
#DTU_TESTING="/data/mvsnet/TT/intermediate/"
#CKPT_FILE="./checkpoint/model_000008.ckpt"
#model_000008.ckpt normal model_00000.ckpt no_normal
CKPT_FILE="./checkpoint/22.ckpt"
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@
