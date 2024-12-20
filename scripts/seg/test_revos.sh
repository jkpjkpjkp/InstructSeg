#--------- ReVOS -------

save_path=output/revos

deepspeed instructseg/eval/seg/eval_revos.py \
    --revos_path dataset/ReVOS \
    --model_path model/InstructSeg \
    --save_path ${save_path} \
    --use_temporal_query True \
    --use_vmtf True \


python instructseg/eval/eval_tools/revos-evaluation/revos_evaluation.py \
    --revos_pred_path ${save_path}/Annotations \
