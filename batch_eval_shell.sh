
# -----------------------------------------chatgpt class ----------------------------------------#
# different geo factors

# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.1.py  grounding_dino_chatgpt_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.3_3 "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." grounding_dino_chatgpt_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.3_3.json
# wait

# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.3.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.4 "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." grounding_dino-chatgpt-text_dual_encoder_decoder_lr1e-4_grad_clip0.1_ema0.99_select_thr0.4_geo0.3.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.4.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.4 "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." grounding_dino-chatgpt-text_dual_encoder_decoder_lr1e-4_grad_clip0.1_ema0.99_select_thr0.4_geo0.4.json
# wait

# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.5.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.4 "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." grounding_dino-chatgpt-text_dual_encoder_decoder_lr1e-4_grad_clip0.1_ema0.99_select_thr0.4_geo0.5.json


# different exp settings
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.1.py  grounding_dino_chatgpt_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_no_sematic_loss "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." grounding_dino_chatgpt_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_no_sematic_loss_geo0.1.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.1.py  grounding_dino_chatgpt_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_no_all_text_class_no_sematic_loss "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." grounding_dino_chatgpt_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_no_all_text_class_no_sematic_loss_geo0.1.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.1.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.1 "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." grounding_dino-chatgpt-text_dual_encoder_decoder_lr1e-4_grad_clip0.1_ema0.99_select_thr0.1_geo0.1.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.1.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.7_select_thr0.3 "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." grounding_dino-chatgpt-text_dual_encoder_decoder_lr1e-4_grad_clip0.1_ema0.7_select_thr0.3_geo0.1.json


# -----------------------------------------super class ----------------------------------------#
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.1.py  grounding_dino_chatgpt_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.3_3 "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.6_geo0.2.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.2.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.5 "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.5_geo0.2.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.2.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.4 "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.4_geo0.2.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.2.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.2_2 "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.2_geo0.2_2.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.2.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.1 "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.2.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.2.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0 "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.json

# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.1.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_no_sematic_loss "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_no_sematic_loss_geo0.1.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.1.py  grounding_dino_chatgpt_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_no_all_text_class_no_sematic_loss "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_chatgpt_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_no_all_text_class_no_sematic_loss_geo0.1.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.3.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.3_2 "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.3_geo0.3_2.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.4.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.3_2 "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.3_geo0.4_2.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/geo_ablation/grounding_dino_swin-t_finetune_16xb2_1x_soda_geo0.5.py  grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.3_2 "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." grounding_dino_small_prompt_swin-t_dual_encoder_decoder_query_head_lr_1e-4_1x_ema_bank0.99_select_thr0.3_geo0.5_2.json

# bash eval_shell.sh configs/auto_driving_grounding_dino/ablation/auto_driving_grounding_dino_swin-t_16xb2_1x_soda_geo0.6.py  auto_driving_grounding_dino_swin-t_no_text "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." auto_driving_grounding_dino_swin-t_no_text_geo0.6.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/auto_driving_grounding_dino_swin-t_16xb2_1x_soda.py  auto_driving_grounding_dino_swin-t_no_text "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." auto_driving_grounding_dino_swin-t_no_text_llm_text_test_geo0.1.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/ablation/auto_driving_grounding_dino_swin-t_16xb2_1x_soda_geo0.2.py auto_driving_grounding_dino_swin-t_no_text "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." auto_driving_grounding_dino_swin-t_no_text_llm_text_test_geo0.2.json
# wait
# bash eval_shell.sh configs/auto_driving_grounding_dino/ablation/auto_driving_grounding_dino_swin-t_16xb2_1x_soda_geo0.2.py  auto_driving_grounding_dino_swin-t_no_text "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." auto_driving_grounding_dino_swin-t_no_text_llm_text_test_geo0.2.json
# bash eval_shell.sh configs/auto_driving_grounding_dino/ablation/auto_driving_grounding_dino_swin-t_16xb2_1x_soda_geo0.4.py  auto_driving_grounding_dino_swin-t_no_text "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." auto_driving_grounding_dino_swin-t_no_text_geo0.4.json
# wait

bash eval_shell.sh configs/auto_driving_grounding_dino/ablation/auto_driving_grounding_dino_swin-t_16xb2_1x_soda_geo0.2.py  auto_driving_grounding_dino_swin-t_no_text "pedestrian . cyclist . car . truck . bus . tricycle . motorcycle . bicycle . van . scooter . moped . construction equipment . emergency vehicle . farm equipment . recreational vehicle . electric vehicle . hybrid vehicle . public transportation . aircraft .  watercraft . specialty vehicle . road sign . traffic signal  . speed bump . traffic cone . barrier system . road marker . traffic signal and sign post . traffic camera . variable message sign . school zone sign . traffic light . roundabout . bollard . rumble strip . deer . squirrel . bird . dog . cat . livestock . wild animal . insect . amphibian . reptile . litter . natural debris . vehicle part . debris from accidents . animal remains . obstacle . debris from storm . debris from landslide . miscellaneous item ." auto_driving_grounding_dino_swin-t_no_text_llm_text_test_geo0.2.json