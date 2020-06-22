# KorQuAD-Challenge
Enliple배 언어모델 튜닝대회 소스코드 (KorQuAD Challenge) 참가 소스 코드입니다.

기존 코드에서 변경되지 않은 부분들은 업로드 하지 않았습니다.

distill 폴더에는 Knowledge Distillation을 적용하는 코드들이 포함되어 있습니다.

train_distill.py는 soft_label 및 hard_label에 학습하기 위한 코드이며, Temperature 값과 Alpha 값은 modeling_bert_custom.py의 forward 함수에서 수정해주시면 됩니다.

get_distill_input.py 및 get_distill_prob.py는 각각 soft_label 추출을 위한 입력 생성과 Large Model을 Propgate하여 Soft Label을 생성하기 위한 코드입니다. 이 코드는 Large 모델을 세팅해놓은 프로젝트 폴더에서 사용됩니다.

https://www.slideshare.net/SanghyunCho9/enliple-korquad-challenge
