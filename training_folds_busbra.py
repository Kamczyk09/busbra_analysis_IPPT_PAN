# importy
#
# for fold in k-folds:
#     resnet: trening + eval + saving state (busbra)
#     clip: trening + eval + saving state (busbra)

from models import ResNet18, CLIP_lora

n_folds = range(1,11)

for fold in n_folds:

    #resnet18
    ResNet18.train(nEpochs=10,
                   lr=0.0001,
                   data_fold_no=fold,
                   name=f"resnet18_busbra_raw_fold{fold}")
    ResNet18.evaluate(name=f"resnet18_busbra_raw_fold{fold}",
                      data_fold_no=fold)

    #clip + lora
    CLIP_lora.train(nEpochs=10,
               lr=0.0001,
               data_fold_no=fold,
               name=f"CLIP_lora_busbra_raw_fold{fold}")
    CLIP_lora.evaluate(name=f"CLIP_lora_busbra_raw_fold{fold}",
                  data_fold_no=fold)