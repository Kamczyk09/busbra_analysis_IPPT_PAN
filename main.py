from data.cub200 import load_data_with_segmentation

train_ds, test_ds, val_ds = load_data_with_segmentation()

print(len(train_ds))
print(len(test_ds))
print(len(val_ds))
####################################################################

"""
Ten fragment służy do treningu sieci. Funkcja 'train' przyjmuje parametr 'model',
czyli będzie dotrenowywać istniejąca sieć o nazwie model. Wagi modelu zapisywane są w katalogu models_checkpoints

Funkcja evaluate liczy ACC dla wytrenowanej sieci
"""
# from models import ResNet18
# from torchvision.models import resnet18
#
# model = ResNet18.return_model(200, pretrained=True)
#
# model = ResNet18.train(nEpochs=20,
#                           lr=1e-4,
#                           model=model,
#                           name="resnet18_cub_pretrained")
# ResNet18.evaluate(name="resnet18_cub_pretrained")
#
# #najlepszy model: resnet18_busbra_pretrained -- ACC = 77%
# #najlepszy model: resnet101_busbra_pretrained_freezed -- ACC = 75%
# #najlepszy model: resnet152_busbra_pretrained_freezed -- ACC = 77%

# ##########################################################################################


