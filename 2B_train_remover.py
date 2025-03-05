from procedures.trainer import *

print("Training CT-GAN Remover...")
CTGAN_rem = Trainer(isInjector = False)
CTGAN_rem.train(epochs=10, batch_size=8, sample_interval=5)
print('Done.')