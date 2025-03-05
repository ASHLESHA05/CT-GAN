from procedures.trainer import *

print("Training CT-GAN Injector...")
CTGAN_inj = Trainer(isInjector = True)
CTGAN_inj.train(epochs=20, batch_size=16, sample_interval=5)
print('Done.')