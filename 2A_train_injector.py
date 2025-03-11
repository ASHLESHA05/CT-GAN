import sys
import os
os.environ["CLI_ARGS"] = " ".join(arg.lower() for arg in sys.argv[1:]) if len(sys.argv) > 1 else "b"

from procedures.trainer import *

print("Training CT-GAN Injector...")
CTGAN_inj = Trainer(isInjector = True)
CTGAN_inj.train(epochs=50, batch_size=32, sample_interval=5)
print('Done.')