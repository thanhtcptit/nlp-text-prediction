import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.callbacks import LearningRateScheduler


def bell_curve_lr_scheduler(lr_start=1e-5, lr_max=5e-5, lr_min=1e-6, lr_rampup_epochs=7, 
                            lr_sustain_epochs=0, lr_exp_decay=.87):
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * \
                lr_exp_decay ** (epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


def decay_lr_scheduler(lr=1e-5):
    def lrfn(epoch):
        return lr / (epoch + 1)
    return lrfn


def get_lr_scheduler(config):
    if not isinstance(config, dict):
        return config
    if config["type"] == "decay":
        return LearningRateScheduler(decay_lr_scheduler(config["lr"]))
    if config["type"] == "bell":
        return LearningRateScheduler(bell_curve_lr_scheduler(
            config["lr_start"]), config["lr_max"], config["lr_min"],
            config["lr_rampup_epochs"], config["lr_sustain_epochs"], config["lr_exp_decay"])


def get_optimizer(config):
    if config["type"] == "adam":
        return keras.optimizers.Adam(lr=get_lr_scheduler(config["learning_rate"]))
    raise ValueError(config)
