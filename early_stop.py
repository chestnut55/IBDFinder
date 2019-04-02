from keras.callbacks import EarlyStopping

class LossCallBack(EarlyStopping):
    def __init__(self, loss, **kwargs):
        super(LossCallBack, self).__init__(**kwargs)
        self.loss = loss  # training loss

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('loss')
        # acc = logs.get('val_acc')
        if current is None:
            return

        # implement your own logic here
        if (current <= self.loss):
            self.stopped_epoch = epoch
            self.model.stop_training = True