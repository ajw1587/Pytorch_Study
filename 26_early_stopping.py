
class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self.step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        print('self.step: ', self.step)
        print('self._loss: ', self._loss)
        print('self._patience: {} \n'.format(self.patience))

        if self._loss < loss:
            self.step += 1
            if self.step > self.patience:
                if self.verbose:
                    print(f'Training process is stopped early....')
                return True
        else:
            self.step = 0
            self._loss = loss

        return False


early_stopping = EarlyStopping(patience=10, verbose=1)
early_stopping.validate(10)
early_stopping.validate(11)
early_stopping.validate(12)
early_stopping.validate(13)
early_stopping.validate(14)
early_stopping.validate(15)