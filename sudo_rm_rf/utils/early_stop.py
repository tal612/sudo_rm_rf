class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    # relates to the loss as negative value
    def early_stop(self, validation_loss):
        if (validation_loss + self.min_delta) < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early Stopping :)")
                return True
        return False

    # def early_stop(self, validation_loss):
    #     if validation_loss < self.min_validation_loss:
    #         self.min_validation_loss = validation_loss
    #         self.counter = 0
    #     elif validation_loss > (self.min_validation_loss + self.min_delta):
    #         self.counter += 1
    #         if self.counter >= self.patience:
    #             return True
    #     return False