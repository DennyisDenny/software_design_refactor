class Parameter:
    def __init__(self, model):
        self.__model = model
        
    def count_parameters(self):
        print("Modules, Parameters")
        total_params = 0
        for name, parameter in self.__model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params+=param
            print(name, param)
        print(f"Total Trainable Params: {total_params}")
        return total_params