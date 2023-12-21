class LayerControl:
    def __init__(self, opt):
        self.__opt = opt
        
    def freeze_layer(self, model):# freeze some layers
        try:
            if self.__opt.freeze_FeatureFxtraction:
                for param in model.module.FeatureExtraction.parameters():
                    param.requires_grad = False
                    
            if self.__opt.freeze_SequenceModeling:
                for param in model.module.SequenceModeling.parameters():
                    param.requires_grad = False
        except:
            pass   
        
        return model    