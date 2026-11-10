import torch
import torch.nn as nn
import timm

# 1. АРХИТЕКТУРА МОДЕЛИ С 2 ГОЛОВАМИ И ОБУЧАЕМЫМИ ВЕСАМИ LOSS (TODO 1)
class MultiTaskModel(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=4):
        super().__init__()
        # Бэкбон без классификационной головы
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0)
        in_features = self.encoder.num_features # 512 для resnet34
        
        # Головы
        self.class_head = nn.Linear(in_features, num_classes)
        self.reg_head = nn.Linear(in_features, 1)
        
        # TODO 1.1: Объяви 2 обучаемых параметра log_var1 и log_var2 типа nn.Parameter (начальное значение 0.0)
        # self.log_var1 = nn.Parameter(...)
        # self.log_var2 = nn.Parameter(...)
        self.log_var1 = nn.Parameter(torch.tensor(0,0))
        self.log_var2 = nn.Parameter(torch.tensor(0,0))
    def forward(self, x):
        features = self.encoder(x)
        
        # TODO 1.2: Верни предсказания классификации, регрессии и сами параметры log_var1, log_var2
        out_class = self.class_head(x)
        out_reg = self.encoder(x)
        return out_class, out_reg, self.log_var1, self.log_var2
        pass


# 2. РАСЧЕТ МУЛЬТИТАСК LOSS (TODO 2)
def compute_multitask_loss(out_class, out_reg, target_class, target_reg, log_var1, log_var2):
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    l_class = criterion_class(out_class, target_class)
    l_reg = criterion_reg(out_reg.squeeze(-1), target_reg)
    
    # TODO 2.1: Запиши формулу L_total через torch.exp(-log_var1), l_class, l_reg и log_var

    loss_total = 0.5 * torch.exp(-log_var1) * l_class + 0.5 * torch.exp(-log_var2) * l_reg + 0.5 * log_var1 + 0.5 * log_var2
    return loss_total
    pass

