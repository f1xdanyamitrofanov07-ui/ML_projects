    import torch
    import torch.nn as nn
    import timm

    model = timm.create_model('resnet34', pretrained=True, num_classes=5)

    def get_llrd_optimizer_params(model):
        g1, g2, g3 = [], [], []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Determine parameter group and learning rate
            if 'layer1' in name or 'layer2' in name:
                g1.append((name, param))
            elif 'layer3' in name or 'layer4' in name:
                g2.append((name, param))
            else:
                g3.append((name, param))

        def make_group(params_list, lr):
            decay_params = [p for n, p in params_list if 'bias' not in n and 'bn' not in n]
            no_decay_params = [p for n, p in params_list if 'bias' in n or 'bn' in n]
            
            groups = []
            if decay_params:
                groups.append({'params': decay_params, 'lr': lr, 'weight_decay': 1e-2})
            if no_decay_params:
                groups.append({'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0})
            return groups

        param_groups = []
        param_groups.extend(make_group(g1, 1e-5))
        param_groups.extend(make_group(g2, 1e-4))
        param_groups.extend(make_group(g3, 1e-3))
        
        return param_groups

    # --- ИНИЦИАЛИЗАЦИЯ ОПТИМИЗАТОРА (TODO 2) ---
    # 1. Получи param_groups через get_llrd_optimizer_params(model)
    param_groups = get_llrd_optimizer_params(model)

    # 2. Инициализируй optimizer = torch.optim.AdamW(param_groups)
    optimizer = torch.optim.AdamW(param_groups)
