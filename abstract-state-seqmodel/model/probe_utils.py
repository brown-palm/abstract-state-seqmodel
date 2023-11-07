import torch
import torch.nn as nn
from torch.nn import functional as F



class ClassificationProbe(nn.Module):
    """
    Implementation from https://github.com/likenneth/othello_world/blob/b6d57e4b7d6078c934d36032e73a6cc7871a72ed/mingpt/probe_model.py#L81
    """
    def __init__(self, device, num_cells, num_classes, intermediate_dim, input_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.num_cells = num_cells
        self.num_classes = num_classes

        self.probe = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.intermediate_dim, self.num_classes * self.num_cells, bias=True),
        )  # Binary classification using CrossEntropyLoss
        
        self.apply(self._init_weights)
        self.to(device)
        
    def forward(self, inputs, labels=None):
        # inputs: [B, input_dim]
        logits = self.probe(inputs).reshape(-1, self.num_cells, self.num_classes)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            return logits, loss
        
        else:
            return logits, None
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, lr, weight_decay, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn # full param name
                if pn.endswith("bias"):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=lr, betas=betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=0)
        return optimizer, scheduler


class AgentStateProbe(nn.Module):
    """
    Implementation from https://github.com/likenneth/othello_world/blob/b6d57e4b7d6078c934d36032e73a6cc7871a72ed/mingpt/probe_model.py#L81
    
    Probe for agent state in BabyAI, where agent state involves agent"s location and orientation.
    """
    def __init__(self, device, num_locations, intermediate_dim, input_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.num_locations = num_locations
        
        self.location_probe = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.intermediate_dim, self.num_locations, bias=True),
        )
        
        self.apply(self._init_weights)
        self.to(device)
        
    def forward(self, inputs, y_locations=None):
        # inputs: [B, input_dim]
        location_logits = self.location_probe(inputs).reshape(-1, self.num_locations)
        
        if y_locations is not None:
            location_loss = F.cross_entropy(
                location_logits.view(-1, location_logits.size(-1)), y_locations.view(-1)
            )
            return location_logits, location_loss
        
        else:
            return location_logits, None
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, lr, weight_decay, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won"t experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn # full param name
                if pn.endswith("bias"):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=lr, betas=betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=0)
        return optimizer, scheduler
    
    
class BoardRecoveryProbe(nn.Module):
    """
    Implementation from https://github.com/likenneth/othello_world/blob/b6d57e4b7d6078c934d36032e73a6cc7871a72ed/mingpt/probe_model.py#L81
    
    Probe for board recovery in BabyAI, where each cell on the board involves an object and a color.
    """
    def __init__(self, device, num_cells, num_objects, num_colors, intermediate_dim, input_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.num_cells = num_cells
        self.num_objects = num_objects
        self.num_colors = num_colors
        
        self.object_probe = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.intermediate_dim, self.num_objects * self.num_cells, bias=True),
        )
        self.color_probe = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.intermediate_dim, self.num_colors * self.num_cells, bias=True),
        )
        
        self.apply(self._init_weights)
        self.to(device)
        
    def forward(self, inputs, y_objects=None, y_colors=None):
        # inputs: [B, input_dim]
        object_logits = self.object_probe(inputs).reshape(-1, self.num_cells, self.num_objects)
        color_logits = self.color_probe(inputs).reshape(-1, self.num_cells, self.num_colors)
                
        if (y_objects is not None) and (y_colors is not None):
            object_loss = F.cross_entropy(
                object_logits.view(-1, object_logits.size(-1)), y_objects.view(-1)
            )
            color_loss = F.cross_entropy(
                color_logits.view(-1, color_logits.size(-1)), y_colors.view(-1), 
            )
            return object_logits, color_logits, object_loss, color_loss
        
        else:
            return object_logits, color_logits, None, None
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, lr, weight_decay, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn # full param name
                if pn.endswith("bias"):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=lr, betas=betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=0)
        return optimizer, scheduler

class OccupancyBoardRecoveryProbe(nn.Module):
    """
    Implementation from https://github.com/likenneth/othello_world/blob/b6d57e4b7d6078c934d36032e73a6cc7871a72ed/mingpt/probe_model.py#L81
    
    Probe for board recovery in BabyAI, where each cell on the board involves an object and a color.
    """
    def __init__(self, device, num_cells, num_classes, intermediate_dim, input_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.num_cells = num_cells
        self.num_classes = num_classes

        self.occupancy_probe = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.intermediate_dim, self.num_classes * self.num_cells, bias=True),
        )  # Binary classification using CrossEntropyLoss
        
        self.apply(self._init_weights)
        self.to(device)
        
    def forward(self, inputs, y_occupancy=None):
        # inputs: [B, input_dim]
        occupancy_logits = self.occupancy_probe(inputs).reshape(-1, self.num_cells, self.num_classes)

        if y_occupancy is not None:
            occupancy_loss = F.cross_entropy(
                occupancy_logits.view(-1, occupancy_logits.size(-1)), y_occupancy.view(-1)
            )
            return occupancy_logits, occupancy_loss
        
        else:
            return occupancy_logits, None
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, lr, weight_decay, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn # full param name
                if pn.endswith("bias"):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=lr, betas=betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=0)
        return optimizer, scheduler
