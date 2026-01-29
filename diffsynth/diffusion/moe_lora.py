import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist






class TokenWiseGatedMoELoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        num_experts: int,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        top_k: int = 1,
        full_name: str = "",
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_experts = num_experts
        self.r = r
        self.top_k = top_k
        self.full_name = full_name
        self.training = True

        self.current_aux_loss = 0.0 


        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.ModuleDict({
            f"expert_{i}": nn.Linear(base_layer.in_features, r, bias=False)
            for i in range(num_experts)
        })
        self.lora_B = nn.ModuleDict({
            f"expert_{i}": nn.Linear(r, base_layer.out_features, bias=False)
            for i in range(num_experts)
        })
        
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        

        self.gate = nn.Linear(base_layer.in_features, num_experts, bias=False)
        

        nn.init.normal_(self.gate.weight, mean=0, std=0.01)
        
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[f"expert_{i}"].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[f"expert_{i}"].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [Batch, Tokens, Dim] (例如 [B, N, C])
        """
        orig_dtype = x.dtype

        result = self.base_layer(x)
        

        route_logits = self.gate(x) # [B, N, num_experts]
        

        all_probs = F.softmax(route_logits, dim=-1, dtype=torch.float32) 
        

        top_k_probs, top_k_indices = torch.topk(all_probs, k=self.top_k, dim=-1)
        
        if self.top_k == 1:

            top_k_probs = 1.0 + (top_k_probs - top_k_probs.detach())
        else:

            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        top_k_probs = top_k_probs.to(orig_dtype)

        if self.training:

            me = torch.mean(all_probs, dim=(0, 1)) # [num_experts]
            

            expert_mask = torch.zeros_like(route_logits, dtype=torch.float32)
            expert_mask.scatter_(-1, top_k_indices, 1.0)
            ce = torch.mean(expert_mask, dim=(0, 1)) # [num_experts]

            self.current_aux_loss = self.num_experts * torch.sum(me * ce)
        else:
            self.current_aux_loss = 0.0

        route_weight = torch.zeros_like(route_logits, dtype=orig_dtype)
        route_weight.scatter_(-1, top_k_indices, top_k_probs)
        
        x_dn = self.dropout(x)
        lora_delta = torch.zeros_like(result, dtype=orig_dtype)
        
        for i in range(self.num_experts):

            mask = route_weight[:, :, i].unsqueeze(-1)


            expert_key = f"expert_{i}"

            expert_out = self.lora_B[expert_key](self.lora_A[expert_key](x_dn))


        return result + lora_delta * self.scaling


def replace_target_modules_with_moe_lora(
    model: nn.Module,
    target_modules: list[str],
    num_experts: int = 4,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    top_k: int = 1,
    upcast_dtype: torch.dtype = None,
    use_weighted_routing: bool = True,
    moe_type: str = "moe",
):


    target_names = []
    for name, module in model.named_modules():

        if any(target_str in name for target_str in target_modules):
            if isinstance(module, nn.Linear):
                target_names.append(name)
    
    if not target_names:
        print(f"⚠️ No modules found containing any of: {target_modules}")
        return model
    
    print(f"✅ Found {len(target_names)} modules to replace with MoE LoRA:")

    target_names.sort(key=lambda x: len(x.split('.')), reverse=True)
    

    for full_name in target_names:
        *parent_path, module_name = full_name.split('.')
        

        parent_module = model
        for p in parent_path:
            parent_module = getattr(parent_module, p)

        original_layer = getattr(parent_module, module_name)
        if moe_type == "ranmoe":

            moe_lora_layer = RandomGatedMoELoraLinear(
                base_layer=original_layer,
                num_experts=num_experts,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                top_k=top_k,
                use_weighted_routing=True,
            )
        elif moe_type == "dismoe":
            moe_lora_layer = DistributedGatedMoELoraLinear(
                base_layer=original_layer,
                num_experts=num_experts,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                top_k=top_k,
                full_name=full_name

            )
        elif moe_type == "tokenmoe":
            moe_lora_layer = TokenWiseGatedMoELoraLinear(
                base_layer=original_layer,
                num_experts=num_experts,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                top_k=top_k,
                full_name=full_name
            )
        else:
            moe_lora_layer = GatedMoELoraLinear(
                base_layer=original_layer,
                num_experts=num_experts,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                top_k=top_k
            )
        

        setattr(parent_module, module_name, moe_lora_layer)
        # print(f"✅ Replaced {full_name} with MoE LoRA ({num_experts} experts)")
    
    if upcast_dtype is not None:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.to(upcast_dtype)

    return model
