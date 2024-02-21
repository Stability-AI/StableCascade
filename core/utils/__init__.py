# MOVE IT SOMERWHERE ELSE
def update_weights_ema(tgt_model, src_model, beta=0.999):
    for self_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
        self_params.data = self_params.data * beta + src_params.data.clone().to(
            self_params.device
        ) * (1 - beta)
    for self_buffers, src_buffers in zip(tgt_model.buffers(), src_model.buffers()):
        self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(
            self_buffers.device
        ) * (1 - beta)
