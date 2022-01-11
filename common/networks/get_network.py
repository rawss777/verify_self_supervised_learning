from .encoder import resnet, sk_resnet
from .ssl import byol, deepclusterv2, moco, simclr, simsiam, swav


def get_network(criterion, device, cfg_encoder, cfg_self_supervised, cfg_dataset):
    img_size = cfg_dataset.img_size
    input_channel = cfg_dataset.input_channel

    # get encoder
    encoder_class = getattr(eval(cfg_encoder.root), cfg_encoder.name)
    variant_params = cfg_encoder[cfg_encoder.variant]
    encoder_model = encoder_class(img_size, input_channel, **cfg_encoder.params, **variant_params)

    # get self supervised
    self_supervised_class = getattr(eval(cfg_self_supervised.root), cfg_self_supervised.name)
    self_supervised_model = self_supervised_class(encoder_model, criterion, device, **cfg_self_supervised.params)

    return self_supervised_model


