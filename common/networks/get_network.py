from .encoder import resnet, sk_resnet
# from .ssl import barlow_twins, byol, deepclusterv2, moco, simclr, simsiam, swav
from . import ssl


def get_network(device, cfg_encoder, cfg_self_supervised, cfg_dataset):
    img_size = cfg_dataset.img_size
    input_channel = cfg_dataset.input_channel

    # get encoder
    encoder_class = getattr(eval(cfg_encoder.root), cfg_encoder.name)
    variant_params = cfg_encoder[cfg_encoder.variant]
    encoder_model = encoder_class(img_size, input_channel, **cfg_encoder.params, **variant_params)

    # get self supervised
    self_supervised_class = getattr(eval(cfg_self_supervised.root), cfg_self_supervised.name)
    self_supervised_model = self_supervised_class(encoder_model, device, cfg_self_supervised.loss_params,
                                                  **cfg_self_supervised.ssl_params)

    return self_supervised_model


