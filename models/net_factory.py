from .fishnet import fish


def fishnet99(**kwargs):
    """

    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 2, 6, 2, 1, 1, 1, 1, 2, 2],
        'num_trans_blks': [1, 1, 1, 1, 1, 4],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)


def fishnet150(**kwargs):
    """

    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 4, 8, 4, 2, 2, 2, 2, 2, 4],
        'num_trans_blks': [2, 2, 2, 2, 2, 4],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)


def fishnet201(**kwargs):
    """

    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [3, 4, 12, 4, 2, 2, 2, 2, 3, 10],
        'num_trans_blks': [2, 2, 2, 2, 2, 9],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)
