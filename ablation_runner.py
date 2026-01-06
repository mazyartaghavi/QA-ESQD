def run_ablation(config, disable_qd=False, disable_qa=False):
    config = config.copy()

    if disable_qd:
        config["qd"]["archive_bins"] = None

    if disable_qa:
        config["quantum"]["enabled"] = False

    return train_q_evoqD(config)
