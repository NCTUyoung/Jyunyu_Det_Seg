import os
import torch

def save_model(checkpoint_name, state, logger):
    logger.info("Save checkpoint: {}".format(checkpoint_name))
    torch.save(state, checkpoint_name)

def load_model(checkpoint, model, optimizer, start_epoch, logger, strict = True):
    assert os.path.exists(checkpoint),"{} doesn't exist!".format(checkpoint)
    state = torch.load(checkpoint)
    start_epoch = state["epoch"]
    model.load_state_dict(state["model_state"], strict = strict)
    # print(state["model_state"])
    # x = input("pause")
    if optimizer: 
        optimizer.load_state_dict(state["optimizer_state"])
    logger.info("Resume from previous checkpoint: {}, Strcit: {}".format(checkpoint, strict))
    return start_epoch, model, optimizer

def count_parameters(net):
    number_parameters = sum(p.numel() for p in net.parameters())
    return number_parameters