import torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC


class TorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "mask" in orig_space.spaces
            and "obs" in orig_space.spaces
        )

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["obs"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal"
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["mask"]
        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["obs"]})
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
