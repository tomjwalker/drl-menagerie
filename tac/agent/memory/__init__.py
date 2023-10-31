from .onpolicy import OnPolicyReplay
from .onpolicy import OnPolicyBatchReplay

########################################################################################################################
# Name mappings
########################################################################################################################

memory_classes = {
    "on_policy": OnPolicyReplay,
    "on_policy_batch": OnPolicyBatchReplay,
}