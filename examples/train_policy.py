# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of training a policy with OmniSafe."""

import argparse

import omnisafe
print(omnisafe.__file__)
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        metavar='ALGO',
        default='PPOLag',
        help='algorithm to train',
        choices=omnisafe.ALGORITHMS['all'],
    )
    parser.add_argument(
        '--env-id',
        type=str,
        metavar='ENV',
        default='SafetyPointGoal1-v0',
        help='the name of test environment',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=10000000,
        metavar='STEPS',
        help='total number of steps to train for algorithm',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        metavar='DEVICES',
        help='device to use for training',
    )
    parser.add_argument(
        '--task_description',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        metavar='SEED',
        help='random seed for training',
    )
    parser.add_argument(
        '--feasibility_threshold',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--weight_schedule',
        type=str,
        default='exp',
        help='exp/lin/fix',
    )
    parser.add_argument(
        '--feasibility_type',
        type=str,
        default='cdf',
        help='cdf / cvf',
    )
    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        seed=args.seed,
        train_terminal_cfgs=vars(args),
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
