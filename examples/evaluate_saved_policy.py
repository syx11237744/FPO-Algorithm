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
"""One example for evaluate saved policy."""

import os

import omnisafe


# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
LOG_DIR = ''
if __name__ == '__main__':
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    scan_dir = os.scandir(os.path.join(LOG_DIR, 'torch_save'))
    last_pt_file = None
    for item in scan_dir:
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            if last_pt_file is None:
                last_pt_file = item
            elif item.name > last_pt_file.name:
                last_pt_file = item
    if last_pt_file is None:
        raise ValueError('No pt file found in the directory.')
    evaluator.load_saved(
        save_dir=LOG_DIR,
        model_name=last_pt_file.name,
        camera_name='3',
        # camera_id='track',
        width=1024,
        height=1024,
    )

    evaluator.render(num_episodes=1)

    # seed = 42
    # if not os.path.exists(LOG_DIR):
    #     raise ValueError(f"Checkpoint directory {LOG_DIR} does not exist.")
    # saved_dir = os.path.join(LOG_DIR, 'feasible_value')
    # os.makedirs(saved_dir, exist_ok=True)
    # for item in scan_dir:
    #     evaluator.load_saved(
    #         save_dir=LOG_DIR,
    #         model_name=item.name,
    #         camera_name='track',
    #         width=256,
    #         height=256,
    #     )
    #     evaluator.collect_obs(seed=seed,save_path=os.path.join(saved_dir, f'saved_obs_{item.name.split(".")[0]}_{seed}.npz'))
    scan_dir.close()
