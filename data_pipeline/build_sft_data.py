import json
import random
import re
from pathlib import Path
from typing import Dict, List

from names import get_first_name

from .prompt_templates import LANG_CHAIN_PROMPT


# Replace these path placeholders before running the pipeline.
RESULTS_DIR = Path("path/to/VillagerAgent/result")
SFT_OUTPUT_PATH = Path("path/to/sft_dataset.json")
TRAJECTORY_OUTPUT_PATH = Path("path/to/trajectory_data.json")

SUCCESS_SCORE_THRESHOLD = 50

TOOL_LIST = '''eat: eat(player_name: str, item_name: str, emotion: list, murmur: str) - Eat Item, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
talkTo: talkTo(player_name: str, entity_name: str, message: str, emotion: list = ['😊']) - Talk to the Entity with Emojis, entity_name is the name of other player., args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'entity_name': {'title': 'Entity Name', 'type': 'string'}, 'message': {'title': 'Message', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'default': ['😊'], 'type': 'array', 'items': {}}}
fetchContainerContents: fetchContainerContents(player_name: str, item_name: str, position: list, emotion: list, murmur: str) - Get the details of item_name at [x, y, z] 'chest' | 'container' | 'furnace', arg position is [x, y, z], return ('message': msg, 'status': True/False, 'data':[('name':name, 'count':count),...]), args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'position': {'title': 'Position', 'type': 'array', 'items': {}}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
performMovement: performMovement(player_name: str, action_name: str, seconds: int, emotion: list, murmur: str) - Perform Action jump forward back left right for Seconds, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'action_name': {'title': 'Action Name', 'type': 'string'}, 'seconds': {'title': 'Seconds', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
attackTarget: attackTarget(player_name: str, target_name: str, emotion: list = ['😢'], murmur: str = '') - Attack the Nearest Entity with a Specific Name, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'target_name': {'title': 'Target Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'default': ['😢'], 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'default': '', 'type': 'string'}}
navigateTo: navigateTo(player_name: str, x: int, y: int, z: int, emotion: list, murmur: str) - Move to a Specific Position x y z, return string result, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'x': {'title': 'X', 'type': 'integer'}, 'y': {'title': 'Y', 'type': 'integer'}, 'z': {'title': 'Z', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
craftBlock: craftBlock(player_name: str, item_name: str, count: int, emotion: list, murmur: str) - Craft Item in the Crafting Table, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'count': {'title': 'Count', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
placeBlock: placeBlock(player_name: str, item_name: str, x: int, y: int, z: int, facing: str, emotion: list, murmur: str) - Place a Specific Item at Specific Position x y z with Specific facing in one of [W, E, S, N, x, y, z, A] default is 'A'., return ('message': msg, 'status': True/False), args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'x': {'title': 'X', 'type': 'integer'}, 'y': {'title': 'Y', 'type': 'integer'}, 'z': {'title': 'Z', 'type': 'integer'}, 'facing': {'title': 'Facing', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
mountEntity: mountEntity(player_name: str, entity_name: str, emotion: list = ['🏇', '😊'], murmur: str = '') - Mount the Entity, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'entity_name': {'title': 'Entity Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'default': ['🏇', '😊'], 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'default': '', 'type': 'string'}}
handoverBlock: handoverBlock(player_name: str, target_player_name: str, item_name: str, item_count: int, emotion: list, murmur: str) - Hand Item to a target player you work with, return ('message': msg, 'status': True/False), item num will be automatically checked and player will automatically move to the target player, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'target_player_name': {'title': 'Target Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'item_count': {'title': 'Item Count', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
equipItem: equipItem(player_name: str, slot: str, item_name: str, emotion: list, murmur: str) - Equip a Specific Item on a Specific Slot | to equip item on hand,head,torso,legs,feet., args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'slot': {'title': 'Slot', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
useItemOnBlock: useItemOnBlock(player_name: str, item_name: str, x: int, y: int, z: int, emotion: list, murmur: str) - Use a Specific Item on a Specific block at x y z, return string result (minecaft on rail, hoe on dirt, seeds on farmland, bucket on water, etc), args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'x': {'title': 'X', 'type': 'integer'}, 'y': {'title': 'Y', 'type': 'integer'}, 'z': {'title': 'Z', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
useItemOnEntity: useItemOnEntity(player_name: str, item_name: str, entity_name: str, emotion: list, murmur: str) - Use a Specific Item on a Specific Entity, return string result (bone on dog, bucket on cow, shears on sheep, saddle on horse, etc), args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'entity_name': {'title': 'Entity Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
openContainer: openContainer(player_name: str, container_name: str, position: list, emotion: list, murmur: str) - Open the nearest or at [x, y, z] 'chest' | 'container' | 'furnace' position is optional, return ('message': msg, 'status': True/False, 'data':[('name':name, 'count':count),...]), args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'container_name': {'title': 'Container Name', 'type': 'string'}, 'position': {'title': 'Position', 'type': 'array', 'items': {}}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
scanNearbyEntities: scanNearbyEntities(player_name: str, item_name: str, radius: int, item_num: int, emotion: list, murmur: str) - Find minecraft item blocks chests creatures in a radius, return ('message': msg, 'status': True/False, 'data':[('x':x,'y':y,'z':z),...]) This function can not find items in the chest, container,or player's inventory., args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'radius': {'title': 'Radius', 'type': 'integer'}, 'item_num': {'title': 'Item Num', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
ToggleAction: ToggleAction(player_name: str, item_name: str, x: int, y: int, z: int, emotion: list, murmur: str) - open/close Gate, Lever, Press Button (pressure_plate need to stand on it, iron door need to be powered, they are not included), at Specific Position x y z, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'x': {'title': 'X', 'type': 'integer'}, 'y': {'title': 'Y', 'type': 'integer'}, 'z': {'title': 'Z', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
dismountEntity: dismountEntity(player_name: str, emotion: list, murmur: str) - Dismount the Entity, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
sleep: sleep(player_name: str, emotion: list, murmur: str) - Go to Sleep, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
wake: wake(player_name: str, emotion: list, murmur: str) - Wake Up, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
SmeltingCooking: SmeltingCooking(player_name: str, item_name: str, item_count: int, fuel_item_name: str, emotion: list, murmur: str) - Smelt or Cook Item in the Furnace, item_name is the item to be smelted, item_count is the number of items to be smelted, fuel_item_name is the fuel item., args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'item_count': {'title': 'Item Count', 'type': 'integer'}, 'fuel_item_name': {'title': 'Fuel Item Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
waitForFeedback: waitForFeedback(player_name: str, entity_name: str, seconds: int = 10, emotion: list = ['⏱️'], murmur: str = '') - Wait for other player's reply, except you or others are expecting to end the conversation., args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'entity_name': {'title': 'Entity Name', 'type': 'string'}, 'seconds': {'title': 'Seconds', 'default': 10, 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'default': ['⏱️'], 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'default': '', 'type': 'string'}}
storeItem: storeItem(player_name: str, item_name: str, to_name: str, item_count: int, emotion: list, murmur: str) - Put in Item to One Chest, Container, etc, return string result, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'to_name': {'title': 'To Name', 'type': 'string'}, 'item_count': {'title': 'Item Count', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
withdrawItem: withdrawItem(player_name: str, item_name: str, from_name: str, item_count: int, emotion: list, murmur: str) - Take out Item from nearest 'chest' | 'container' | 'furnace' return string result, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'from_name': {'title': 'From Name', 'type': 'string'}, 'item_count': {'title': 'Item Count', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
startFishing: startFishing(player_name: str, fish_name: str, emotion: list, murmur: str) - Start Fishing, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'fish_name': {'title': 'Fish Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
read: read(player_name: str, item_name: str, emotion: list, murmur: str) - Read Book or Sign neaby, return string details, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'item_name': {'title': 'Item Name', 'type': 'string'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
MineBlock: MineBlock(player_name: str, x: int, y: int, z: int, emotion: list, murmur: str) - Dig Block at Specific Position x y z, args: {'player_name': {'title': 'Player Name', 'type': 'string'}, 'x': {'title': 'X', 'type': 'integer'}, 'y': {'title': 'Y', 'type': 'integer'}, 'z': {'title': 'Z', 'type': 'integer'}, 'emotion': {'title': 'Emotion', 'type': 'array', 'items': {}}, 'murmur': {'title': 'Murmur', 'type': 'string'}}
'''

def shuffle_tool_list():
    # Split the tool description into individual entries.
    tool_lines = [line.strip() for line in TOOL_LIST.split('\n') if line.strip()]

    tools = []
    for line in tool_lines:
        tool_name = line.split(':', 1)[0].strip()
        tools.append((tool_name, line))

    random.shuffle(tools)

    shuffled_tool_list = '\n'.join([tool[1] for tool in tools])
    shuffled_order = ', '.join([tool[0] for tool in tools])

    return shuffled_tool_list, shuffled_order


def load_successful_trajectories(results_dir: Path = RESULTS_DIR) -> List[Dict]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"VillagerAgent result directory not found: {results_dir}")

    trajectories = []
    for task_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        score_path = task_dir / "score.json"
        history_path = task_dir / "Alice_history.json"
        config_path = task_dir / "config.json"
        required_paths = (score_path, history_path, config_path)
        if not all(path.is_file() for path in required_paths):
            continue

        try:
            with score_path.open("r", encoding="utf-8") as f:
                score = json.load(f).get("score")
            if not isinstance(score, (int, float)) or score < SUCCESS_SCORE_THRESHOLD:
                continue

            with history_path.open("r", encoding="utf-8") as f:
                action_history = json.load(f)
            with config_path.open("r", encoding="utf-8") as f:
                action_config = json.load(f)
        except (json.JSONDecodeError, OSError) as error:
            print(f"Skipping unreadable task directory {task_dir}: {error}")
            continue

        trajectories.append({
            "action": action_history,
            "config": action_config,
        })

    return trajectories


def replace_names_recursive(data, random_name_alice, random_name_bob):
    if isinstance(data, dict):
        return {
            key: replace_names_recursive(value, random_name_alice, random_name_bob)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [
            replace_names_recursive(item, random_name_alice, random_name_bob)
            for item in data
        ]
    elif isinstance(data, str):
        data = data.replace("Alice", random_name_alice)
        data = data.replace("alice", random_name_alice.lower())
        data = data.replace("Bob", random_name_bob)
        data = data.replace("bob", random_name_bob.lower())
        return data
    else:
        return data


def replace_player_names(trajectory):
    random_name_alice = get_first_name()
    random_name_bob = get_first_name()
    while random_name_bob == random_name_alice:
        random_name_bob = get_first_name()

    return replace_names_recursive(trajectory, random_name_alice, random_name_bob)


def is_similar_action(action1, action2) -> bool:
    """Check whether two actions differ only in emotion or murmur fields."""
    if not action1 or not action2:
        return False

    tool1, tool2 = action1["action"]["tool"], action2["action"]["tool"]
    if tool1 != tool2:
        return False

    ignored_fields = {"emotion", "murmur"}
    ti1 = {
        key: value
        for key, value in action1["action"]["tool_input"].items()
        if key not in ignored_fields
    }
    ti2 = {
        key: value
        for key, value in action2["action"]["tool_input"].items()
        if key not in ignored_fields
    }

    return ti1 == ti2


def filter_repeated_actions(trajectories):
    filtered_actions = []

    for task in trajectories:
        task_list = task["action"]
        for chain in task_list:
            input_, action_list, final_answer = chain['input'], chain['action_list'], chain['final_answer']
            filtered_action_list = []
            prev_action = None
            for action in action_list:
                obs = action['feedback']
                if not isinstance(obs, dict):
                    continue

                if prev_action and is_similar_action(prev_action, action):
                    if obs.get("status", False):
                        filtered_action_list[-1] = action
                        prev_action = action
                    else:
                        if not filtered_action_list[-1]["feedback"].get("status", False):
                            filtered_action_list[-1] = action
                            prev_action = action
                else:
                    filtered_action_list.append(action)
                    prev_action = action

            new_action_history = {
                "input": input_,
                "action_list": filtered_action_list,
                "final_answer": final_answer,
            }
            filtered_actions.append(new_action_history)

    return filtered_actions


def format_string(template: str, data: dict) -> str:
    keys = re.findall(r'{{(.*?)}}', template)
    for key in keys:
        if key not in data:
            raise ValueError(f"Missing template value for {key!r}.")

    for key, value in data.items():
        template = template.replace('{{' + key + '}}', str(value))
    return template


def build_sft_records(filtered_list):
    final_dataset = []
    for chain in filtered_list:
        first_action = True
        action_list = chain.get("action_list")
        if not isinstance(action_list, list):
            raise ValueError("Each trajectory chain must contain an action_list.")
        if not action_list:
            continue

        shuffled_tool_list, shuffled_order = shuffle_tool_list()
        prompt_values = {
            "tool_list": shuffled_tool_list,
            "tool_order": shuffled_order,
        }
        input_str = format_string(LANG_CHAIN_PROMPT, prompt_values) + chain["input"]
        for action in action_list:
            final_dataset.append(
                {
                    "input": input_str,
                    "output": action["action"]["log"],
                }
            )
            if first_action:
                input_str += "\n\nThis was your previous work (but I haven't seen any of it! I only see what you return as final answer):\n"
                first_action = False
            input_str += action["action"]["log"] + "\n"
            input_str += "Observation: " + str(action["feedback"]) + "\nThought:"
        final_ans = {
            "action": "Final Answer",
            "action_input": chain["final_answer"],
        }
        final_dataset.append(
            {
                "input": input_str,
                "output": f"Thought: {chain['final_answer']}\n\nAction: \n```\n{json.dumps(final_ans, indent=2, ensure_ascii=False)}\n```\n",
            }
        )

    return final_dataset


if __name__ == "__main__":
    trajectories = load_successful_trajectories()
    trajectories = [replace_player_names(trajectory) for trajectory in trajectories]

    TRAJECTORY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAJECTORY_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)

    filtered_actions = filter_repeated_actions(trajectories)
    sft_dataset = build_sft_records(filtered_actions)
    SFT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SFT_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, indent=2, ensure_ascii=False)

    print(
        f"Wrote {len(trajectories)} trajectories to {TRAJECTORY_OUTPUT_PATH} "
        f"and {len(sft_dataset)} SFT records to {SFT_OUTPUT_PATH}"
    )
