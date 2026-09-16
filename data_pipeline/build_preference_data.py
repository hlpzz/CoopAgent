import difflib
import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

from .prompt_templates import CREATE_NOISE_PROMPT, EDIT_NOISE_PROMPT, LANG_CHAIN_PROMPT


# Replace these path placeholders before running the pipeline.
VILLAGERAGENT_ROOT = Path("path/to/VillagerAgent")
BLOCKS_PATH = VILLAGERAGENT_ROOT / "data" / "blocks.json"
INPUT_DATA_PATH = Path("path/to/trajectory_data.json")
OUTPUT_DATA_PATH = Path("path/to/preference_dataset.json")
NOISE_CACHE_PATH = Path("path/to/noise_cache.json")

# OpenAI-compatible API configuration. The API key is read from the environment.
API_KEY_ENV_VAR = "DASHSCOPE_API_KEY"
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PERTURBATION_MODEL = "qwen3-next-80b-a3b-instruct"

PERTURBATION_MODE = "task_related"  # "task_related" or "random"
CACHE_SAVE_INTERVAL = 5
MIN_PROMPT_SIMILARITY = 0.96
MAX_PROMPT_SIMILARITY = 0.9997
MAX_PERTURBATION_RETRIES = 10

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

universe_tool = ["scanNearbyEntities", "navigateTo", "withdrawItem", "fetchContainerContents", "openContainer"]
specialize_tool = {
    "craft": ["craftBlock"],
    "dig": ["MineBlock", "equipItem"],
    "attack": ["attackTarget", "equipItem"],
    "bed": ["sleep", "wake"],
    "boat": ["placeBlock", "useItemOnBlock", "mountEntity", "dismountEntity"],
    "bone_meal": ["useItemOnEntity", "useItemOnBlock", "placeBlock"],
    "chat": ["talkTo", "waitForFeedback"],
    "cook": ["SmeltingCooking"],
    "feed": ["useItemOnEntity"],
    "fishing": ["startFishing", "equipItem"],
    "handover": ["handoverBlock"],
    "milk": ["useItemOnEntity"],
    "minecart": ["useItemOnBlock", "placeBlock", "mountEntity", "dismountEntity"],
    "saddle": ["useItemOnEntity", "mountEntity", "dismountEntity"],
    "shear": ["useItemOnEntity"],
    "sign": ["read"],
    "store": ["storeItem"],
    "till": ["useItemOnBlock", "placeBlock"],
    "toggle": ["ToggleAction", "placeBlock", "useItemOnBlock"],
    "water": ["useItemOnBlock"],
    "move": ["placeBlock", "MineBlock", "equipItem"],
    "place": ["placeBlock"],
    "useitem": ["equipItem"]
}


def string_similarity(a: str, b: str) -> float:
    """
    Compare the similarity between two strings using Levenshtein-like ratio.

    Returns:
        float: similarity score between 0.0 and 1.0
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV_VAR} before generating perturbations.")

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=api_key,
    )


def random_noise() -> dict:
    with BLOCKS_PATH.open("r", encoding="utf-8") as f:
        block_list = json.load(f)
    noise_block = random.choice(block_list)["name"]
    noise_json = {
        "noise_type": "add",
        "noise_coordinate": [
            random.randint(0, 25),
            random.randint(-61, -57),
            random.randint(0, 25),
        ],
        "noise_block": noise_block,
    }
    return noise_json


def step1_generate_noise(orig_prompt: str) -> str:
    noise_type = random.choices(["add", "remove"], [4, 1])[0]
    prompt = format_string(CREATE_NOISE_PROMPT, {"orig_prompt": orig_prompt, "noise_type": noise_type})
    res = get_client().chat.completions.create(
        model=PERTURBATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        top_p=0.9,
    )
    return res.choices[0].message.content


def step2_apply_noise(orig_prompt: str, noise_json: str) -> str:
    prompt = format_string(EDIT_NOISE_PROMPT, {"orig_prompt": orig_prompt, "noise_json": noise_json})
    res = get_client().chat.completions.create(
        model=PERTURBATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return res.choices[0].message.content


def add_noise(orig_prompt: str) -> str:
    if PERTURBATION_MODE == "task_related":
        noise_json = step1_generate_noise(orig_prompt)
    elif PERTURBATION_MODE == "random":
        noise_json = json.dumps(random_noise(), ensure_ascii=False)
    else:
        raise ValueError(
            f"Unsupported PERTURBATION_MODE={PERTURBATION_MODE!r}; "
            "expected 'task_related' or 'random'."
        )

    return step2_apply_noise(orig_prompt, noise_json)


def shuffle_tool_list() -> Tuple[str, str]:
    # Split the tool description into individual entries.
    tool_lines = [line.strip() for line in TOOL_LIST.split('\n') if line.strip()]

    # Retain both the tool name and its complete description.
    tools = []
    for line in tool_lines:
        tool_name = line.split(':', 1)[0].strip()
        tools.append((tool_name, line))

    random.shuffle(tools)

    shuffled_tool_list = '\n'.join([tool[1] for tool in tools])
    shuffled_order = ', '.join([tool[0] for tool in tools])

    return shuffled_tool_list, shuffled_order

def format_string(template: str, data: dict) -> str:
    keys = re.findall(r'{{(.*?)}}', template)
    for key in keys:
        if key not in data:
            raise ValueError(f"Missing template value for {key!r}.")

    for key, value in data.items():
        template = template.replace('{{' + key + '}}', str(value))
    return template


def load_noise_cache() -> List[str]:
    if not NOISE_CACHE_PATH.exists():
        return []

    with NOISE_CACHE_PATH.open("r", encoding="utf-8") as f:
        noise_cache = json.load(f)
    if not isinstance(noise_cache, list):
        raise ValueError(f"Noise cache must be a JSON list: {NOISE_CACHE_PATH}")
    return noise_cache


def save_noise_cache(noise_cache: List[str]) -> None:
    NOISE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOISE_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(noise_cache, f, indent=2, ensure_ascii=False)


def build_preference_records(trajectories: List[Dict]) -> List[Dict]:
    final_dataset = []
    t = 0
    noise_cache = load_noise_cache()
    cache_changed = False

    for task in trajectories:
        task_list = task["action"]
        config = task["config"]
        for chain in task_list:
            input_, action_list = chain['input'], chain['action_list']
            regenerate_noise = t >= len(noise_cache)
            if not regenerate_noise:
                noise_input = noise_cache[t]
                regenerate_noise = (
                    not isinstance(noise_input, str)
                    or abs(len(noise_input) - len(input_)) < 5
                )

            if regenerate_noise:
                noise_input = add_noise(input_)
                similarity = string_similarity(input_, noise_input)
                retry_time = 0
                while similarity > MAX_PROMPT_SIMILARITY or similarity < MIN_PROMPT_SIMILARITY:
                    retry_time += 1
                    if retry_time >= MAX_PERTURBATION_RETRIES:
                        raise RuntimeError(f"Too many retries at t={t}, similarity={similarity}")
                    noise_input = add_noise(input_)
                    similarity = string_similarity(input_, noise_input)

                if t < len(noise_cache):
                    noise_cache[t] = noise_input
                else:
                    noise_cache.append(noise_input)
                cache_changed = True

                if (t + 1) % CACHE_SAVE_INTERVAL == 0:
                    save_noise_cache(noise_cache)
                    cache_changed = False

            t += 1

            shuffled_tool_list, shuffled_order = shuffle_tool_list()
            prompt_values = {
                "tool_list": shuffled_tool_list,
                "tool_order": shuffled_order,
            }
            prompt_prefix = format_string(LANG_CHAIN_PROMPT, prompt_values)
            input_str = prompt_prefix + chain["input"]
            noise_str = prompt_prefix + noise_input
            chain_len = len(action_list)
            tag = []
            first_action = True
            for action in action_list:
                status = action["feedback"]["status"]
                if isinstance(status, str):
                    status = status.lower() == "true"  # 转换字符串 "True"/"False" 为布尔值
                task_type = config["task_scenario"] if config["task_scenario"] != "interact" else config["evaluation_arg"]["action"]
                tool_used = action["action"]["tool"]
                tag.append((tool_used in universe_tool or tool_used in specialize_tool[task_type]) and status)
            for i in range(chain_len):
                action = action_list[i]
                if not tag[i]:  # Rejected action
                    win_found = False
                    for j in range(i+1, chain_len):
                        if tag[j]:
                            win_found = True
                            win_action = action_list[j]
                            final_dataset.append(
                                {
                                    "prompt": input_str,
                                    "noise_prompt": noise_str,
                                    "chosen": win_action["action"]["log"],
                                    "rejected": action["action"]["log"]
                                }
                            )
                            break
                    if not win_found:
                        final_ans = {
                            "action": "Final Answer",
                            "action_input": chain["final_answer"]
                        }
                        final_dataset.append(
                            {
                                "prompt": input_str,
                                "noise_prompt": noise_str,
                                "chosen": f"Thought: {chain['final_answer']}\n\nAction: \n```\n{json.dumps(final_ans, indent=2, ensure_ascii=False)}\n```\n",
                                "rejected": action["action"]["log"]
                            }
                        )
                if first_action:
                    input_str += "\n\nThis was your previous work (but I haven't seen any of it! I only see what you return as final answer):\n"
                    noise_str += "\n\nThis was your previous work (but I haven't seen any of it! I only see what you return as final answer):\n"
                    first_action = False
                input_str += action["action"]["log"] + "\n"
                noise_str += action["action"]["log"] + "\n"

                input_str += "Observation: " + str(action["feedback"])  + "\nThought:"
                noise_str += "Observation: " + str(action["feedback"])  + "\nThought:"

    if cache_changed:
        save_noise_cache(noise_cache)

    return final_dataset


if __name__ == "__main__":
    with INPUT_DATA_PATH.open("r", encoding="utf-8") as f:
        trajectories = json.load(f)

    final_dataset = build_preference_records(trajectories)
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(final_dataset)} preference pairs to {OUTPUT_DATA_PATH}")
