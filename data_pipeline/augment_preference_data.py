import json
import os
import random
from pathlib import Path
from names import get_first_name
import re
from openai import OpenAI
from .perturbation_prompts import NOISE_PROMPT, EDIT_NOISE_PROMPT, CREATE_NOISE_PROMPT
import difflib


REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = REPO_ROOT / "tmp"

LANG_CHAIN_PROMPT = '''System: Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{{tool_list}}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {{tool_order}}

Provide only ONE action per $JSON_BLOB, as shown:

```
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{
  "action": "Final Answer",
  "action_input": "Final response to human"
}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
Thought:
Human: 
'''

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


def get_client():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY before generating perturbations.")

    return OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )


def write_debug_pair(original_prompt: str, perturbed_prompt: str):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    (TMP_DIR / "orig_output.txt").write_text(original_prompt, encoding="utf-8")
    (TMP_DIR / "output.txt").write_text(perturbed_prompt, encoding="utf-8")

'''
{
  "noise_type": "add" | "remove",
  "target_coordinate": [x, y, z],
  "noise_coordinate": [tx, ty, tz],   // only required if noise_type="add"
  "noise_block": "grass_block" | "dirt" | <block_from_task>
}
'''

def random_noise():
    villageragent_root = os.getenv("VILLAGERAGENT_ROOT")
    if not villageragent_root:
        raise RuntimeError(
            "Set VILLAGERAGENT_ROOT to the local VillagerAgent repository."
        )

    blocks_path = Path(villageragent_root) / "data" / "blocks.json"
    with blocks_path.open("r", encoding="utf-8") as f:
        block_list = json.load(f)
    noise_block = random.choice(block_list)["name"]
    noise_json = {
        "noise_type": "add",
        "noise_coordinate": [random.randint(0, 25), random.randint(-61, -57),random.randint(0, 25)],
        "noise_block": noise_block
    }
    return noise_json



def step1_generate_noise(orig_prompt: str):
    noise_type = random.choices(["add", "remove"], [4, 1])[0]
    prompt = format_string(CREATE_NOISE_PROMPT, {"orig_prompt": orig_prompt, "noise_type": noise_type})
    res = get_client().chat.completions.create(
        model="qwen3-next-80b-a3b-instruct",
        # model="qwen3-max",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        top_p=0.9
    )
    return res.choices[0].message.content

def step2_apply_noise(orig_prompt: str, noise_json: str):
    prompt = format_string(EDIT_NOISE_PROMPT, {"orig_prompt": orig_prompt, "noise_json": noise_json})
    res = get_client().chat.completions.create(
        model="qwen3-next-80b-a3b-instruct",
        # model="qwen3-max",
        messages=[{"role": "user", "content": prompt}],
    )

    return res.choices[0].message.content

def add_noise(orig_prompt: str):
    # noise_json = step1_generate_noise(orig_prompt)      # TASK RELATED NOISE
    noise_json = random_noise()    # RANDOM NOISE
    noise_str = json.dumps(noise_json)
    # with (TMP_DIR / "dict_format.json").open("w", encoding="utf-8") as f:
    #     json.dump(noise_str, f, indent=4)
    noisy_prompt = step2_apply_noise(orig_prompt, noise_str)
    return noisy_prompt


def shuffle_tool_list():
    # 分割原始字符串为单独的工具行
    tool_lines = [line.strip() for line in TOOL_LIST.split('\n') if line.strip()]
    
    # 提取工具名称和完整行
    tools = []
    for line in tool_lines:
        # 获取工具名称（第一个冒号前的部分）
        tool_name = line.split(':', 1)[0].strip()
        tools.append((tool_name, line))
    
    # 随机打乱工具顺序
    random.shuffle(tools)
    
    # 构建新的TOOL_LIST字符串和工具名称列表
    shuffled_tool_list = '\n'.join([tool[1] for tool in tools])
    shuffled_order = ', '.join([tool[0] for tool in tools])
    
    return shuffled_tool_list, shuffled_order



def format_string(template: str, data: dict) -> str:
    # 检查template中的{{}}是否都在data中
    keys = re.findall(r'{{(.*?)}}', template)
    for key in keys:
        if key not in data:
            raise ValueError(f'when format:\n{template} \nkey {key} not found in data')

    # 替换{{}}为data中的值
    for key, value in data.items():
        template = template.replace('{{' + key + '}}', str(value))
    return template

def dpo_convert(new_name_files):
    final_dataset = []
    t = 0
    with open ("temp_random_noise.json", "r", encoding='utf-8') as f:
    # with open ("temp_noise.json", "r", encoding='utf-8') as f:
        noise_temp = json.load(f)
    for task in new_name_files:
        task_list = task["action"]
        config = task["config"]
        for chain in task_list:
            input_, action_list, final_answer = chain['input'], chain['action_list'], chain['final_answer']
            fail_noise = False
            if t < len(noise_temp):
                noise_input = noise_temp[t]
                if abs(len(noise_input) - len(input_)) < 5:
                    fail_noise = True
            if t >= len(noise_temp) or fail_noise:
                noise_input = add_noise(input_)

                # output_txt = input_ + "=" * 120 + noise_input
                # with (TMP_DIR / "output.txt").open("w", encoding="utf-8") as f:
                #     f.write(output_txt)
                # print("ok")
                # while 1:
                #     pass
                similarity = string_similarity(input_, noise_input)
                retry_time = 0
                while similarity > 0.9997 or similarity < 0.96:
                    if similarity > 0.9997:
                        write_debug_pair(input_, noise_input)
                    print(f"{t}: {similarity} -- retry")
                    retry_time += 1
                    if retry_time >= 10:
                        raise RuntimeError(f"Too many retries at t={t}, similarity={similarity}")
                    noise_input = add_noise(input_)
                    similarity = string_similarity(input_, noise_input)
                print(f"{t}: {similarity}")
                noise_temp.append(noise_input)
                if t % 5 == 0:
                # if t == 1409:
                    with open("temp_random_noise.json", "w", encoding='utf-8') as f:
                    # with open("temp_noise.json", "w", encoding='utf-8') as f:
                        json.dump(noise_temp, f ,indent=4)

            t += 1

            shuffled_tool_list, shuffled_order = shuffle_tool_list()
            input_str = format_string(LANG_CHAIN_PROMPT, {"tool_list": shuffled_tool_list, "tool_order": shuffled_order}) + chain["input"]
            noise_str = format_string(LANG_CHAIN_PROMPT, {"tool_list": shuffled_tool_list, "tool_order": shuffled_order}) + noise_input
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
                if not tag[i]: # lose action
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
                    first_action =  False
                input_str += action["action"]["log"] + "\n"
                noise_str += action["action"]["log"] + "\n"
                
                input_str += "Observation: " + str(action["feedback"])  + "\nThought:"
                noise_str += "Observation: " + str(action["feedback"])  + "\nThought:"

    return final_dataset

def test(new_name_files):
    for task in new_name_files:
        task_list = task["action"]
        config = task["config"]
        for chain in task_list:
            input_, action_list, final_answer = chain['input'], chain['action_list'], chain['final_answer']
            noise_input = add_noise(input_)
            
            write_debug_pair(input_, noise_input)
            print(string_similarity(input_, noise_input))
            
            return

if __name__ == "__main__":
    with open("temp_output.json", "r", encoding='utf-8') as f:
        new_name_files = json.load(f)
    final_dataset = dpo_convert(new_name_files)
    with open("random_noise_dpo_dataset.json", "w", encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4)
    print(len(final_dataset))
    # test(new_name_files)
