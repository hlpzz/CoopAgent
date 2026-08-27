import json
import os
import random
from names import get_first_name
import re
from openai import OpenAI
from .perturbation_prompts import NOISE_PROMPT
import difflib

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


def add_noise(orig_propmt):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY before generating perturbations.")

    client = OpenAI(
        base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key
    )
    input_str = NOISE_PROMPT + orig_propmt
    response = client.chat.completions.create(
        model = "qwen3-next-80b-a3b-instruct",
        messages=[
            {"role": "user", "content": input_str}
        ],
        temperature=0.7,
        top_p=0.9
    )
    noise_prompt = response.choices[0].message.content
    return noise_prompt


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


def raise_history_files():
    # Get current directory
    current_dir = os.getcwd()
    
    # Track deleted folders
    action_history_files = []
    
    # Iterate through all folders in current directory
    for folder in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            score_file = os.path.join(folder_path, 'score.json')
            action_history_file = os.path.join(folder_path, 'Alice_history.json')
            config_file = os.path.join(folder_path, 'config.json')
            # Check if score file exists
            if os.path.exists(score_file) and os.path.exists(action_history_file) and os.path.exists(config_file):
                try:
                    with open(score_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Check score value
                    if not data.get('score') or data.get('score') < 50:
                        continue
                    else:
                        with open(action_history_file, 'r', encoding='utf-8') as f:
                            action_history = json.load(f)
                        with open(config_file, 'r', encoding='utf-8') as f:
                            action_config = json.load(f)
                        action_history_files.append({
                            "action": action_history,
                            "config": action_config
                            })
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading {score_file}: {e}")

    return action_history_files

def replace_names_recursive(data, random_name_alice, random_name_bob):
    if isinstance(data, dict):
        # 如果是字典，递归处理每个值
        return {key: replace_names_recursive(value, random_name_alice, random_name_bob) 
                for key, value in data.items()}
    elif isinstance(data, list):
        # 如果是列表，递归处理每个元素
        return [replace_names_recursive(item, random_name_alice, random_name_bob) 
                for item in data]
    elif isinstance(data, str):
        # 如果是字符串，执行替换
        data = data.replace("Alice", random_name_alice)
        data = data.replace("alice", random_name_alice.lower())
        data = data.replace("Bob", random_name_bob)
        data = data.replace("bob", random_name_bob.lower())
        return data
    else:
        # 其他类型（数字、布尔值等）直接返回
        return data

def replace_names_in_action_history(action_history_files):
    # 生成两个不同的随机名字
    random_name_alice = get_first_name()
    random_name_bob = get_first_name()
    while random_name_bob == random_name_alice:
        random_name_bob = get_first_name()
    
    # 递归处理整个数据结构
    return replace_names_recursive(action_history_files, random_name_alice, random_name_bob)


def is_similar_action(action1, action2) -> bool:
    """
    判断两个动作是否相似：
    - tool 必须相同
    - tool_input 里除了 emotion 和 murmur 外，其余字段完全一致
    """
    if not action1 or not action2:
        return False
    
    tool1, tool2 = action1["action"]["tool"], action2["action"]["tool"]
    if tool1 != tool2:
        return False
    
    # 提取核心字段（排除 emotion 和 murmur）
    ti1 = {k: v for k, v in action1["action"]["tool_input"].items() if k not in ["emotion", "murmur"]}
    ti2 = {k: v for k, v in action2["action"]["tool_input"].items() if k not in ["emotion", "murmur"]}
    
    return ti1 == ti2


def filter_action(new_name_files):
    filtered_actions = []

    for task in new_name_files:
        task_list = task["action"]
        config = task["config"]
        for chain in task_list:
            input_, action_list, final_answer = chain['input'], chain['action_list'], chain['final_answer']
            filtered_action_list = []
            prev_action = None  
            for action in action_list:
                obs = action['feedback']
                if type(obs) != dict:
                    continue

                if prev_action and is_similar_action(prev_action, action):
                    # 如果重复，优先保留 status=True 的最后一个
                    if obs.get("status", False):
                        filtered_action_list[-1] = action
                        prev_action = action
                    else:
                        if not filtered_action_list[-1]["feedback"].get("status", False):
                            # 前一个也是失败 → 替换掉，保留最后一个 
                            filtered_action_list[-1] = action
                            prev_action = action
                        # 如果前一个是成功，就不动，仍然保留成功的
                else:
                    filtered_action_list.append(action)
                    prev_action = action

            # refiltered_action_list = []
            # for action in filtered_action_list:
            #     task_type = config["task_scenario"] if config["task_scenario"] != "interact" else config["evaluation_arg"]["action"]
            #     tool_used = action["action"]["tool"]
            #     tool_input = action["action"]["tool_input"]
            #     action_res = action["feedback"]["status"]

            #     if tool_used not in universe_tool and tool_used not in specialize_tool[task_type]:
            #         continue
            #     if (tool_used == "craftBlock" or tool_used == "placeBlock") and (tool_input["item_name"] == "crafting_table" or tool_input["item_name"] == "furnace") and not action_res:
            #         continue
            #     refiltered_action_list.append(action)

            new_action_history = {
                'input': input_,
                'action_list': filtered_action_list,
                'final_answer': final_answer
            }
            filtered_actions.append(new_action_history)

    return filtered_actions


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

def convert_format(filtered_list):
    final_dataset = []
    for chain in filtered_list:
        first_action = True
        try:
            if len(chain["action_list"]) < 1:
                continue
        except Exception as e:
            print(len(chain))


        shuffled_tool_list, shuffled_order = shuffle_tool_list()
        input_str = format_string(LANG_CHAIN_PROMPT, {"tool_list": shuffled_tool_list, "tool_order": shuffled_order}) + chain["input"]
        for action in chain["action_list"]:
            final_dataset.append(
                {
                    "input": input_str,
                    "output": action["action"]["log"]
                }
            )
            if first_action:
                input_str += "\n\nThis was your previous work (but I haven't seen any of it! I only see what you return as final answer):\n"
                first_action =  False
            input_str += action["action"]["log"] + "\n"

            input_str += "Observation: " + str(action["feedback"])  + "\nThought:"
        final_ans = {
            "action": "Final Answer",
            "action_input": chain["final_answer"]
        }
        final_dataset.append(
            {
                "input": input_str,
                "output": f"Thought: {chain['final_answer']}\n\nAction: \n```\n{json.dumps(final_ans, indent=2, ensure_ascii=False)}\n```\n"
            }
        )
    
    return final_dataset


def dpo_convert(new_name_files):
    final_dataset = []
    t = 0
    for task in new_name_files:
        task_list = task["action"]
        config = task["config"]
        for chain in task_list:
            input_, action_list, final_answer = chain['input'], chain['action_list'], chain['final_answer']

            noise_input = add_noise(input_)

            # output_txt = input_ + "=" * 120 + noise_input
            # Write temporary inspection output under the repository's tmp/ directory.
            #     f.write(output_txt)
            # print("ok")
            # while 1:
            #     pass
            t += 1
            similarity = string_similarity(input_, noise_input)
            while similarity == 1.0 or similarity < 0.97:
                noise_input = add_noise(input_)
                similarity = string_similarity(input_, noise_input)
                print(f"{t}: {similarity} -- retry")
            print(f"{t}: {similarity}")

            shuffled_tool_list, shuffled_order = shuffle_tool_list()
            input_str = format_string(LANG_CHAIN_PROMPT, {"tool_list": shuffled_tool_list, "tool_order": shuffled_order}) + chain["input"]
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
                                    "noise_prompt": noise_input,
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
                                "noise_prompt": noise_input,
                                "chosen": f"Thought: {chain['final_answer']}\n\nAction: \n```\n{json.dumps(final_ans, indent=2, ensure_ascii=False)}\n```\n",
                                "rejected": action["action"]["log"]
                            }
                        )
                if first_action:
                    input_str += "\n\nThis was your previous work (but I haven't seen any of it! I only see what you return as final answer):\n"
                    first_action =  False
                input_str += action["action"]["log"] + "\n"
                
                input_str += "Observation: " + str(action["feedback"])  + "\nThought:"

    return final_dataset

if __name__ == "__main__":
    action_history_files = raise_history_files()
    new_name_files = [replace_names_in_action_history(task_history) for task_history in action_history_files]
    # filtered_list = filter_action(new_name_files)
    # final_dataset = convert_format(filtered_list)
    # with open("aligned_action.json", "w", encoding="utf-8") as f:
    #     json.dump(final_dataset, f, indent=4)
    # print(len(final_dataset))
    final_dataset = dpo_convert(new_name_files)
    with open("dpo_dataset.json", "w", encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4)
    print(len(final_dataset))
