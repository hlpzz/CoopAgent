"""Prompt templates used by the CoopAgent data builders."""


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


CREATE_NOISE_PROMPT = '''
You are a noise generator for Minecraft task prompts.

You will receive a full prompt that contains:
- task description
- agent state
- environment information

Your job is to analyze the environment information and produce ONE noise operation.
You MUST perform this noise type: {{noise_type}}
Your noise operation must be in the following JSON format:

{
  "noise_type": "add" | "remove",
  "target_coordinate": [x, y, z],
  "noise_coordinate": [tx, ty, tz],   // only required if noise_type="add"
  "noise_block": "grass_block" | "dirt" | <block_from_task>
}

Rules:
1. Identify all block types involved in the task. If none exist, use "dirt".
2. Identify all coordinates mentioned in the environment info.
3. Randomly choose one coordinate as the target.
4. If noise_type="add":
      Create a nearby coordinate (within ±2 blocks) and select a noise_block.
5. If noise_type="remove":
      The target coordinate represents the block to be removed.
6. Your output MUST be ONLY the JSON object, with no explanation.
7. Ensure that the generated noise does not interfere with task execution.

Now here is the original prompt:
==== INPUT PROMPT START ====
{{orig_prompt}}
==== INPUT PROMPT END ====

'''

EDIT_NOISE_PROMPT = '''
You are a precise text editor.

You will receive:
1. The original prompt.
2. A noise operation in JSON format.

Your job:
- Locate the environment description section in the original prompt.
- Apply the noise operation that does not affect the task execution EXACTLY as specified.

Guidelines:
1. If noise_type="add":
       Insert the sentence describing the noise block into the environment info.
       The inserted sentence must follow the writing style of the environment text.
       Insert it in a position that naturally fits the environment block list.

2. If noise_type="remove":
       Remove EXACTLY the environment entry corresponding to the "target_coordinate".

3. DO NOT modify or reorder any other text in the prompt.
4. DO NOT generate any additional explanations.
5. Return ONLY the modified full prompt.
6. Noise operations that may affect task execution should NOT be applied.

Noise operation to apply:
==== NOISE OP START ====
{{noise_json}}
==== NOISE OP END ====

Original prompt:
==== INPUT PROMPT START ====
{{orig_prompt}}
==== INPUT PROMPT END ====

'''
