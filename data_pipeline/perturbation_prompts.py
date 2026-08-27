NOISE_PROMPT = '''
Use the RANDOM_SEED value as high-level randomness to produce different noise every time.
You are a noise injector responsible for adding perturbations to a given prompt. You will receive a prompt intended for an Agent performing tasks in the game Minecraft. This prompt contains information about the task, the agent state, and the environment. Your job is to precisely identify the section that describes the environment information, and inject noise into that section only, while keeping all other parts of the original prompt exactly as they are.
The noise must be added according to the following rules:

1. Identify the block entities that the task may involve (if none can be found, default to “dirt”). Then randomly select one block **B** from the set {“grass_block”, “dirt”, the blocks you identified}.
2. The environment information includes coordinates of certain blocks. Select one of these coordinates, and choose a nearby coordinate **t** within ±2 blocks of it.
3. Insert the following sentence into the original environment description:
   **“There is a block B at coordinate t.”**

Keep all other content unchanged and return only the prompt with the injected noise.

*** Important Notice ***
1. Ensure the chosen coordinate **t** is not far from the coordinates originally provided.
2. The inserted noise must appear within the environment description section, and the writing style must match that of the original prompt.
3. Aside from inserting the noise, do not alter any other characters—no additions, deletions, or modifications.
4. Output only the original prompt with the injected noise, and nothing else.

The original prompt is as below:


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