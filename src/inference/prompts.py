"""Prompts used by the TEMPURA inference and benchmark-evaluation scripts.

``<your_query>`` is replaced by the natural-language query of a sample. The benchmark prompts are kept
byte-identical (including whitespace) to the ones used for the paper results.
"""

DVC = "\n            Partition and identify events by dividing the video into a series of non-overlapping segments,\n            determining the start and end time for each event, and arranging them in chronological order to ensure complete coverage of all video frames.\n            Each event should be accompanied by a detailed description. Please follow the format:\n            'From <start time1> to <end time1>, <detailed description1>.\n            \n\nFrom <start time2> to <end time2>, <detailed description2>.\n\n...'\n            "

MEP = (
    "Given the input video with the masked segment from <start> to <end>, \n please step-by-step reason and "
    "predict what might happen in the masked time segment based on the context of the video and give your "
    "answer in the following format: <think>[reasoning steps]</think><answer>So the description of the "
    "predicted event from <start> to <end> is: <predicted masked event description></answer>"
)

# Video temporal grounding (Charades-STA), zero-shot JSON prompt.
VTG = 'You are a highly capable AI assistant trained to analyze video content and accurately identify the relevant time intervals in response to natural language queries.\n\n            ### **Task Description**\n            Your goal is to perform **video temporal grounding** by identifying the start and end timestamps in a video that match a given user query.\n\n            ### **Guidelines**\n            1. **Comprehend Video Content**: Analyze the sequence of frames to understand the event described in the query.\n            2. **Understand Temporal Dynamics**: Consider motion, object interactions, and scene transitions to determine when the event occurs.\n            3. **Handle Multiple Occurrences**: The same query may appear **multiple times** throughout the video. Identify and return all relevant time intervals.\n            4. **Output Format**: Provide the results in **JSON format** as specified below.\n\n            ### **Expected Output JSON Format**\n            { "query": "<user query>", "relevant_windows": [ [<start time>, <end time>], [<start time>, <end time>], ... ] }\n            User Query: <your_query>\n            '

# Short instruction used by checkpoints fine-tuned on Charades-STA.
VTG_FT = 'Find the video segment that corresponds to the given textual query <your_query> and determine its start and end seconds.'

# Video highlight detection (QVHighlights): timestamps (2-second clips) + saliency scores.
VHD_FT = "You are given a video from the QVHighlights dataset. Please find the highlight contents in the video described by a sentence query, determining the highlight timestamps and its saliency score on a scale from 1 to 5. The output format should be like: 'The highlight timestamps are in the 82, 84, 86, 88, 90, 92, 94, 96, 98, 100 seconds. Their saliency scores are 1.3, 1.7, 1.7, 1.7, 1.7, 1.3, 1.7, 2.3, 2.3, 2.3'. Now I will give you the sentence query: <your_query>. Please return the query-based highlight timestamps and salient scores."

# Text-only refinement of VTG windows with the dense video caption of the same video.
VTG_REFINE = """You are an intelligent video understanding assistant. I need your help to refine video temporal grounding results.

I have a video with the following dense caption description of its content:
{dense_caption}

For the query: "{query}"

1. Find the most relevant time window for the query from the dense caption description.
2. If you cannot find it, use the initial system identified time windows:
{windows}
3. Merge the time windows if they are the same or close event for the query.
4. IMPORTANT: Each time window MUST be at least {min_window_duration} seconds long. If a relevant action is shorter, extend the window to include more context.
5. Do not output empty relevant_windows.

Please output your answer in the following JSON format:
{{
    "relevant_windows": [[start_time, end_time]],
}}
"""

# Text-only refinement of highlight detection with the dense video caption of the same video.
VHD_REFINE = """You are an intelligent video understanding assistant. I need your help to refine video highlight detection results.

I have a {duration:.0f}-second video with the following dense caption description of its content:
{dense_caption}

The user query is: "{query}"

An initial system produced this answer:
{initial_answer}

Using the dense caption, decide which 2-second clips of the video match the query. List their timestamps (even numbers of seconds between 0 and {duration:.0f}) and give each a saliency score from 1 to 5 (higher = more relevant). Keep the initial answer when the caption does not contradict it.

Answer in exactly this format and nothing else:
'The highlight timestamps are in the <t1>, <t2>, ... seconds. Their saliency scores are <s1>, <s2>, ...'
"""

PROMPTS = {"dvc": DVC, "mep": MEP, "vtg": VTG, "vtg_ft": VTG_FT, "vhd_ft": VHD_FT}

TIME_INSTRUCTION = ("You are given a video sampled at {fps} frame per second, so input video frames are sampled at "
                    "{time_points} second sequentially, so in total the video is {duration:.2f} seconds long.\n ")


def time_instruction(fps: float, timestamps) -> str:
    return TIME_INSTRUCTION.format(fps=fps, time_points=", ".join(f"{t:.2f}" for t in timestamps),
                                   duration=timestamps[-1] + 1 if timestamps else 0.0)
