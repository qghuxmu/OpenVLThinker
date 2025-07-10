# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from typing import Dict

from mathruler.grader import grade_answer


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        final_reason_punishment = 0.0
        if "</think>" in predict_str and "<answer>" in predict_str:
            final_reason = predict_str.split("</think>")[-1].split("<answer>")[0].strip()
            if not final_reason: final_reason_punishment = 0.5
        
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()

        if grade_answer(given_answer, ground_truth.strip()) or given_answer.lower() == ground_truth.lower():
            return 1.0 - final_reason_punishment
        else:
            return 0.0

    except Exception:
        pass

    return 0.0


def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.5) -> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward(predict_str, ground_truth)
    return {
        "overall": accuracy_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }


if __name__ == "__main__":
    predict_str = """<think>
Okay, let me see. The question is asking for the difference in value between Magenta and Web Purple based on the bar chart described. 

First, I need to recall the details given about each bar. The Magenta bar goes up to 65 on the x-axis, and the Web Purple bar reaches 50. So, the x-axis is the value here, right? The y-axis is the categories, so the x-axis values are the actual numbers. 

To find the difference, I subtract the smaller value from the larger one. So that would be 65 minus 50. Let me do that calculation: 65 - 50 equals 15.
</think>

The Magenta bar reaches 65 on the x-axis, while the Web Purple bar reaches 50. The difference is calculated by subtracting the smaller value from the larger one: 65 − 50 = 15. 

<answer>15</answer>"""
    ground_truth = "15"
    score = compute_score(predict_str, ground_truth)
    print(score)