# TASK
Generate an original $genre script formatted as JSON for a $duration-second $style video generator.$direction

Title: "$title"

$rules

Exclusions -- do not reuse species, roles, or tropes from these previous scripts:
$exclusions

## OUTPUT FORMAT
Return ONLY a valid JSON object following this exact schema. Use escaped
newlines (\n) within string values to maintain valid JSON formatting:
{
  "title": "$title",
  "premise": "Brief summary of comedic premise",
  "positive_prompt": "[Scene Setup]: <environment/character description>\n\n[Timeline & Audio Sync]:\n00:xx-00:xx:\n- Video: <visuals>\n- Audio: <dialogue/sfx>",
  "negative_prompt": "$negative_baseline, <scene specific negative prompts>",
  "description": "Short overview",
  "tags": "comma, separated, tags",
  "fml2v_keyframe_prompts": {
    "first": "<Full visual setup>",
    "middle": "<Delta visual change>",
    "last": "<Delta visual change>"
  }
}
