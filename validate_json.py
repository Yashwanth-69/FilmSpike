import json
from collections import Counter

# Load the fingerprint
with open('sample_fingerprint.json', 'r') as f:
    data = json.load(f)

print("=== Basic Info ===")
print(f"Film: {data['video_info']['filename']}")
print(f"Duration: {data['video_info']['duration']:.2f}s")
print(f"Shots detected: {data['video_info']['shot_count']}")

print("\n=== Shot 0 Zero-Shot Concepts ===")
shot0 = data['shots'][0]
if 'zero_shot_concepts' in shot0 and shot0['zero_shot_concepts']:
    zs = shot0['zero_shot_concepts']
    print("Primary concepts:")
    for pc in zs.get('primary_concepts', []):
        print(f"  - {pc['concept']}: {pc['confidence']:.3f}")
    if 'concept_groups' in zs:
        print("Concept groups:", list(zs['concept_groups'].keys()))
else:
    print("No zero-shot data for this shot.")

print("\n=== Overall Zero-Shot Concept Summary ===")
# Count concept occurrences across all shots
concept_counter = Counter()
concept_confidence_sum = {}
concept_confidence_count = {}

for shot in data['shots']:
    if 'zero_shot_concepts' in shot:
        zs = shot['zero_shot_concepts']
        for pc in zs.get('primary_concepts', []):
            concept = pc['concept']
            conf = pc['confidence']
            concept_counter[concept] += 1
            if concept not in concept_confidence_sum:
                concept_confidence_sum[concept] = 0
                concept_confidence_count[concept] = 0
            concept_confidence_sum[concept] += conf
            concept_confidence_count[concept] += 1

# Compute average confidence for concepts that appear
concept_avg_conf = {c: concept_confidence_sum[c]/concept_confidence_count[c] 
                    for c in concept_confidence_count}

# Show top 10 concepts by frequency
print("Top 10 concepts by shot frequency:")
for concept, freq in concept_counter.most_common(10):
    avg_conf = concept_avg_conf.get(concept, 0)
    print(f"  - {concept}: appears in {freq} shots (avg conf: {avg_conf:.3f})")

print("\n=== Concept Group Distribution ===")
group_counter = Counter()
for shot in data['shots']:
    if 'zero_shot_concepts' in shot:
        groups = shot['zero_shot_concepts'].get('concept_groups', {})
        for group in groups.keys():
            group_counter[group] += 1

for group, count in group_counter.most_common():
    print(f"  - {group}: appears in {count} shots")

print("\n=== Original Summary ===")
print(data['summary'])