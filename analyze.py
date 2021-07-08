import sys

count_pattern_by_size = {}
pattern_list = {}
pid = 0

with open(sys.argv[1], 'r', encoding='utf-8') as file:
    pattern_size = 0

    for line in file:
        if "Saving plots" in line:
            plot_name = line.split("/")[-1]
            pattern_size = int(plot_name.split("-")[0])

            if pattern_size not in count_pattern_by_size:
                count_pattern_by_size[pattern_size] = 1
            else:
                count_pattern_by_size[pattern_size] += 1

        if "{" in line and "}" in line:
            pattern_list[pid] = (pattern_size, [int(x) for x in line[1:-2].split(", ")])
            pid += 1

print("Count pattern")
for k, v in count_pattern_by_size.items():
    print(k, v)

print("Pattern:")
for k, v in pattern_list.items():
    print(v[0])
