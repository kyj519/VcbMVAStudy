# Read the contents of the text file
with open('./infer_log/error_list', 'r') as file:
    lines = file.readlines()

# Sort the lines based on the first letter of each line
# Sort the lines based on the folders in the file paths
import os
sorted_lines = sorted(lines, key=lambda x: os.path.dirname(x.strip()))
# Write the sorted lines back to the text file
with open('./infer_log/error_list_sorted', 'w') as file:
    file.writelines(sorted_lines)
