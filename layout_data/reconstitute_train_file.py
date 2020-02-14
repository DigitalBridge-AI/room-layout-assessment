"""
Github limits files to 100MB. To get around this we have split the file into 3.
This scripts reforms the 3 parts in to a parent file.
"""

with open('./train_projects.json', 'w') as output:
    for i in range(3):
        with open('./train_projects_{}.json'.format(i), 'r') as input:
            output.write(input.read())
