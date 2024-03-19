from clearml import Task

task = Task.init(project_name='first ClearML steps', task_name='hello')

print('hello ClearML')