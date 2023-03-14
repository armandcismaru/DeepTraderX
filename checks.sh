# Runs all code checks and formatting on the project

black $(find . -name "*.py" | xargs)
pylint $(find . -name "*.py" | xargs)
